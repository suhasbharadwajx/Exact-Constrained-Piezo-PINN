import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingLR
import time
import os
import sys
from physics import *
from model import SoftPenaltyPINN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float32)

worker_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
n_workers = int(sys.argv[2]) if len(sys.argv) > 2 else 1

print("BASELINE: SOFT-PENALTY PINN ON THE ASYMMETRIC MMS WAVEFIELD")
print(f"Using Compute Device: {device}")

N_colloc = 25000
N_bc = 5000
N_ic = 5000
N_sensors = 5000
T_final = 1.0
snr_db = 30
N_adam = 15000
N_lbfgs = 1500
N_eval_field = 20000
N_eval_bc = 4000
N_eval_pde = 5000

configs = {
    "A": None,
    "B": [1.0, 1.0, 5000.0, 5000.0, 5000.0, 5000.0]
}
seeds = [1, 2, 3, 4, 5]

def generate_data(seed):
    g = torch.Generator().manual_seed(seed)

    X_domain = torch.rand(N_colloc, 3, generator=g)
    X_domain[:, 2] *= T_final

    X_sensor = torch.rand(N_sensors, 3, generator=g)
    X_sensor[:, 1] = 0.9
    X_sensor[:, 2] *= T_final

    X_bc = torch.rand(N_bc, 3, generator=g)
    X_bc[:, 2] *= T_final
    face = torch.randint(0, 4, (N_bc,), generator=g)
    X_bc[face == 0, 0] = 0.0
    X_bc[face == 1, 0] = 1.0
    X_bc[face == 2, 1] = 0.0
    X_bc[face == 3, 1] = 1.0

    X_ic = torch.rand(N_ic, 3, generator=g)
    X_ic[:, 2] = 0.0

    clean = torch.cat(manufactured_wavefield(X_sensor.double()), dim=1).float()
    noise = torch.randn(N_sensors, 3, generator=g)
    sigma = clean.pow(2).mean(0).sqrt() * 10.0 ** (-snr_db / 20.0)
    U_sensor = clean + noise * sigma

    X_domain = X_domain.to(device)
    f1, f3, f_phi = generate_mms_forcing(X_domain)
    return X_domain, f1, f3, f_phi, X_sensor.to(device), U_sensor.to(device), X_bc.to(device), X_ic.to(device)

def pde_residuals(model, X_g):
    u1, u3, phi = model(X_g)

    u1_x, u1_z, u1_t = torch.autograd.grad(u1.sum(), X_g, create_graph=True)[0].split(1, dim=1)
    u3_x, u3_z, u3_t = torch.autograd.grad(u3.sum(), X_g, create_graph=True)[0].split(1, dim=1)
    phi_x, phi_z, _ = torch.autograd.grad(phi.sum(), X_g, create_graph=True)[0].split(1, dim=1)

    g_u1_x = torch.autograd.grad(u1_x.sum(), X_g, create_graph=True)[0]
    u1_xx, u1_xz = g_u1_x[:, 0:1], g_u1_x[:, 1:2]
    u1_zz = torch.autograd.grad(u1_z.sum(), X_g, create_graph=True)[0][:, 1:2]
    u1_tt = torch.autograd.grad(u1_t.sum(), X_g, create_graph=True)[0][:, 2:3]

    g_u3_x = torch.autograd.grad(u3_x.sum(), X_g, create_graph=True)[0]
    u3_xx, u3_xz = g_u3_x[:, 0:1], g_u3_x[:, 1:2]
    u3_zz = torch.autograd.grad(u3_z.sum(), X_g, create_graph=True)[0][:, 1:2]
    u3_tt = torch.autograd.grad(u3_t.sum(), X_g, create_graph=True)[0][:, 2:3]

    g_phi_x = torch.autograd.grad(phi_x.sum(), X_g, create_graph=True)[0]
    phi_xx, phi_xz = g_phi_x[:, 0:1], g_phi_x[:, 1:2]
    phi_zz = torch.autograd.grad(phi_z.sum(), X_g, create_graph=True)[0][:, 1:2]

    e15_p = model.e15_pred

    r_u1 = u1_tt - (c11*u1_xx + c44*u1_zz + (c13+c44)*u3_xz + k0_sq*(true_e31+e15_p)*phi_xz)
    r_u3 = u3_tt - (c44*u3_xx + c33*u3_zz + (c13+c44)*u1_xz + k0_sq*e15_p*phi_xx + k0_sq*true_e33*phi_zz)
    r_phi = (e15_p+true_e31)*u1_xz + e15_p*u3_xx + true_e33*u3_zz - eps11*phi_xx - eps33*phi_zz

    return r_u1, r_u3, r_phi

def compute_loss(model, data):
    X_domain, f1, f3, f_phi, X_sensor, U_sensor, X_bc, X_ic = data

    X_g = X_domain.clone().detach().requires_grad_(True)
    r_u1, r_u3, r_phi = pde_residuals(model, X_g)

    L_mech = ((r_u1 - f1)**2).mean() + ((r_u3 - f3)**2).mean()
    L_gauss = ((r_phi - f_phi)**2).mean()

    b1, b3, bphi = model(X_bc)
    L_bc = (b1**2).mean() + (b3**2).mean() + (bphi**2).mean()

    X_i = X_ic.clone().detach().requires_grad_(True)
    i1, i3, iphi = model(X_i)
    L_ic_val = (i1**2).mean() + (i3**2).mean() + (iphi**2).mean()
    i1_t = torch.autograd.grad(i1.sum(), X_i, create_graph=True)[0][:, 2:3]
    i3_t = torch.autograd.grad(i3.sum(), X_i, create_graph=True)[0][:, 2:3]
    iphi_t = torch.autograd.grad(iphi.sum(), X_i, create_graph=True)[0][:, 2:3]
    L_ic_vel = (i1_t**2).mean() + (i3_t**2).mean() + (iphi_t**2).mean()

    u1_s, u3_s, phi_s = model(X_sensor)
    L_data = ((u1_s - U_sensor[:, 0:1])**2).mean() + ((u3_s - U_sensor[:, 1:2])**2).mean() + ((phi_s - U_sensor[:, 2:3])**2).mean()

    return [L_mech, L_gauss, L_bc, L_ic_val, L_ic_vel, L_data]

def update_weights(losses, w, params, alpha=0.1, floor=5000.0):
    grad_sq = []
    for L in losses:
        gs = torch.autograd.grad(L, params, retain_graph=True, allow_unused=True)
        grad_sq.append(sum((g**2).sum() for g in gs if g is not None).item())
    grad_sq = np.maximum(np.array(grad_sq, dtype=np.float64), 1e-12)
    target = grad_sq.sum() / grad_sq
    target = target / target.sum() * len(losses)
    w_new = (1 - alpha) * w + alpha * target
    w_new[5] = max(w_new[5], floor)
    return w_new, grad_sq

def evaluate(model):
    g = torch.Generator().manual_seed(12345)

    X_t = torch.rand(N_eval_field, 3, generator=g)
    X_t[:, 2] *= T_final
    with torch.no_grad():
        pred = torch.cat(model(X_t.to(device)), 1).cpu()
        exact = torch.cat(manufactured_wavefield(X_t.double()), 1).float()
    rel_l2 = ((pred - exact).norm(dim=0) / exact.norm(dim=0)).tolist()

    X_b = torch.rand(N_eval_bc, 3, generator=g)
    X_b[:, 2] *= T_final
    face = torch.randint(0, 4, (N_eval_bc,), generator=g)
    X_b[face == 0, 0] = 0.0
    X_b[face == 1, 0] = 1.0
    X_b[face == 2, 1] = 0.0
    X_b[face == 3, 1] = 1.0
    with torch.no_grad():
        pb = torch.cat(model(X_b.to(device)), 1).cpu()
    bc_rms = pb.pow(2).mean(0).sqrt().tolist()

    X_i = torch.rand(N_eval_bc, 3, generator=g)
    X_i[:, 2] = 0.0
    X_i = X_i.to(device).requires_grad_(True)
    ic_rate_rms = [torch.autograd.grad(v.sum(), X_i, retain_graph=True)[0][:, 2:3].detach().pow(2).mean().sqrt().item() for v in model(X_i)]

    X_p = torch.rand(N_eval_pde, 3, generator=g)
    X_p[:, 2] *= T_final
    sq = [0.0, 0.0, 0.0]
    for s in range(0, N_eval_pde, 1250):
        X_q = X_p[s:s+1250].to(device).requires_grad_(True)
        r = pde_residuals(model, X_q)
        f = generate_mms_forcing(X_p[s:s+1250].to(device))
        for k in range(3):
            sq[k] += ((r[k].detach() - f[k])**2).sum().item()
    pde_rms = [np.sqrt(v / N_eval_pde) for v in sq]

    return rel_l2, bc_rms, ic_rate_rms, pde_rms

def run_training(cfg_name, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    data = generate_data(seed)
    model = SoftPenaltyPINN().to(device)
    params_nn = [p for n, p in model.named_parameters() if n != "e15_pred"]

    adaptive = configs[cfg_name] is None
    w = np.ones(6) if adaptive else np.array(configs[cfg_name], dtype=np.float64)
    wt = torch.tensor(w, dtype=torch.float32, device=device)

    opt_adam = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = CosineAnnealingLR(opt_adam, T_max=N_adam, eta_min=1e-5)

    e15_hist, terms_hist, total_hist, w_hist, grad_hist = [], [], [], [], []

    start_time = time.time()
    for ep in range(N_adam):
        opt_adam.zero_grad(set_to_none=True)
        L = compute_loss(model, data)

        if adaptive and ep % 100 == 0:
            w, grad_sq = update_weights(L, w, params_nn)
            wt = torch.tensor(w, dtype=torch.float32, device=device)
            w_hist.append([ep] + w.tolist())
            grad_hist.append([ep] + grad_sq.tolist())

        loss = sum(wt[i] * L[i] for i in range(6))
        loss.backward()
        opt_adam.step()
        scheduler.step()

        e15_hist.append(model.e15_pred.item() * e33_SI)
        terms_hist.append(torch.stack([l.detach() for l in L]))
        total_hist.append(loss.detach())

        if (ep + 1) % 1500 == 0:
            print(f"    Epoch {ep+1:5d} | e15 = {model.e15_pred.item()*e33_SI:.5f} | Loss: {loss.item():.2e}")
        if (ep + 1) % 250 == 0 and not torch.isfinite(loss):
            print("    Diverged")
            break

    e15_adam = model.e15_pred.item() * e33_SI

    opt_lbfgs = torch.optim.LBFGS(
        model.parameters(), lr=0.1, max_iter=N_lbfgs, history_size=100,
        line_search_fn='strong_wolfe', tolerance_grad=1e-7, tolerance_change=1e-9
    )
    lbfgs_e15, lbfgs_total = [], []

    def closure():
        opt_lbfgs.zero_grad()
        L = compute_loss(model, data)
        loss = sum(wt[i] * L[i] for i in range(6))
        loss.backward()
        lbfgs_e15.append(model.e15_pred.item() * e33_SI)
        lbfgs_total.append(loss.item())
        return loss

    if torch.isfinite(loss):
        opt_lbfgs.step(closure)

    lbfgs_iters = int(opt_lbfgs.state[opt_lbfgs._params[0]].get("n_iter", 0))
    final_e15 = model.e15_pred.item() * e33_SI
    time_mins = (time.time() - start_time) / 60.0

    rel_l2, bc_rms, ic_rate_rms, pde_rms = evaluate(model)

    np.savez_compressed(f"Baseline-Runs/{cfg_name}_seed{seed}.npz",
                        e15_adam=np.array(e15_hist), terms_adam=torch.stack(terms_hist).cpu().numpy(),
                        total_adam=torch.stack(total_hist).cpu().numpy(),
                        weight_updates=np.array(w_hist), grad_norm_sq=np.array(grad_hist),
                        e15_lbfgs=np.array(lbfgs_e15), total_lbfgs=np.array(lbfgs_total))

    err_adam = abs(e15_adam - true_e15_SI) / true_e15_SI * 100
    err_final = abs(final_e15 - true_e15_SI) / true_e15_SI * 100
    return [e15_adam, final_e15, err_adam, err_final, lbfgs_iters, len(lbfgs_total),
            rel_l2[0], rel_l2[1], rel_l2[2], bc_rms[1], bc_rms[2], ic_rate_rms[1], pde_rms[2], time_mins]

os.makedirs("Baseline-Runs", exist_ok=True)
csv_file = "Baseline-Results.csv"

if not os.path.exists(csv_file):
    with open(csv_file, "w") as f:
        f.write("Config,Seed,e15_adam,e15_final,Err_adam,Err_final,LBFGS_iters,Closure_evals,relL2_u1,relL2_u3,relL2_phi,bc_rms_u3,bc_rms_phi,ic_rate_rms_u3,pde_rms_gauss,Minutes\n")

done = []
prev = pd.read_csv(csv_file)
if len(prev) > 0:
    done = list(zip(prev["Config"], prev["Seed"]))

tasks = [(c, s) for s in seeds for c in configs][worker_id::n_workers]

for cfg_name, seed in tasks:
    if (cfg_name, seed) in done:
        continue
    print(f"\n>>> INITIATING CONFIG {cfg_name} | SEED {seed}")
    res = run_training(cfg_name, seed)

    print(f"    e15 = {res[1]:.4f} C/m^2 | Error = {res[3]:.3f}% | Time: {res[-1]:.1f} m")

    with open(csv_file, "a") as f:
        f.write(f"{cfg_name},{seed}," + ",".join(f"{v:.6g}" for v in res) + "\n")

df = pd.read_csv(csv_file)
stats = df.groupby('Config').agg(
    n=('Seed', 'count'),
    err_final_mean=('Err_final', 'mean'), err_final_std=('Err_final', 'std'),
    err_adam_mean=('Err_adam', 'mean'), err_adam_std=('Err_adam', 'std'),
    relL2_u3=('relL2_u3', 'mean'), bc_rms_u3=('bc_rms_u3', 'mean')
)
print("\n" + stats.round(4).to_string())

plt.rcParams.update({'font.size': 14, 'font.family': 'serif'})

fig, axes = plt.subplots(1, len(configs), figsize=(7 * len(configs), 5), dpi=300)

for ax, cfg_name in zip(axes, configs):
    for seed in seeds:
        path = f"Baseline-Runs/{cfg_name}_seed{seed}.npz"
        if not os.path.exists(path):
            continue
        z = np.load(path)
        n_a = len(z['e15_adam'])
        ax.plot(np.arange(n_a), z['e15_adam'], linewidth=1, alpha=0.8)
        ax.plot(n_a + np.arange(len(z['e15_lbfgs'])), z['e15_lbfgs'], linewidth=1.2, alpha=0.8)
    ax.axhline(y=true_e15_SI, color='k', linestyle='--', linewidth=1.5)
    ax.set_title(f"Baseline {cfg_name}")
    ax.set_xlabel("Optimization Steps")
    ax.set_ylabel(r"$e_{15}$ (C/m$^2$)")
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("Baseline_Trajectories.png", format="png", dpi=300, bbox_inches="tight")
