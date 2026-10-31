import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingLR
import time
import os
import io
import sys
import zipfile
from physics import *
from model import ConstrainedPINN

worker_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
n_workers = int(sys.argv[2]) if len(sys.argv) > 2 else 1

if torch.cuda.is_available():
    device = torch.device(f"cuda:{worker_id % torch.cuda.device_count()}")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")
torch.set_default_dtype(torch.float32)

N_colloc = 25000
N_adam = 15000
batch_size = 5000
snr_db = 30
lam = 5000.0
N_eval_field = 20000
N_eval_pde = 5000
T_final_mms = 1.0
T_final_fem = 0.875

mms_lbfgs = dict(lr=0.1, max_iter=1500)
fem_lbfgs = dict(lr=1.0, max_iter=1250)

studies = {
    "FEM-all": dict(data="FEM", use_phi=True),
    "FEM-disp": dict(data="FEM", use_phi=False),
    "MMS-disp": dict(data="MMS", use_phi=False)
}

seeds = [1, 2, 3, 4, 5]
tasks_all = [("FEM-all", 42)]
for s in seeds:
    tasks_all += [("MMS-disp", s), ("FEM-disp", s), ("FEM-all", s)]

runs_dir = "Extended-Runs"
csv_file = "Extended-Results.csv"

def load_fem():
    candidates = ["FEM-Dataset.txt", "FEM-Dataset.txt.zip",
                  "../FEM-framework/FEM-Dataset.txt.zip", "../SNR-sweep/FEM-Dataset.txt.zip"]
    path = next((p for p in candidates if os.path.exists(p)), None)
    if path is None:
        raise FileNotFoundError("FEM-Dataset.txt not found")
    if path.endswith(".zip"):
        with zipfile.ZipFile(path) as zf:
            member = [m for m in zf.namelist() if m.endswith(".txt") and not m.startswith("__MACOSX")][0]
            df = pd.read_csv(io.BytesIO(zf.read(member)), comment='%', sep=r'\s+', header=None)
    else:
        df = pd.read_csv(path, comment='%', sep=r'\s+', header=None)

    x_raw = df.iloc[:, 0].values
    x_coords = (x_raw - x_raw.min()) / (x_raw.max() - x_raw.min())

    u_data = df.iloc[:, 1::3].values * 1e6
    v_data = df.iloc[:, 2::3].values * 1e6
    phi_data = df.iloc[:, 3::3].values / Phi_ref

    n_time_steps = u_data.shape[1]
    t_array = np.linspace(0, 5.0e-6, n_time_steps) / t_ref

    X_grid, T_grid = np.meshgrid(x_coords, t_array, indexing='ij')
    Z_grid = np.full_like(X_grid, 0.9)

    X_sensor_np = np.hstack([X_grid.reshape(-1, 1), Z_grid.reshape(-1, 1), T_grid.reshape(-1, 1)])
    U_clean_np = np.hstack([u_data.reshape(-1, 1), v_data.reshape(-1, 1), phi_data.reshape(-1, 1)])
    return x_coords, t_array, X_sensor_np, U_clean_np

def add_awgn(signal, snr_db=30):
    sig_power = np.mean(signal**2)
    if sig_power == 0: return signal
    noise_power = sig_power / (10 ** (snr_db / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), signal.shape)
    return signal + noise

def last_time(arr, n_nodes, n_t):
    return arr.reshape(n_nodes, n_t, 3)[:, -1, :]

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

def mms_loss(model, X_b, f1_b, f3_b, f_phi_b, X_sensor, U_sensor, use_phi):
    X_g = X_b.clone().detach().requires_grad_(True)
    r_u1, r_u3, r_phi = pde_residuals(model, X_g)

    L_pde = ((r_u1 - f1_b)**2).mean() + ((r_u3 - f3_b)**2).mean() + ((r_phi - f_phi_b)**2).mean()

    u1_s, u3_s, phi_s = model(X_sensor)
    L_data = ((u1_s - U_sensor[:, 0:1])**2).mean() + ((u3_s - U_sensor[:, 1:2])**2).mean()
    if use_phi:
        L_data = L_data + ((phi_s - U_sensor[:, 2:3])**2).mean()

    return L_pde, L_data

def fem_loss(model, X_colloc, X_sens_batch, U_sens_batch, use_phi):
    x_g = X_colloc[:, 0:1].clone().requires_grad_(True)
    z_g = X_colloc[:, 1:2].clone().requires_grad_(True)
    t_g = X_colloc[:, 2:3].clone().requires_grad_(True)
    x_tensor = torch.cat([x_g, z_g, t_g], dim=1)

    u1, u3, phi = model(x_tensor)

    u1_t = torch.autograd.grad(u1.sum(), t_g, create_graph=True)[0]
    u1_tt = torch.autograd.grad(u1_t.sum(), t_g, create_graph=True)[0]
    u1_x = torch.autograd.grad(u1.sum(), x_g, create_graph=True)[0]
    u1_xx = torch.autograd.grad(u1_x.sum(), x_g, create_graph=True)[0]
    u1_z = torch.autograd.grad(u1.sum(), z_g, create_graph=True)[0]
    u1_zz = torch.autograd.grad(u1_z.sum(), z_g, create_graph=True)[0]
    u1_xz = torch.autograd.grad(u1_x.sum(), z_g, create_graph=True)[0]

    u3_t = torch.autograd.grad(u3.sum(), t_g, create_graph=True)[0]
    u3_tt = torch.autograd.grad(u3_t.sum(), t_g, create_graph=True)[0]
    u3_x = torch.autograd.grad(u3.sum(), x_g, create_graph=True)[0]
    u3_xx = torch.autograd.grad(u3_x.sum(), x_g, create_graph=True)[0]
    u3_z = torch.autograd.grad(u3.sum(), z_g, create_graph=True)[0]
    u3_zz = torch.autograd.grad(u3_z.sum(), z_g, create_graph=True)[0]
    u3_xz = torch.autograd.grad(u3_x.sum(), z_g, create_graph=True)[0]

    phi_x = torch.autograd.grad(phi.sum(), x_g, create_graph=True)[0]
    phi_xx = torch.autograd.grad(phi_x.sum(), x_g, create_graph=True)[0]
    phi_z = torch.autograd.grad(phi.sum(), z_g, create_graph=True)[0]
    phi_zz = torch.autograd.grad(phi_z.sum(), z_g, create_graph=True)[0]
    phi_xz = torch.autograd.grad(phi_x.sum(), z_g, create_graph=True)[0]

    f1_true, f3_true, q_true = get_analytical_forces(x_g, z_g, t_g)
    e15_p = model.e15_pred

    res_1 = u1_tt - c11*u1_xx - u1_zz - (c13+1)*u3_xz - k0_sq*(e31+e15_p)*phi_xz - f1_true
    res_3 = u3_tt - u3_xx - c33*u3_zz - (c13+1)*u1_xz - k0_sq*e15_p*phi_xx - k0_sq*phi_zz - f3_true
    res_phi = (e15_p+e31)*u1_xz + e15_p*u3_xx + u3_zz - eps11*phi_xx - phi_zz - q_true

    L_PDE = (res_1**2).mean() + (res_3**2).mean() + (res_phi**2).mean()

    u1_s, u3_s, phi_s = model(X_sens_batch)
    L_Data = ((u1_s - U_sens_batch[:, 0:1])**2).mean() + ((u3_s - U_sens_batch[:, 1:2])**2).mean()
    if use_phi:
        L_Data = L_Data + ((phi_s - U_sens_batch[:, 2:3])**2).mean()

    return L_PDE, L_Data

def generate_mms_data(seed):
    g = torch.Generator().manual_seed(seed)

    X_domain = torch.rand(N_colloc, 3, generator=g)
    X_domain[:, 2] *= T_final_mms

    X_sensor = torch.rand(5000, 3, generator=g)
    X_sensor[:, 1] = 0.9
    X_sensor[:, 2] *= T_final_mms

    X_bc = torch.rand(5000, 3, generator=g)
    face = torch.randint(0, 4, (5000,), generator=g)
    X_ic = torch.rand(5000, 3, generator=g)

    clean = torch.cat(manufactured_wavefield(X_sensor.double()), dim=1).float()
    noise = torch.randn(5000, 3, generator=g)
    sigma = clean.pow(2).mean(0).sqrt() * 10.0 ** (-snr_db / 20.0)
    U_sensor = clean + noise * sigma

    X_domain = X_domain.to(device)
    f1, f3, f_phi = generate_mms_forcing(X_domain)
    return X_domain, f1, f3, f_phi, X_sensor.to(device), U_sensor.to(device)

def evaluate(model, t_max):
    g = torch.Generator().manual_seed(12345)

    X_t = torch.rand(N_eval_field, 3, generator=g)
    X_t[:, 2] *= t_max
    with torch.no_grad():
        pred = torch.cat(model(X_t.to(device)), 1).cpu()
        exact = torch.cat(manufactured_wavefield(X_t.double()), 1).float()
    rel_l2 = ((pred - exact).norm(dim=0) / exact.norm(dim=0)).tolist()

    torch.rand(4000, 3, generator=g)
    torch.randint(0, 4, (4000,), generator=g)
    torch.rand(4000, 3, generator=g)

    X_p = torch.rand(N_eval_pde, 3, generator=g)
    X_p[:, 2] *= t_max
    sq = [0.0, 0.0, 0.0]
    for s in range(0, N_eval_pde, 1250):
        X_q = X_p[s:s+1250].to(device).requires_grad_(True)
        r = pde_residuals(model, X_q)
        f = generate_mms_forcing(X_p[s:s+1250].to(device))
        for k in range(3):
            sq[k] += ((r[k].detach() - f[k])**2).sum().item()
    pde_rms = [np.sqrt(v / N_eval_pde) for v in sq]

    g_early = torch.Generator().manual_seed(777)
    X_e = torch.rand(2000, 3, generator=g_early)
    X_e[:, 2] *= 0.05
    X_q = X_e.to(device).requires_grad_(True)
    r = pde_residuals(model, X_q)
    f = generate_mms_forcing(X_e.to(device))
    early_rms = [np.sqrt(((r[k].detach() - f[k])**2).mean().item()) for k in range(3)]

    return rel_l2, pde_rms, early_rms

def run_training(study, seed):
    use_phi = studies[study]["use_phi"]
    is_fem = studies[study]["data"] == "FEM"

    torch.manual_seed(seed)
    np.random.seed(seed)

    extra = {}
    if is_fem:
        x_coords, t_array, X_sensor_np, U_clean_np = fem_data
        U_noisy_np = np.zeros_like(U_clean_np)
        U_noisy_np[:, 0] = add_awgn(U_clean_np[:, 0], snr_db)
        U_noisy_np[:, 1] = add_awgn(U_clean_np[:, 1], snr_db)
        U_noisy_np[:, 2] = add_awgn(U_clean_np[:, 2], snr_db)

        X_sensor = torch.tensor(X_sensor_np, dtype=torch.float32, device=device)
        U_sensor = torch.tensor(U_noisy_np, dtype=torch.float32, device=device)
        U_clean = torch.tensor(U_clean_np, dtype=torch.float32, device=device)
        N_sensor = X_sensor.shape[0]

        model = ConstrainedPINN().to(device)
        X_domain = torch.rand(N_colloc, 3, device=device)
        X_domain[:, 2] = X_domain[:, 2] * T_final_fem
        lbfgs_cfg = fem_lbfgs
        t_max = T_final_fem
    else:
        X_domain, f1, f3, f_phi, X_sensor, U_sensor = generate_mms_data(seed)
        model = ConstrainedPINN().to(device)
        lbfgs_cfg = mms_lbfgs
        t_max = T_final_mms

    opt_adam = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = CosineAnnealingLR(opt_adam, T_max=N_adam, eta_min=1e-5)

    pde_hist, data_hist, e15_hist = [], [], []

    start_time = time.time()
    for ep in range(N_adam):
        model.train()
        opt_adam.zero_grad()

        if is_fem:
            idx_pde = torch.randperm(N_colloc)[:batch_size]
            idx_sens = torch.randperm(N_sensor)[:2000]
            L_pde, L_data = fem_loss(model, X_domain[idx_pde], X_sensor[idx_sens], U_sensor[idx_sens], use_phi)
        else:
            idx = torch.randperm(N_colloc, device=device)[:batch_size]
            L_pde, L_data = mms_loss(model, X_domain[idx], f1[idx], f3[idx], f_phi[idx], X_sensor, U_sensor, use_phi)

        loss = L_pde + lam * L_data
        loss.backward()
        opt_adam.step()
        scheduler.step()

        pde_hist.append(L_pde.item())
        data_hist.append(L_data.item())
        e15_hist.append(model.e15_pred.item() * e33_SI)

        if (ep + 1) % 250 == 0 and not np.isfinite(loss.item()):
            break

    e15_adam = model.e15_pred.item() * e33_SI

    opt_lbfgs = torch.optim.LBFGS(
        model.parameters(), lr=lbfgs_cfg["lr"], max_iter=lbfgs_cfg["max_iter"], history_size=100,
        line_search_fn='strong_wolfe', tolerance_grad=1e-7, tolerance_change=1e-9
    )

    X_colloc_lbfgs = X_domain[:15000]
    if is_fem:
        idx_sens_lbfgs = torch.randperm(N_sensor)[:5000]
        X_sens_lbfgs = X_sensor[idx_sens_lbfgs]
        U_sens_lbfgs = U_sensor[idx_sens_lbfgs]
    else:
        f1_l, f3_l, f_phi_l = f1[:15000], f3[:15000], f_phi[:15000]

    n_closure = [0]

    def closure():
        opt_lbfgs.zero_grad()
        if is_fem:
            L_pde, L_data = fem_loss(model, X_colloc_lbfgs, X_sens_lbfgs, U_sens_lbfgs, use_phi)
        else:
            L_pde, L_data = mms_loss(model, X_colloc_lbfgs, f1_l, f3_l, f_phi_l, X_sensor, U_sensor, use_phi)
        loss = L_pde + lam * L_data
        loss.backward()

        n_closure[0] += 1
        pde_hist.append(L_pde.item())
        data_hist.append(L_data.item())
        e15_hist.append(model.e15_pred.item() * e33_SI)
        return loss

    if np.isfinite(loss.item()):
        opt_lbfgs.step(closure)

    lbfgs_iters = int(opt_lbfgs.state[opt_lbfgs._params[0]].get("n_iter", 0))
    final_e15 = model.e15_pred.item() * e33_SI
    time_mins = (time.time() - start_time) / 60.0

    model.eval()
    rel_l2, pde_rms, early_rms = evaluate(model, t_max)

    line_l2 = [np.nan, np.nan, np.nan]
    if is_fem:
        with torch.no_grad():
            pred = torch.cat([torch.cat(model(X_sensor[i:i+100000]), 1) for i in range(0, N_sensor, 100000)], 0)
        line_l2 = ((pred - U_clean).norm(dim=0) / U_clean.norm(dim=0)).tolist()

        n_nodes, n_t = len(x_coords), len(t_array)
        res = 200
        x_l = np.linspace(0, 1, res)
        z_l = np.linspace(0, 1, res)
        X_g, Z_g = np.meshgrid(x_l, z_l)
        t_snap = 5.0e-6 / t_ref
        pts = np.hstack([X_g.flatten()[:, None], Z_g.flatten()[:, None], np.full((res**2, 1), t_snap)])
        grid_tensor = torch.tensor(pts, dtype=torch.float32, device=device)
        with torch.no_grad():
            u1_g, u3_g, phi_g = model(grid_tensor)
        extra = dict(
            grid_u3_pm=(u3_g.cpu().numpy().reshape(res, res) * U_ref) * 1e12,
            grid_phi_V=phi_g.cpu().numpy().reshape(res, res) * Phi_ref,
            line_x=x_coords,
            line_clean=last_time(U_clean.cpu().numpy(), n_nodes, n_t),
            line_noisy=last_time(U_sensor.cpu().numpy(), n_nodes, n_t),
            line_pred=last_time(pred.cpu().numpy(), n_nodes, n_t),
            line_t=t_array[-1],
        )

    np.savez_compressed(f"{runs_dir}/{study}_seed{seed}.npz",
                        e15=np.array(e15_hist), pde=np.array(pde_hist), data=np.array(data_hist),
                        n_adam=N_adam, **extra)

    err_adam = abs(e15_adam - true_e15_SI) / true_e15_SI * 100
    err_final = abs(final_e15 - true_e15_SI) / true_e15_SI * 100
    return [e15_adam, final_e15, err_adam, err_final, lbfgs_iters, n_closure[0],
            rel_l2[0], rel_l2[1], rel_l2[2], pde_rms[0], pde_rms[1], pde_rms[2], early_rms[2],
            line_l2[0], line_l2[1], line_l2[2], time_mins]

os.makedirs(runs_dir, exist_ok=True)

try:
    with open(csv_file, "x") as f:
        f.write("Study,Seed,e15_adam,e15_final,Err_adam,Err_final,LBFGS_iters,Closure_evals,"
                "relL2_u1,relL2_u3,relL2_phi,pde_rms_u1,pde_rms_u3,pde_rms_gauss,early_gauss,"
                "line_relL2_u1,line_relL2_u3,line_relL2_phi,Minutes\n")
except FileExistsError:
    pass

prev = pd.read_csv(csv_file)
done = list(zip(prev["Study"], prev["Seed"]))

tasks = [t for t in tasks_all[worker_id::n_workers] if t not in done]

fem_data = load_fem() if any(studies[s]["data"] == "FEM" for s, _ in tasks) else None

for study, seed in tasks:
    out = run_training(study, seed)
    with open(csv_file, "a") as f:
        f.write(f"{study},{seed}," + ",".join(f"{v:.6g}" for v in out) + "\n")

df = pd.read_csv(csv_file)
lines = []

cols = ["Err_final", "Err_adam", "relL2_u1", "relL2_u3", "relL2_phi", "pde_rms_u1", "pde_rms_u3",
        "pde_rms_gauss", "early_gauss", "line_relL2_u1", "line_relL2_u3", "line_relL2_phi", "LBFGS_iters"]

lines.append("PER-RUN RESULTS")
lines.append(df.round(4).to_string(index=False))

sub = df[df["Seed"].isin(seeds)]
lines.append("\nMEAN +- SAMPLE STD (N-1) OVER SEEDS 1-5")
for study, g in sub.groupby("Study"):
    lines.append(f"\n{study}  (n = {len(g)})")
    for c in cols:
        if g[c].notna().any():
            lines.append(f"  {c:15s} {g[c].mean():10.4f} +- {g[c].std():.4f}")

rep = df[(df["Study"] == "FEM-all") & (df["Seed"] == 42)]
if len(rep) > 0:
    r = rep.iloc[0]
    lines.append(f"\nFEM-all, seed 42: e15 = {r['e15_final']:.4f} C/m^2, error {r['Err_final']:.3f}%")

piv = sub.pivot(index="Seed", columns="Study", values="Err_final")
if "FEM-all" in piv.columns and "FEM-disp" in piv.columns:
    lines.append("\nFEM DATA, SAME SEEDS: error in e15 (%), all fields versus displacement only")
    lines.append(piv[["FEM-all", "FEM-disp"]].round(3).to_string())

abl = None
for p in ["Ablation-Results.csv", os.path.join("..", "Ablation-study", "Ablation-Results.csv")]:
    if os.path.exists(p):
        abl = pd.read_csv(p)
        break
if abl is not None and "MMS-disp" in piv.columns:
    c1 = abl[abl["Arm"] == "C1"].set_index("Seed")["Err_final"]
    lines.append("\nMMS DATA, SAME SEEDS: error in e15 (%), C1 with all fields versus displacement only")
    lines.append(pd.DataFrame({"C1 (all fields)": c1, "MMS-disp": piv["MMS-disp"]}).dropna().round(3).to_string())

with open("Extended-Summary.txt", "w") as f:
    f.write("\n".join(lines) + "\n")

plt.rcParams.update({'font.size': 12, 'font.family': 'DejaVu Sans'})

def exact_slice(x_value, z_values, t_value):
    pts = np.stack([np.full_like(z_values, x_value), z_values, np.full_like(z_values, t_value)], axis=1)
    u1, u3, phi = manufactured_wavefield(torch.tensor(pts, dtype=torch.float64))
    return u3.numpy().ravel() * U_ref * 1e12, phi.numpy().ravel() * Phi_ref

path42 = f"{runs_dir}/FEM-all_seed42.npz"
if os.path.exists(path42):
    z = np.load(path42)
    u3_dimensional = z["grid_u3_pm"]
    phi_dimensional = z["grid_phi_V"]
    pde_hist, data_hist, e15_hist = z["pde"], z["data"], z["e15"]
    total_steps = list(range(len(pde_hist)))
    n_adam = int(z["n_adam"])

    res = 200
    x_l = np.linspace(0, 1, res)
    z_l = np.linspace(0, 1, res)
    X_g, Z_g = np.meshgrid(x_l, z_l)
    X_g_mm, Z_g_mm, z_l_mm = X_g * 10.0, Z_g * 10.0, z_l * 10.0
    slice_idx = int(res * 0.125)
    t_snap = 5.0e-6 / t_ref
    u3_ex, phi_ex = exact_slice(x_l[slice_idx], z_l, t_snap)

    def draw_panel_b(ax2):
        line1 = ax2.plot(z_l_mm, u3_dimensional[:, slice_idx], 'b-', linewidth=2.5, label=r'PINN $u_3$')
        line3 = ax2.plot(z_l_mm, u3_ex, 'k:', linewidth=2.5, label=r'Exact $u_3$')
        ax2.set_xlabel("Domain $x_3$ (mm)")
        ax2.set_ylabel(r"Displacement $u_3$ (pm)", color='b', weight='bold')
        ax2.tick_params(axis='y', labelcolor='b')
        ax2.grid(True, alpha=0.3)
        ax2_twin = ax2.twinx()
        line2 = ax2_twin.plot(z_l_mm, phi_dimensional[:, slice_idx], 'r--', linewidth=2.5, label=r'PINN $\phi$')
        line4 = ax2_twin.plot(z_l_mm, phi_ex, color='k', linestyle='-.', linewidth=2.0, label=r'Exact $\phi$')
        ax2_twin.set_ylabel(r"Electric Potential $\phi$ (V)", color='r', weight='bold')
        ax2_twin.tick_params(axis='y', labelcolor='r')
        lines_all = line1 + line3 + line2 + line4
        ax2.legend(lines_all, [l.get_label() for l in lines_all], loc="lower left", framealpha=0.9)
        ax2.set_title(r"Coupled Electromechanical Response ($x_1 = 1.25$ mm)")

    fig = plt.figure(figsize=(18, 11), dpi=300)

    ax1 = plt.subplot(2, 2, 1)
    vmax = np.max(np.abs(u3_dimensional))
    c1 = ax1.contourf(X_g_mm, Z_g_mm, u3_dimensional, levels=60, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    ax1.set_title(r"Displacement $u_3$ at $t = 5.0 \mu s$")
    ax1.set_xlabel("Domain $x_1$ (mm)")
    ax1.set_ylabel("Domain $x_3$ (mm)")
    cbar = fig.colorbar(c1, ax=ax1)
    cbar.set_label("Displacement (pm)")

    ax2 = plt.subplot(2, 2, 2)
    draw_panel_b(ax2)

    ax3 = plt.subplot(2, 2, 3)
    ax3.plot(total_steps, pde_hist, color="#1f77b4", alpha=0.8, linewidth=2, label=r"$\mathcal{L}_{PDE}$")
    ax3.plot(total_steps, data_hist, color="#ff7f0e", alpha=0.8, linewidth=2, label=r"$\mathcal{L}_{Data}$")
    ax3.axvline(x=n_adam, color='gray', linestyle='--', alpha=0.7, label="L-BFGS Transition")
    ax3.set_yscale("log")
    ax3.set_title("Constrained Loss Dynamics (Adam + L-BFGS)")
    ax3.set_xlabel("Optimization Steps")
    ax3.set_ylabel("Log Loss")
    ax3.grid(True, alpha=0.3, which="both", ls="--")
    ax3.legend(loc="upper right", framealpha=0.9)

    ax4 = plt.subplot(2, 2, 4)
    ax4.plot(total_steps, e15_hist, color="indigo", linewidth=2.5, label=r"PINN $e_{15}$ Trajectory")
    ax4.axhline(y=true_e15_SI, color="crimson", linestyle="--", linewidth=2.5, label=r"True Target $e_{15} = 17.0$")
    ax4.set_title("Inverse Discovery of Shear Piezoelectric Coupling")
    ax4.set_xlabel("Optimization Steps")
    ax4.set_ylabel(r"$e_{15}$ (C/m$^2$)")
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc="lower right", framealpha=0.9)

    plt.tight_layout(pad=2.0)
    plt.savefig("Figure_6_FEM_CrossValidation_Overlay.png", format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure(figsize=(9, 5.5), dpi=300)
    draw_panel_b(plt.subplot(1, 1, 1))
    plt.tight_layout()
    plt.savefig("Figure_6b_Overlay_Panel.png", format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    x_mm = 0.001 + z["line_x"] * 9.998
    clean, noisy, pred = z["line_clean"], z["line_noisy"], z["line_pred"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=300)
    for ax, k, sc, nm in zip(axes[:2], [1, 2], [1e3, Phi_ref],
                             [r"Displacement $u_3$ (pm)", r"Electric potential $\phi$ (V)"]):
        ax.plot(x_mm[::10], noisy[::10, k] * sc, '.', color='lightgray', markersize=3, label="Noisy observations (every 10th)")
        ax.plot(x_mm, clean[:, k] * sc, 'k-', linewidth=2, label="Clean FEM")
        ax.plot(x_mm, pred[:, k] * sc, 'r--', linewidth=2, label="PINN")
        ax.set_xlabel("Domain $x_1$ (mm)")
        ax.set_ylabel(nm)
        ax.set_title(r"Observation line $x_3 = 9$ mm, $t = 5.0\ \mu$s")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", framealpha=0.9)
    ax = axes[2]
    for k, lab, col in zip([0, 1, 2], [r"$u_1$", r"$u_3$", r"$\phi$"], ["tab:green", "tab:blue", "tab:red"]):
        err = 100 * np.abs(pred[:, k] - clean[:, k]) / np.abs(clean[:, k]).max()
        ax.plot(x_mm, err, color=col, linewidth=1.5, label=lab)
    ax.set_xlabel("Domain $x_1$ (mm)")
    ax.set_ylabel("|PINN - clean FEM| (% of peak)")
    ax.set_title("Pointwise error on the observation line")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", framealpha=0.9)
    plt.tight_layout()
    plt.savefig("Figure_FEM_Observation_Line.png", format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)

plt.rcParams.update({'font.size': 14, 'font.family': 'serif'})
fig, axes = plt.subplots(1, 3, figsize=(20, 6), dpi=300)
for ax, study, title in zip(axes, ["FEM-all", "FEM-disp", "MMS-disp"],
                            ["FEM data, all fields", "FEM data, displacement only", "MMS data, displacement only"]):
    for seed in seeds:
        path = f"{runs_dir}/{study}_seed{seed}.npz"
        if not os.path.exists(path):
            continue
        ax.plot(np.load(path)["e15"], linewidth=1.2, alpha=0.8)
    ax.axhline(y=true_e15_SI, color='k', linestyle='--', linewidth=1.5)
    ax.set_ylim(-15, 85)
    ax.set_title(title)
    ax.set_xlabel("Optimization Steps")
    ax.set_ylabel(r"$e_{15}$ (C/m$^2$)")
    ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("Figure_Displacement_Only_Trajectories.png", format="png", dpi=300, bbox_inches="tight")
plt.close(fig)
