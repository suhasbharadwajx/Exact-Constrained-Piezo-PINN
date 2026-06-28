import torch
import numpy as np
import pandas as pd
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt
import time
import os
from physics import *
from model import ConstrainedPINN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float32)
torch.set_float32_matmul_precision('high')

print("Loading FEM Dataset...")
df = pd.read_csv("FEM-Dataset.txt", comment='%', delim_whitespace=True, header=None)

x_raw = df.iloc[:, 0].values
x_coords = (x_raw - x_raw.min()) / (x_raw.max() - x_raw.min()) 
z_coords = np.full_like(x_coords, 0.9)

u_data_raw = df.iloc[:, 1::3].values
v_data_raw = df.iloc[:, 2::3].values
phi_data_raw = df.iloc[:, 3::3].values
n_time_steps = u_data_raw.shape[1]

u_data = u_data_raw * 1e6
v_data = v_data_raw * 1e6
phi_data = phi_data_raw / Phi_ref
t_array = np.linspace(0, 5.0e-6, n_time_steps) / t_ref 

X_grid, T_grid = np.meshgrid(x_coords, t_array, indexing='ij')
Z_grid = np.full_like(X_grid, 0.9) 

X_sensor_np = np.hstack([X_grid.reshape(-1, 1), Z_grid.reshape(-1, 1), T_grid.reshape(-1, 1)])
U_clean_np = np.hstack([u_data.reshape(-1, 1), v_data.reshape(-1, 1), phi_data.reshape(-1, 1)])

X_sensor = torch.tensor(X_sensor_np, dtype=torch.float32, device=device)
N_sensor = X_sensor.shape[0]

def compute_loss(model, X_colloc, X_sens_batch, U_sens_batch):
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
    L_Data = ((u1_s - U_sens_batch[:, 0:1])**2).mean() + ((u3_s - U_sens_batch[:, 1:2])**2).mean() + ((phi_s - U_sens_batch[:, 2:3])**2).mean()
    
    return L_PDE, L_Data

def run_inversion(snr_db, run_idx):
    seed = int(time.time()) + run_idx
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    def add_awgn(signal, snr):
        sig_power = np.mean(signal**2)
        if sig_power == 0: return signal
        noise_power = sig_power / (10 ** (snr / 10))
        return signal + np.random.normal(0, np.sqrt(noise_power), signal.shape)
    
    U_noisy_np = np.zeros_like(U_clean_np)
    U_noisy_np[:, 0] = add_awgn(U_clean_np[:, 0], snr_db)
    U_noisy_np[:, 1] = add_awgn(U_clean_np[:, 1], snr_db)
    U_noisy_np[:, 2] = add_awgn(U_clean_np[:, 2], snr_db)
    U_sensor = torch.tensor(U_noisy_np, dtype=torch.float32, device=device)

    model = ConstrainedPINN().to(device)
    
    N_colloc = 25000
    X_domain = torch.rand(N_colloc, 3, device=device)
    X_domain[:, 2] = X_domain[:, 2] * 0.875 

    opt_adam = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = CosineAnnealingLR(opt_adam, T_max=12000, eta_min=1e-5) 

    for ep in range(12000):
        model.train()
        opt_adam.zero_grad()
        idx_pde = torch.randperm(N_colloc)[:4000]
        idx_sens = torch.randperm(N_sensor)[:2000]
        
        L_pde, L_data = compute_loss(model, X_domain[idx_pde], X_sensor[idx_sens], U_sensor[idx_sens])
        loss = L_pde + 5000.0 * L_data
        loss.backward()
        opt_adam.step()
        scheduler.step()

    opt_lbfgs = torch.optim.LBFGS(model.parameters(), max_iter=500, tolerance_grad=1e-7, tolerance_change=1e-9, line_search_fn="strong_wolfe")
    X_colloc_lbfgs = X_domain[:15000]
    idx_sens_lbfgs = torch.randperm(N_sensor)[:5000]
    X_sens_lbfgs = X_sensor[idx_sens_lbfgs]
    U_sens_lbfgs = U_sensor[idx_sens_lbfgs]

    def closure():
        opt_lbfgs.zero_grad()
        L_pde, L_data = compute_loss(model, X_colloc_lbfgs, X_sens_lbfgs, U_sens_lbfgs)
        loss = L_pde + 5000.0 * L_data
        loss.backward()
        return loss

    opt_lbfgs.step(closure)
    
    final_e15 = model.e15_pred.item() * e33_SI
    error_pct = abs(final_e15 - true_e15_SI) / true_e15_SI * 100
    return final_e15, error_pct

snr_levels = [40, 35, 30, 25, 20, 15, 10, 5, 0]
n_runs = 5
csv_file = "SNR-Sweep-Results.csv"

if not os.path.exists(csv_file):
    with open(csv_file, "w") as f:
        f.write("SNR_dB,Run_ID,Discovered_e15,Error_Percent\n")

for snr in snr_levels:
    print(f"\n>>> INITIATING SNR LEVEL: {snr} dB")
    for run in range(n_runs):
        start_t = time.time()
        e15_res, err_res = run_inversion(snr, run)
        time_mins = (time.time() - start_t) / 60.0
        
        print(f"    Run {run+1}/5 | e15 = {e15_res:.4f} C/m^2 | Error = {err_res:.3f}% | Time: {time_mins:.1f} m")
        
        with open(csv_file, "a") as f:
            f.write(f"{snr},{run+1},{e15_res:.5f},{err_res:.5f}\n")

plt.rcParams.update({'font.size': 14, 'font.family': 'serif'})

try:
    df = pd.read_csv("SNR-Sweep-Results.csv")
except FileNotFoundError:
    print("Error: SNR-Sweep-Results.csv not found.")
    exit()

stats = df.groupby('SNR_dB')['Error_Percent'].agg(['mean', 'std']).reset_index()

snr = stats['SNR_dB'].values
err_mean = stats['mean'].values
err_std = stats['std'].values

fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

ax.fill_between(snr, 
                np.clip(err_mean - err_std, a_min=1e-3, a_max=None),
                err_mean + err_std, 
                color='#1f77b4', alpha=0.2, label=r'$\pm 1\sigma$ Variance ($N=5$)')

ax.plot(snr, err_mean, color='#1f77b4', marker='o', markersize=8, linewidth=2.5, 
        markeredgecolor='black', label=r'Mean Absolute Error')

ax.set_yscale("log")
ax.invert_xaxis()
ax.set_xlim(42, -2)

ax.set_xlabel("Signal-to-Noise Ratio (dB)", weight='bold')
ax.set_ylabel(r"Absolute Parameter Error (%)", weight='bold')
ax.set_title(r"Inverse $e_{15}$ Noise Sensitivity ($N=5$ Independent Runs per Level)", pad=15)

ax.grid(True, which="major", ls="-", alpha=0.3)
ax.grid(True, which="minor", ls="--", alpha=0.1)
ax.legend(loc="upper left", framealpha=0.9)

plt.tight_layout()
plt.savefig("Figure_6_SNR_Sweep_N5.png", format="png", dpi=300, bbox_inches="tight")
