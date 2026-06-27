import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingLR
import time
from physics import *
from model import ConstrainedPINN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)
torch.set_default_dtype(torch.float32)

df = pd.read_csv("Raw-FEM-Dataset.txt", delim_whitespace=True, header=None)

x_raw = df.iloc[:, 0].values
x_coords = (x_raw - x_raw.min()) / (x_raw.max() - x_raw.min()) 
z_coords = np.full_like(x_coords, 0.9)

u_data_raw = df.iloc[:, 1::3].values 
v_data_raw = df.iloc[:, 2::3].values 
phi_data_raw = df.iloc[:, 3::3].values 

n_nodes = u_data_raw.shape[0]
n_time_steps = u_data_raw.shape[1]

u_data = u_data_raw * 1e6
v_data = v_data_raw * 1e6
phi_data = phi_data_raw / Phi_ref

t_physical = np.linspace(0, 5.0e-6, n_time_steps)
t_array = t_physical / t_ref 

X_grid, T_grid = np.meshgrid(x_coords, t_array, indexing='ij')
Z_grid = np.full_like(X_grid, 0.9) 

X_sensor_np = np.hstack([X_grid.reshape(-1, 1), Z_grid.reshape(-1, 1), T_grid.reshape(-1, 1)])
U_clean_np = np.hstack([u_data.reshape(-1, 1), v_data.reshape(-1, 1), phi_data.reshape(-1, 1)])

def add_awgn(signal, snr_db=30):
    sig_power = np.mean(signal**2)
    if sig_power == 0: return signal
    noise_power = sig_power / (10 ** (snr_db / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), signal.shape)
    return signal + noise

U_noisy_np = np.zeros_like(U_clean_np)
U_noisy_np[:, 0] = add_awgn(U_clean_np[:, 0], snr_db=30)
U_noisy_np[:, 1] = add_awgn(U_clean_np[:, 1], snr_db=30)
U_noisy_np[:, 2] = add_awgn(U_clean_np[:, 2], snr_db=30)

X_sensor = torch.tensor(X_sensor_np, dtype=torch.float32, device=device)
U_sensor = torch.tensor(U_noisy_np, dtype=torch.float32, device=device)
N_sensor = X_sensor.shape[0]

model = ConstrainedPINN().to(device)

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
    L_Data = ((u1_s - U_sens_batch[:, 0:1])**2).mean() + \
             ((u3_s - U_sens_batch[:, 1:2])**2).mean() + \
             ((phi_s - U_sens_batch[:, 2:3])**2).mean()
             
    return L_PDE, L_Data

N_colloc = 25000
X_domain = torch.rand(N_colloc, 3, device=device)
X_domain[:, 2] = X_domain[:, 2] * 0.875

opt_adam = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = CosineAnnealingLR(opt_adam, T_max=15000, eta_min=1e-5)

pde_hist = []
data_hist = []
e15_hist = []

for ep in range(15000):
    model.train()
    opt_adam.zero_grad()
    
    idx_pde = torch.randperm(N_colloc)[:5000]
    idx_sens = torch.randperm(N_sensor)[:2000]
    
    L_pde, L_data = compute_loss(model, X_domain[idx_pde], X_sensor[idx_sens], U_sensor[idx_sens])
    
    loss = L_pde + 5000.0 * L_data
    loss.backward()
    opt_adam.step()
    scheduler.step()
    
    pde_hist.append(L_pde.item())
    data_hist.append(L_data.item())
    e15_hist.append(model.e15_pred.item() * e33_SI)

opt_lbfgs = torch.optim.LBFGS(model.parameters(), max_iter=1250, tolerance_grad=1e-7, tolerance_change=1e-9, line_search_fn="strong_wolfe")

X_colloc_lbfgs = X_domain[:15000]
idx_sens_lbfgs = torch.randperm(N_sensor)[:5000]
X_sens_lbfgs = X_sensor[idx_sens_lbfgs]
U_sens_lbfgs = U_sensor[idx_sens_lbfgs]

def closure():
    opt_lbfgs.zero_grad()
    L_pde, L_data = compute_loss(model, X_colloc_lbfgs, X_sens_lbfgs, U_sens_lbfgs)
    loss = L_pde + 5000.0 * L_data
    loss.backward()
    
    pde_hist.append(L_pde.item())
    data_hist.append(L_data.item())
    e15_hist.append(model.e15_pred.item() * e33_SI)
    
    return loss

opt_lbfgs.step(closure)

total_steps = list(range(len(pde_hist)))

plt.rcParams.update({'font.size': 12, 'font.family': 'DejaVu Sans'})

model.eval()

res = 200 
x_l = np.linspace(0, 1, res)
z_l = np.linspace(0, 1, res)
X_g, Z_g = np.meshgrid(x_l, z_l)

t_snapshot_dimensional = 5.0e-6
t_snapshot_normalized = t_snapshot_dimensional / t_ref

pts = np.hstack([X_g.flatten()[:, None], 
                 Z_g.flatten()[:, None], 
                 np.full((res**2, 1), t_snapshot_normalized)])
grid_tensor = torch.tensor(pts, dtype=torch.float32, device=device)

with torch.no_grad():
    u1_pred, u3_pred, phi_pred = model(grid_tensor)
    
    u3_dimensional = (u3_pred.cpu().numpy().reshape(res, res) * U_ref) * 1e12 
    phi_dimensional = (phi_pred.cpu().numpy().reshape(res, res) * Phi_ref)    
    
X_g_mm = X_g * 10.0
Z_g_mm = Z_g * 10.0
z_l_mm = z_l * 10.0

fig = plt.figure(figsize=(18, 11), dpi=300)

ax1 = plt.subplot(2, 2, 1)
vmax = np.max(np.abs(u3_dimensional))
c1 = ax1.contourf(X_g_mm, Z_g_mm, u3_dimensional, levels=60, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
ax1.set_title(r"Predicted Displacement $u_3$ at $t = 5.0 \mu s$")
ax1.set_xlabel("Domain $x_1$ (mm)")
ax1.set_ylabel("Domain $x_3$ (mm)")
cbar = fig.colorbar(c1, ax=ax1)
cbar.set_label("Raw Displacement (pm)")

ax2 = plt.subplot(2, 2, 2)
slice_idx = int(res * 0.125)

line1 = ax2.plot(z_l_mm, u3_dimensional[:, slice_idx], 'b-', linewidth=2.5, label=r'Raw Displacement $u_3$')
ax2.set_xlabel("Domain $x_3$ (mm)")
ax2.set_ylabel(r"Raw Displacement $u_3$ (pm)", color='b', weight='bold')
ax2.tick_params(axis='y', labelcolor='b')
ax2.grid(True, alpha=0.3)

ax2_twin = ax2.twinx()
line2 = ax2_twin.plot(z_l_mm, phi_dimensional[:, slice_idx], 'r--', linewidth=2.5, label=r'Raw Electric Potential $\phi$')
ax2_twin.set_ylabel(r"Raw Electric Potential $\phi$ (V)", color='r', weight='bold')
ax2_twin.tick_params(axis='y', labelcolor='r')

lines = line1 + line2
labels = [l.get_label() for l in lines]
ax2.legend(lines, labels, loc="lower left", framealpha=0.9)
ax2.set_title(r"Coupled Electromechanical Response ($x_1 = 1.25$ mm)")

ax3 = plt.subplot(2, 2, 3)
ax3.plot(total_steps, pde_hist, color="#1f77b4", alpha=0.8, linewidth=2, label=r"$\mathcal{L}_{PDE}$")
ax3.plot(total_steps, data_hist, color="#ff7f0e", alpha=0.8, linewidth=2, label=r"$\mathcal{L}_{Data}$")
ax3.axvline(x=15000, color='gray', linestyle='--', alpha=0.7, label="L-BFGS Transition")

ax3.set_yscale("log")
ax3.set_title("Constrained Loss Dynamics (Adam + L-BFGS)")
ax3.set_xlabel("Optimization Steps")
ax3.set_ylabel("Log Loss")
ax3.grid(True, alpha=0.3, which="both", ls="--")
ax3.legend(loc="upper right", framealpha=0.9)

ax4 = plt.subplot(2, 2, 4)
ax4.plot(total_steps, e15_hist, color="indigo", linewidth=2.5, label=r"PINN $e_{15}$ Trajectory")
ax4.axhline(y=17.0, color="crimson", linestyle="--", linewidth=2.5, label=r"True Target $e_{15} = 17.0$")
ax4.set_title("Inverse Discovery of Shear Piezoelectric Coupling")
ax4.set_xlabel("Optimization Steps")
ax4.set_ylabel(r"$e_{15}$ (C/m$^2$)")
ax4.grid(True, alpha=0.3)
ax4.legend(loc="lower right", framealpha=0.9)

plt.tight_layout(pad=2.0)
plt.savefig("Figure_5_FEM_CrossValidation.png", format="png", dpi=300, bbox_inches="tight")