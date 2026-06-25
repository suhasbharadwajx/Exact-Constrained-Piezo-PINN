import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingLR
import gc
import os

from physics import *
from model import PiezoInversePINN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)
torch.set_default_dtype(torch.float32)

print(f"Target e15 to Discover: {true_e15_SI:.5f} C/m^2\n")

model = PiezoInversePINN().to(device)

# FEM Data
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "..", "data", "FEM-Dataset.csv")

if not os.path.exists(file_path):
    raise FileNotFoundError(f"Dataset missing: {os.path.abspath(file_path)}\n"
                            f"Please ensure 'comsol_pinn_dataset.csv' is placed inside the 'data/' directory.")

print(f"Loading FEM dataset from: {os.path.abspath(file_path)}")
df_clean = pd.read_csv(file_path)

X_true = df_clean[["x", "y", "t"]].values.astype(np.float32)

Y_noisy = df_clean[["u_noisy", "v_noisy", "phi_noisy"]].values.astype(np.float32)
t_bar_max = float(df_clean["t"].max() / t_ref)

X_sensor = torch.tensor(X_true, dtype=torch.float32, device=device) / torch.tensor([L_ref, L_ref, t_ref], device=device)
Y_sensor = torch.tensor(Y_noisy, dtype=torch.float32, device=device) / torch.tensor([U_ref, U_ref, Phi_ref], device=device)
N_sensors = X_sensor.shape[0]

N_colloc = 25000
X_domain = torch.rand(N_colloc, 3, device=device)
X_domain[:, 2] *= t_bar_max

f1_mms, f3_mms, f_phi_mms = generate_mms_forcing(X_domain)

def compute_loss(X_b, f1_b, f3_b, f_phi_b, X_s, Y_s, return_components=False):
    X_g = X_b.clone().detach().requires_grad_(True)
    u1, u3, phi = model(X_g)
    
    u1_x, u1_z, u1_t = torch.autograd.grad(u1.sum(), X_g, create_graph=True)[0].split(1, dim=1)
    u3_x, u3_z, u3_t = torch.autograd.grad(u3.sum(), X_g, create_graph=True)[0].split(1, dim=1)
    phi_x, phi_z, _ = torch.autograd.grad(phi.sum(), X_g, create_graph=True)[0].split(1, dim=1)
    
    u1_xx = torch.autograd.grad(u1_x.sum(), X_g, create_graph=True)[0][:, 0:1]
    u1_zz = torch.autograd.grad(u1_z.sum(), X_g, create_graph=True)[0][:, 1:2]
    u1_xz = torch.autograd.grad(u1_x.sum(), X_g, create_graph=True)[0][:, 1:2]
    u1_tt = torch.autograd.grad(u1_t.sum(), X_g, create_graph=True)[0][:, 2:3]
    
    u3_xx = torch.autograd.grad(u3_x.sum(), X_g, create_graph=True)[0][:, 0:1]
    u3_zz = torch.autograd.grad(u3_z.sum(), X_g, create_graph=True)[0][:, 1:2]
    u3_xz = torch.autograd.grad(u3_x.sum(), X_g, create_graph=True)[0][:, 1:2]
    u3_tt = torch.autograd.grad(u3_t.sum(), X_g, create_graph=True)[0][:, 2:3]
    
    phi_xx = torch.autograd.grad(phi_x.sum(), X_g, create_graph=True)[0][:, 0:1]
    phi_zz = torch.autograd.grad(phi_z.sum(), X_g, create_graph=True)[0][:, 1:2]
    phi_xz = torch.autograd.grad(phi_x.sum(), X_g, create_graph=True)[0][:, 1:2]
    
    e15_p = model.e15_pred
    
    r_u1 = u1_tt - (c11*u1_xx + c44*u1_zz + (c13+c44)*u3_xz + k0_sq*(true_e31+e15_p)*phi_xz) - f1_b
    r_u3 = u3_tt - (c44*u3_xx + c33*u3_zz + (c13+c44)*u1_xz + k0_sq*e15_p*phi_xx + k0_sq*true_e33*phi_zz) - f3_b
    r_phi = (e15_p+true_e31)*u1_xz + e15_p*u3_xx + true_e33*u3_zz - eps11*phi_xx - eps33*phi_zz - f_phi_b
    
    L_pde = (r_u1**2).mean() + (r_u3**2).mean() + (r_phi**2).mean()
    
    u1_s, u3_s, phi_s = model(X_s)
    L_data = ((u1_s - Y_s[:, 0:1])**2).mean() + ((u3_s - Y_s[:, 1:2])**2).mean() + ((phi_s - Y_s[:, 2:3])**2).mean()
    
    total_loss = L_pde + 5000.0 * L_data
    
    if return_components:
        return total_loss, L_pde, L_data
    return total_loss

batch_size = 5000
e15_hist, loss_pde_hist, loss_data_hist = [], [], []

print("\nSTAGE 1: Adam Optimizer (15,000 Epochs)")
opt_adam = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = CosineAnnealingLR(opt_adam, T_max=15000, eta_min=1e-5)

for ep in range(15000):
    idx_b = torch.randperm(N_colloc)[:batch_size]
    idx_s = torch.randperm(N_sensors)[:batch_size]
    
    X_b = X_domain[idx_b]
    f1_b, f3_b, f_phi_b = f1_mms[idx_b], f3_mms[idx_b], f_phi_mms[idx_b]
    X_s, Y_s = X_sensor[idx_s], Y_sensor[idx_s]
    
    opt_adam.zero_grad()
    loss, L_p, L_d = compute_loss(X_b, f1_b, f3_b, f_phi_b, X_s, Y_s, return_components=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt_adam.step()
    scheduler.step()
    
    e15_hist.append(model.e15_pred.item())
    loss_pde_hist.append(L_p.item())
    loss_data_hist.append(L_d.item())

    if (ep + 1) % 1500 == 0:
        print(f"Epoch {ep+1:5d} | e15 = {model.e15_pred.item()*e33_SI:.5f} (True: {true_e15_SI:.5f})")

gc.collect()
torch.cuda.empty_cache()

print("\nSTAGE 2: L-BFGS Micro-Curvature Polish")
opt_lbfgs = torch.optim.LBFGS(
    model.parameters(), lr=0.1, max_iter=1000, history_size=100, 
    line_search_fn='strong_wolfe', tolerance_grad=1e-13, tolerance_change=1e-13
)

X_lbfgs = X_domain[:15000]
f1_lbfgs, f3_lbfgs, f_phi_lbfgs = f1_mms[:15000], f3_mms[:15000], f_phi_mms[:15000]
X_s_lbfgs, Y_s_lbfgs = X_sensor[:15000], Y_sensor[:15000]

lbfgs_iter = 0
def closure():
    global lbfgs_iter
    opt_lbfgs.zero_grad()
    loss, L_p, L_d = compute_loss(X_lbfgs, f1_lbfgs, f3_lbfgs, f_phi_lbfgs, X_s_lbfgs, Y_s_lbfgs, return_components=True)
    loss.backward()
    lbfgs_iter += 1
    
    e15_hist.append(model.e15_pred.item())
    loss_pde_hist.append(L_p.item())
    loss_data_hist.append(L_d.item())
        
    if lbfgs_iter % 25 == 0:
        print(f"L-BFGS iter {lbfgs_iter:3d} | Loss = {loss.item():.6e} | e15 = {model.e15_pred.item()*e33_SI:.5f}")
    return loss

opt_lbfgs.step(closure)

final_e15_dim = model.e15_pred.item() * e33_SI
err_margin = abs(final_e15_dim - true_e15_SI)/true_e15_SI * 100
print(f"\nFINAL DISCOVERED e15: {final_e15_dim:.5f} C/m^2 (Target: {true_e15_SI:.5f})")
print(f"Discovery Error: {err_margin:.4f}%\n")

with torch.no_grad():
    u1_p, u3_p, phi_p = model(X_sensor)
    u1_p = (u1_p * U_ref).cpu().numpy().flatten()
    u3_p = (u3_p * U_ref).cpu().numpy().flatten()
    phi_p = (phi_p * Phi_ref).cpu().numpy().flatten()
    
    # Isolate original FEM fields to evaluate true L2 error of the reconstruction
    u1_t = df_clean["u_true"].values.astype(np.float32)
    u3_t = df_clean["v_true"].values.astype(np.float32)
    phi_t = df_clean["phi_true"].values.astype(np.float32)
    
    def rel_l2(pred, true):
        return np.linalg.norm(pred - true) / max(np.linalg.norm(true), 1e-30)

    print("Reconstruction error vs. NOISE-FREE COMSOL fields (Full Domain):")
    print(f"  u1  relative L2 error: {rel_l2(u1_p, u1_t):.4e}")
    print(f"  u3  relative L2 error: {rel_l2(u3_p, u3_t):.4e}")
    print(f"  phi relative L2 error: {rel_l2(phi_p, phi_t):.4e}\n")

print("Generating publication graphics from memory (No Normalization)...")
plt.rcParams.update({'font.size': 12})

res = 100
x_l = np.linspace(0.0, 1.0, res)
z_l = np.linspace(0.0, 1.0, res)
X_g, Z_g = np.meshgrid(x_l, z_l)

t_plot_bar = t_bar_max * 0.5 
pts_norm = np.hstack([X_g.flatten()[:, None], Z_g.flatten()[:, None], np.full((res ** 2, 1), t_plot_bar)])
grid_tensor = torch.tensor(pts_norm, dtype=torch.float32, device=device)

with torch.no_grad():
    _, u3_plot, phi_plot = model(grid_tensor)
    u3_plot_norm = u3_plot.cpu().numpy().reshape(res, res)
    u3_slice_SI = (u3_plot * U_ref).cpu().numpy().reshape(res, res)
    phi_slice_SI = (phi_plot * Phi_ref).cpu().numpy().reshape(res, res)

fig = plt.figure(figsize=(18, 10), dpi=200)

# Panel 1: Wave Contour
ax1 = plt.subplot(2, 2, 1)
vmin = np.min(u3_plot_norm)
vmax = np.max(u3_plot_norm)
c1 = ax1.contourf(X_g * L_ref * 1000, Z_g * L_ref * 1000, u3_plot_norm, levels=100, cmap="RdBu_r", vmin=vmin, vmax=vmax)
ax1.set_title(rf"Displacement $\bar{{u}}_3$ (High Contrast)")
ax1.set_xlabel("Domain $x_1$ (mm)")
ax1.set_ylabel("Domain $x_3$ (mm)")
fig.colorbar(c1, ax=ax1, format="%.2e")

# Panel 2: Mid-Slice Profiles
ax2 = plt.subplot(2, 2, 2)
slice_idx = res // 2

u3_raw = u3_slice_SI[:, slice_idx] * 1e12  
phi_raw = phi_slice_SI[:, slice_idx]       

color1 = 'blue'
ax2.set_xlabel("Domain $x_3$ (mm)")
ax2.set_ylabel(r"Raw Displacement $u_3$ (pm)", color=color1)
l1, = ax2.plot(z_l * L_ref * 1000, u3_raw, color=color1, linestyle="-", linewidth=2, label=r"Raw $u_3$ (pm)")
ax2.tick_params(axis='y', labelcolor=color1)
ax2.grid(True, alpha=0.3)

max_u3 = np.max(np.abs(u3_raw))
ax2.set_ylim(-1.2 * max_u3, 1.2 * max_u3)

ax2_twin = ax2.twinx()
color2 = 'red'
ax2_twin.set_ylabel(r"Raw Electric Potential $\phi$ (V)", color=color2)
l2, = ax2_twin.plot(z_l * L_ref * 1000, phi_raw, color=color2, linestyle="--", linewidth=2, label=r"Raw $\phi$ (V)")
ax2_twin.tick_params(axis='y', labelcolor=color2)

max_phi = np.max(np.abs(phi_raw))
ax2_twin.set_ylim(-1.2 * max_phi, 1.2 * max_phi)

lines = [l1, l2]
labels = [line.get_label() for line in lines]
ax2.legend(lines, labels, loc="lower right")
ax2.set_title("Coupled Electromechanical Response (Raw Scale)")

# Panel 3: Loss Dynamics
ax3 = plt.subplot(2, 2, 3)
ax3.plot(loss_pde_hist, color="teal", alpha=0.8, label=r"$\mathcal{L}_{PDE}$")
ax3.plot(loss_data_hist, color="orange", alpha=0.8, label=r"$\mathcal{L}_{Data}$")
ax3.set_yscale("log")
ax3.set_title("Loss Dynamics (Adam + L-BFGS)")
ax3.set_xlabel("Optimization Steps")
ax3.set_ylabel("Log Loss")
ax3.grid(True, alpha=0.3)
ax3.legend()

# Panel 4: Parameter Trajectory
ax4 = plt.subplot(2, 2, 4)
total_steps = list(range(len(e15_hist)))
e15_dim_hist = [v * e33_SI for v in e15_hist]
ax4.plot(total_steps, e15_dim_hist, color="blue", linewidth=2, label=r"PINN $e_{15}$ Trajectory")
ax4.axhline(y=true_e15_SI, color="green", linestyle="--", linewidth=2, label=f"True $e_{{15}}$ = {true_e15_SI}")
ax4.set_title("Inverse Discovery of Shear Piezoelectric Coupling")
ax4.set_xlabel("Optimization Steps")
ax4.set_ylabel(r"$e_{15}$ (C/m$^2$)")
ax4.grid(True, alpha=0.3)
ax4.legend(loc="lower right")

plt.tight_layout()
plt.savefig("PRA_Manuscript_Graphics_FEM.png", format="png", bbox_inches="tight")
plt.show()
print("\nGraphics successfully saved to 'PRA_Manuscript_Graphics_FEM.png'.")
