import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingLR
import gc

from physics import *
from model import PiezoInversePINN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)
torch.set_default_dtype(torch.float32)

print(f"Target e15 to Discover: {true_e15:.5f}\n")

model = PiezoInversePINN().to(device)

N_colloc = 25000
X_domain = torch.rand(N_colloc, 3, device=device)
f1_mms, f3_mms, f_phi_mms = generate_mms_forcing(X_domain)

N_sensors = 5000
X_sensor = torch.rand(N_sensors, 3, device=device)
X_sensor[:, 1] = 0.9
u1_ts, u3_ts, phi_ts = manufactured_wavefield(X_sensor)

def compute_loss(X_b, f1_b, f3_b, f_phi_b, return_components=False):
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
    
    u1_s, u3_s, phi_s = model(X_sensor)
    L_data = ((u1_s - u1_ts)**2).mean() + ((u3_s - u3_ts)**2).mean() + ((phi_s - phi_ts)**2).mean()
    
    total_loss = L_pde + 5000.0 * L_data
    
    if return_components:
        return total_loss, L_pde, L_data
    return total_loss

batch_size = 5000
e15_hist, loss_pde_hist, loss_data_hist = [], [], []

print("STAGE 1: Adam Optimizer (15,000 Epochs)")
opt_adam = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = CosineAnnealingLR(opt_adam, T_max=15000, eta_min=1e-5)

for ep in range(15000):
    idx = torch.randperm(N_colloc)[:batch_size]
    X_b = X_domain[idx]
    f1_b, f3_b, f_phi_b = f1_mms[idx], f3_mms[idx], f_phi_mms[idx]
    
    opt_adam.zero_grad()
    loss, L_p, L_d = compute_loss(X_b, f1_b, f3_b, f_phi_b, return_components=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt_adam.step()
    scheduler.step()
    
    e15_hist.append(model.e15_pred.item())
    loss_pde_hist.append(L_p.item())
    loss_data_hist.append(L_d.item())

    if (ep + 1) % 1500 == 0:
        print(f"Epoch {ep+1:5d} | e15 = {model.e15_pred.item():.5f} (True: {true_e15:.5f})")

gc.collect()
torch.cuda.empty_cache()

print("\nSTAGE 2: L-BFGS Micro-Curvature Polish")
opt_lbfgs = torch.optim.LBFGS(
    model.parameters(), lr=0.1, max_iter=1000, history_size=100, 
    line_search_fn='strong_wolfe', tolerance_grad=1e-13, tolerance_change=1e-13
)

X_lbfgs = X_domain[:15000]
f1_lbfgs, f3_lbfgs, f_phi_lbfgs = f1_mms[:15000], f3_mms[:15000], f_phi_mms[:15000]

lbfgs_iter = 0
def closure():
    global lbfgs_iter
    opt_lbfgs.zero_grad()
    loss, L_p, L_d = compute_loss(X_lbfgs, f1_lbfgs, f3_lbfgs, f_phi_lbfgs, return_components=True)
    loss.backward()
    lbfgs_iter += 1
    
    e15_hist.append(model.e15_pred.item())
    loss_pde_hist.append(L_p.item())
    loss_data_hist.append(L_d.item())
        
    if lbfgs_iter % 25 == 0:
        print(f"L-BFGS iter {lbfgs_iter:3d} | Loss = {loss.item():.6e} | e15 = {model.e15_pred.item():.5f}")
    return loss

opt_lbfgs.step(closure)

err_margin = abs(model.e15_pred.item() - true_e15)/true_e15 * 100
print(f"\nFINAL DISCOVERED e15: {model.e15_pred.item():.5f} (Target: {true_e15:.5f})")
print(f"Discovery Error: {err_margin:.4f}%\n")

print("Generating publication graphics...")
plt.rcParams.update({'font.size': 12})

res = 100
x_l, z_l = np.linspace(0, 1, res), np.linspace(0, 1, res)
X_g, Z_g = np.meshgrid(x_l, z_l)
pts = np.hstack([X_g.flatten()[:, None], Z_g.flatten()[:, None], np.full((res**2, 1), 0.5)])
grid_tensor = torch.tensor(pts, dtype=torch.float32, device=device)

with torch.no_grad():
    _, u3_pred, phi_pred = model(grid_tensor)
    u3_pred = u3_pred.cpu().numpy().reshape(res, res)
    phi_pred = phi_pred.cpu().numpy().reshape(res, res)

fig = plt.figure(figsize=(18, 10), dpi=200)

ax1 = plt.subplot(2, 2, 1)
c1 = ax1.contourf(X_g, Z_g, u3_pred, levels=50, cmap='RdBu_r')
ax1.set_title(r"2D Wave Caustics: Displacement $u_3$ (t=0.5)")
ax1.set_xlabel("Domain $x_1$")
ax1.set_ylabel("Domain $x_3$")
fig.colorbar(c1, ax=ax1)

ax2 = plt.subplot(2, 2, 2)
slice_idx = res // 4 
ax2.plot(z_l, u3_pred[:, slice_idx], 'b-', linewidth=2, label=r'Displacement $u_3$')
ax2.plot(z_l, phi_pred[:, slice_idx], 'r--', linewidth=2, label=r'Electric Potential $\phi$')
ax2.set_title(r"Coupled Electromechanical Response ($x_1 = 0.25$)")
ax2.set_xlabel("Domain $x_3$")
ax2.set_ylabel("Normalized Amplitude")
ax2.grid(True, alpha=0.3)
ax2.legend()

ax3 = plt.subplot(2, 2, 3)
ax3.plot(loss_pde_hist, color='teal', alpha=0.8, label=r'$\mathcal{L}_{PDE}$')
ax3.plot(loss_data_hist, color='orange', alpha=0.8, label=r'$\mathcal{L}_{Data}$')
ax3.set_yscale('log')
ax3.set_title("Constrained Loss Dynamics")
ax3.set_xlabel("Optimization Steps")
ax3.set_ylabel("Log Loss")
ax3.grid(True, alpha=0.3)
ax3.legend()

ax4 = plt.subplot(2, 2, 4)
ax4.plot(e15_hist, color='blue', linewidth=2, label=r'PINN $e_{15}$ Trajectory')
ax4.axhline(y=true_e15, color='red', linestyle='--', linewidth=2, label='True PZT-5H $e_{15}$')
ax4.set_title("Inverse Discovery of Shear Piezoelectric Coupling")
ax4.set_xlabel("Optimization Steps")
ax4.set_ylabel(r"Dimensionless $e_{15}$")
ax4.grid(True, alpha=0.3)
ax4.legend()

plt.tight_layout()
plt.savefig('PRA_Manuscript_Graphics.png', format='png', bbox_inches='tight')
plt.show()
print("Graphics successfully saved to 'PRA_Manuscript_Graphics.png'")
