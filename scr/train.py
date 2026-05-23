import torch
import torch.nn as nn
from torch.optim import Adam, LBFGS
from model import ConstrainedMultiphysicsPINN
from physics import compute_pde_residuals, manufactured_solutions

def generate_data():
    # 25,000 Interior collocation points
    x1_colloc = torch.rand(25000, 1, requires_grad=True)
    x3_colloc = torch.rand(25000, 1, requires_grad=True)
    t_colloc = torch.rand(25000, 1, requires_grad=True)

    # 5,000 Subsurface sensor points at x3 = 0.9
    x1_sensor = torch.rand(5000, 1, requires_grad=True)
    x3_sensor = torch.full((5000, 1), 0.9, requires_grad=True)
    t_sensor = torch.rand(5000, 1, requires_grad=True)
    
    # Generate exact manufactured targets for the sensors
    u1_target, u3_target, phi_target = manufactured_solutions(x1_sensor, x3_sensor, t_sensor)
    
    # Generate the strict MMS forcing terms evaluated at e15_true = 0.72961
    u1_mms, u3_mms, phi_mms = manufactured_solutions(x1_colloc, x3_colloc, t_colloc)
    force_1, force_3, force_phi = compute_pde_residuals(u1_mms, u3_mms, phi_mms, x1_colloc, x3_colloc, t_colloc, e15_bar=0.72961)

    return (x1_colloc, x3_colloc, t_colloc, force_1.detach(), force_3.detach(), force_phi.detach(), 
            x1_sensor, x3_sensor, t_sensor, u1_target.detach(), u3_target.detach(), phi_target.detach())

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConstrainedMultiphysicsPINN().to(device)
    
    (x1_c, x3_c, t_c, f1_true, f3_true, fphi_true, 
     x1_s, x3_s, t_s, u1_targ, u3_targ, phi_targ) = [tensor.to(device) for tensor in generate_data()]

    mse = nn.MSELoss()
    lam_data = 5000.0

    def compute_loss():
        # PDE Loss
        u1_pred, u3_pred, phi_pred = model(x1_c, x3_c, t_c)
        res_1, res_3, res_phi = compute_pde_residuals(u1_pred, u3_pred, phi_pred, x1_c, x3_c, t_c, model.e15_pred)
        
        loss_pde = mse(res_1, f1_true) + mse(res_3, f3_true) + mse(res_phi, fphi_true)
        
        # Sensor Data Loss
        u1_s_pred, u3_s_pred, phi_s_pred = model(x1_s, x3_s, t_s)
        loss_data = mse(u1_s_pred, u1_targ) + mse(u3_s_pred, u3_targ) + mse(phi_s_pred, phi_targ)
        
        return loss_pde + lam_data * loss_data

    # STAGE 1: Adam
    optimizer_adam = Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_adam, T_max=15000)
    
    print("Beginning Stage 1: Adam Optimization")
    for epoch in range(15000):
        optimizer_adam.zero_grad()
        loss = compute_loss()
        loss.backward()
        optimizer_adam.step()
        scheduler.step()
        
        if epoch % 1000 == 0:
            print(f"Epoch {epoch} | Loss: {loss.item():.4e} | e15_pred: {model.e15_pred.item():.5f}")

    # STAGE 2: L-BFGS
    optimizer_lbfgs = LBFGS(
        model.parameters(), 
        max_iter=50000, 
        tolerance_grad=1e-13, 
        tolerance_change=1e-13, 
        line_search_fn="strong_wolfe"
    )

    print("\nBeginning Stage 2: L-BFGS Optimization")
    def closure():
        optimizer_lbfgs.zero_grad()
        loss = compute_loss()
        loss.backward()
        return loss

    optimizer_lbfgs.step(closure)
    print(f"\nFinal Discovered e15: {model.e15_pred.item():.5f} (Target: 0.72961)")

if __name__ == "__main__":
    train()
