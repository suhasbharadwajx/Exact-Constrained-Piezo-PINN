import torch

# IEEE Standard values for PZT-5H
C44 = 2.30e10
E33 = 23.3
EPS33 = 1.30e-8

# Dimensionless Ratios
C11_BAR = 12.6e10 / C44
C13_BAR = 8.41e10 / C44
C33_BAR = 11.7e10 / C44
E31_BAR = -6.5 / E33
EPS11_BAR = 1.50e-8 / EPS33
K0_SQ = (E33 ** 2) / (C44 * EPS33)

def grad(outputs, inputs):
    return torch.autograd.grad(
        outputs, inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=True,
        retain_graph=True
    )[0]

def manufactured_solutions(x1, x3, t):
    D_bc = x1 * (1.0 - x1) * x3 * (1.0 - x3)
    topology = 0.1 * D_bc * (t ** 2)
    
    u1_mms = topology * torch.sin(torch.pi * x1) * torch.cos(torch.pi * x3)
    u3_mms = topology * torch.cos(torch.pi * x1) * torch.sin(torch.pi * x3)
    phi_mms = topology * torch.sin(torch.pi * x1) * torch.sin(torch.pi * x3)
    
    return u1_mms, u3_mms, phi_mms

def compute_pde_residuals(u1, u3, phi, x1, x3, t, e15_bar):
    u1_t = grad(u1, t)
    u1_tt = grad(u1_t, t)
    u1_1 = grad(u1, x1)
    u1_11 = grad(u1_1, x1)
    u1_3 = grad(u1, x3)
    u1_33 = grad(u1_3, x3)
    u1_13 = grad(u1_1, x3)

    u3_t = grad(u3, t)
    u3_tt = grad(u3_t, t)
    u3_1 = grad(u3, x1)
    u3_11 = grad(u3_1, x1)
    u3_3 = grad(u3, x3)
    u3_33 = grad(u3_3, x3)
    u3_13 = grad(u3_1, x3)

    phi_1 = grad(phi, x1)
    phi_11 = grad(phi_1, x1)
    phi_3 = grad(phi, x3)
    phi_33 = grad(phi_3, x3)
    phi_13 = grad(phi_1, x3)

    # Dimensionally stabilized PDE system
    res_1 = (u1_tt) - (C11_BAR * u1_11 + u1_33 + (C13_BAR + 1.0) * u3_13 + K0_SQ * (E31_BAR + e15_bar) * phi_13)
    res_3 = (u3_tt) - (u3_11 + C33_BAR * u3_33 + (C13_BAR + 1.0) * u1_13 + K0_SQ * e15_bar * phi_11 + K0_SQ * phi_33)
    res_phi = (e15_bar + E31_BAR) * u1_13 + e15_bar * u3_11 + u3_33 - EPS11_BAR * phi_11 - phi_33

    return res_1, res_3, res_phi
