import torch
import numpy as np

e33_SI = 23.3
true_e15_SI = 17.0
L_ref = 0.01
U_ref = 1e-9
c44_SI = 2.30e10
eps33_SI = 1.30e-8

t_ref = L_ref * np.sqrt(7500 / c44_SI)
Phi_ref = U_ref * (e33_SI / eps33_SI)

c11 = 12.6 / 2.3
c13 = 5.3 / 2.3
c33 = 11.7 / 2.3
c44 = 1.0
e31 = -6.5 / 23.3
eps11 = 1.503 / 1.300
eps33 = 1.0
k0_sq = (e33_SI**2) / (c44_SI * eps33_SI)

true_e15_dimless = true_e15_SI / e33_SI
true_e31 = e31
true_e33 = 1.0

def manufactured_wavefield(x_tensor):
    x, z, t = x_tensor[:, 0:1], x_tensor[:, 1:2], x_tensor[:, 2:3]

    spatial_u1 = torch.sin(2 * np.pi * x) * torch.sin(4 * np.pi * z)
    spatial_u3 = torch.sin(4 * np.pi * x) * torch.sin(2 * np.pi * z)
    spatial_phi = torch.sin(2 * np.pi * x) * torch.sin(2 * np.pi * z)

    omega = 2.0 * np.pi
    temporal_form = (1.0 - torch.cos(omega * t))

    u1_mms = 0.08 * temporal_form * spatial_u1
    u3_mms = 0.10 * temporal_form * spatial_u3
    phi_mms = 0.12 * temporal_form * spatial_phi

    return u1_mms, u3_mms, phi_mms

def generate_mms_forcing(x_tensor):
    x_g = x_tensor.clone().detach().double().requires_grad_(True)
    u1, u3, phi = manufactured_wavefield(x_g)

    u1_x, u1_z, u1_t = torch.autograd.grad(u1.sum(), x_g, create_graph=True)[0].split(1, dim=1)
    u3_x, u3_z, u3_t = torch.autograd.grad(u3.sum(), x_g, create_graph=True)[0].split(1, dim=1)
    phi_x, phi_z, _ = torch.autograd.grad(phi.sum(), x_g, create_graph=True)[0].split(1, dim=1)

    g_u1_x = torch.autograd.grad(u1_x.sum(), x_g, create_graph=True)[0]
    u1_xx, u1_xz = g_u1_x[:, 0:1], g_u1_x[:, 1:2]
    u1_zz = torch.autograd.grad(u1_z.sum(), x_g, create_graph=True)[0][:, 1:2]
    u1_tt = torch.autograd.grad(u1_t.sum(), x_g, create_graph=True)[0][:, 2:3]

    g_u3_x = torch.autograd.grad(u3_x.sum(), x_g, create_graph=True)[0]
    u3_xx, u3_xz = g_u3_x[:, 0:1], g_u3_x[:, 1:2]
    u3_zz = torch.autograd.grad(u3_z.sum(), x_g, create_graph=True)[0][:, 1:2]
    u3_tt = torch.autograd.grad(u3_t.sum(), x_g, create_graph=True)[0][:, 2:3]

    g_phi_x = torch.autograd.grad(phi_x.sum(), x_g, create_graph=True)[0]
    phi_xx, phi_xz = g_phi_x[:, 0:1], g_phi_x[:, 1:2]
    phi_zz = torch.autograd.grad(phi_z.sum(), x_g, create_graph=True)[0][:, 1:2]

    f1 = u1_tt - (c11*u1_xx + c44*u1_zz + (c13+c44)*u3_xz + k0_sq*(true_e31+true_e15_dimless)*phi_xz)
    f3 = u3_tt - (c44*u3_xx + c33*u3_zz + (c13+c44)*u1_xz + k0_sq*true_e15_dimless*phi_xx + k0_sq*true_e33*phi_zz)
    f_phi = (true_e15_dimless+true_e31)*u1_xz + true_e15_dimless*u3_xx + true_e33*u3_zz - eps11*phi_xx - eps33*phi_zz

    return f1.detach().float(), f3.detach().float(), f_phi.detach().float()

def get_analytical_forces(x, z, t):
    K = 2 * np.pi
    A1, A3, Aphi = 0.08, 0.10, 0.12

    sin2x = torch.sin(K * x)
    cos2x = torch.cos(K * x)
    sin4x = torch.sin(2 * K * x)
    cos4x = torch.cos(2 * K * x)

    sin2z = torch.sin(K * z)
    cos2z = torch.cos(K * z)
    sin4z = torch.sin(2 * K * z)
    cos4z = torch.cos(2 * K * z)

    C_t = 1.0 - torch.cos(K * t)
    C_tt = (K**2) * torch.cos(K * t)

    u1_tt = A1 * C_tt * sin2x * sin4z
    u1_xx = -A1 * C_t * (K**2) * sin2x * sin4z
    u1_zz = -A1 * C_t * ((2*K)**2) * sin2x * sin4z
    u1_xz =  A1 * C_t * K * (2*K) * cos2x * cos4z

    u3_tt = A3 * C_tt * sin4x * sin2z
    u3_xx = -A3 * C_t * ((2*K)**2) * sin4x * sin2z
    u3_zz = -A3 * C_t * (K**2) * sin4x * sin2z
    u3_xz =  A3 * C_t * (2*K) * K * cos4x * cos2z

    phi_xx = -Aphi * C_t * (K**2) * sin2x * sin2z
    phi_zz = -Aphi * C_t * (K**2) * sin2x * sin2z
    phi_xz =  Aphi * C_t * K * K * cos2x * cos2z

    f1 = u1_tt - c11*u1_xx - u1_zz - (c13+1)*u3_xz - k0_sq*(e31+true_e15_dimless)*phi_xz
    f3 = u3_tt - u3_xx - c33*u3_zz - (c13+1)*u1_xz - k0_sq*true_e15_dimless*phi_xx - k0_sq*phi_zz
    q  = (true_e15_dimless+e31)*u1_xz + true_e15_dimless*u3_xx + u3_zz - eps11*phi_xx - phi_zz

    return f1, f3, q
