import torch
import numpy as np

e33_SI = 23.3
true_e15_SI = 17.0
c0_ref, e0_ref, eps0_ref = 2.3e10, 23.3, 1.30e-8
k0_sq = (e0_ref**2) / (c0_ref * eps0_ref)
c11, c13, c33, c44 = 12.6/2.3, 5.3/2.3, 11.7/2.3, 1.0
eps11, eps33 = 1.503/1.300, 1.0
true_e31, true_e33 = -6.5/23.3, 23.3/23.3
true_e15 = 17.0 / 23.3

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
    x_g = x_tensor.clone().detach().requires_grad_(True)
    u1, u3, phi = manufactured_wavefield(x_g)
    u1_x, u1_z, u1_t = torch.autograd.grad(u1.sum(), x_g, create_graph=True)[0].split(1, dim=1)
    u3_x, u3_z, u3_t = torch.autograd.grad(u3.sum(), x_g, create_graph=True)[0].split(1, dim=1)
    phi_x, phi_z, _ = torch.autograd.grad(phi.sum(), x_g, create_graph=True)[0].split(1, dim=1)
    u1_xx = torch.autograd.grad(u1_x.sum(), x_g, create_graph=True)[0][:, 0:1]
    u1_zz = torch.autograd.grad(u1_z.sum(), x_g, create_graph=True)[0][:, 1:2]
    u1_xz = torch.autograd.grad(u1_x.sum(), x_g, create_graph=True)[0][:, 1:2]
    u1_tt = torch.autograd.grad(u1_t.sum(), x_g, create_graph=True)[0][:, 2:3]
    u3_xx = torch.autograd.grad(u3_x.sum(), x_g, create_graph=True)[0][:, 0:1]
    u3_zz = torch.autograd.grad(u3_z.sum(), x_g, create_graph=True)[0][:, 1:2]
    u3_xz = torch.autograd.grad(u3_x.sum(), x_g, create_graph=True)[0][:, 1:2]
    u3_tt = torch.autograd.grad(u3_t.sum(), x_g, create_graph=True)[0][:, 2:3]
    phi_xx = torch.autograd.grad(phi_x.sum(), x_g, create_graph=True)[0][:, 0:1]
    phi_zz = torch.autograd.grad(phi_z.sum(), x_g, create_graph=True)[0][:, 1:2]
    phi_xz = torch.autograd.grad(phi_x.sum(), x_g, create_graph=True)[0][:, 1:2]
    f1 = u1_tt - (c11*u1_xx + c44*u1_zz + (c13+c44)*u3_xz + k0_sq*(true_e31+true_e15)*phi_xz)
    f3 = u3_tt - (c44*u3_xx + c33*u3_zz + (c13+c44)*u1_xz + k0_sq*true_e15*phi_xx + k0_sq*true_e33*phi_zz)
    f_phi = (true_e15+true_e31)*u1_xz + true_e15*u3_xx + true_e33*u3_zz - eps11*phi_xx - eps33*phi_zz
    return f1.detach(), f3.detach(), f_phi.detach()
