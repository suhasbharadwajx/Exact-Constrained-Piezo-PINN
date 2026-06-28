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

c11, c13, c33 = 12.6/2.3, 5.3/2.3, 11.7/2.3
e31 = -6.5 / 23.3
eps11 = 1.503 / 1.300
k0_sq = (e33_SI**2) / (c44_SI * eps33_SI)
true_e15_dimless = true_e15_SI / e33_SI

def get_analytical_forces(x, z, t):
    K = 2 * np.pi
    A1, A3, Aphi = 0.08, 0.10, 0.12
    
    sin2x, cos2x = torch.sin(K * x), torch.cos(K * x)
    sin4x, cos4x = torch.sin(2 * K * x), torch.cos(2 * K * x)
    sin2z, cos2z = torch.sin(K * z), torch.cos(K * z)
    sin4z, cos4z = torch.sin(2 * K * z), torch.cos(2 * K * z)
    
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
