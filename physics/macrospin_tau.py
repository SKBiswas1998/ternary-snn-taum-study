"""Macrospin LLG estimate of the leak time constant tau_l(alpha, Ku1) for the 32 nm MTJ free layer.
Only input from Joy: K_c ~ 5.6e5 J/m3 (J_th -> 0 there). Demag factors follow from K_c. No fitted parameter.
Closed form it reproduces:  tau = 1 / (alpha * gamma * mu0*Ms * (Nz-Nxy) * (1 - Ku1/K_c)).
Fit uses the same variable as Joy: u = mz/sqrt(1-mz^2) decays as exp(-t/tau).
"""
import numpy as np
from scipy.integrate import solve_ivp
g=1.7595e11; mu0=4e-7*np.pi; Ms=1e6
Kc=5.6e5                                   # from Joy's J_th -> 0 at Ku1 ~ 5.6e5 (no other fit)
Nz_m_Nxy=2*Kc/(mu0*Ms**2)                  # = 1.5 Nz - 0.5
Nz=(Nz_m_Nxy+0.5)/1.5; Nxy=(1-Nz)/2
def rhs(t,m,alpha,Ku1):
    H=np.array([-mu0*Ms*Nxy*m[0],-mu0*Ms*Nxy*m[1],-mu0*Ms*Nz*m[2]+2*Ku1/Ms*m[2]])  # Tesla
    mxH=np.cross(m,H); d=-g/(1+alpha**2)*(mxH+alpha*np.cross(m,mxH))
    return d
def tau(alpha,Ku1,T=60e-9,tilt=0.05):
    m0=np.array([tilt,0,1.0]); m0/=np.linalg.norm(m0)
    s=solve_ivp(rhs,[0,T],m0,args=(alpha,Ku1),rtol=1e-9,atol=1e-12,t_eval=np.linspace(0,T,24000),method='DOP853')
    t=s.t; mz=np.abs(s.y[2]); k=(mz>0.02)&(mz<0.995)
    b=np.where(mz<=0.02)[0]
    if b.size: k[b[0]:]=False
    if k.sum()<10: return np.nan
    z=np.log(mz[k]/np.sqrt(1-mz[k]**2)); ts=t[k]*1e9
    p=np.polyfit(ts,z,1); return -1/p[0]
print('Nz=%.3f Nxy=%.3f'%(Nz,Nxy))
print('--- alpha sweep at Ku1=0 (mumax fits: 0.01->0.495, 0.05->0.104, 0.1->0.052 ns)')
for a in (0.01,0.05,0.1): print(' alpha=%.2f  tau=%.3f ns'%(a,tau(a,0,T=20e-9)))
print('--- Ku1 sweep, alpha=0.01 (mumax fit: 0 -> 0.49-0.52 ns, 2.52e5 -> 0.99-1.11 ns)')
for k in (0,1e5,2e5,2.52e5,3e5,4e5,5e5,5.3e5):
    tt=tau(0.01,k,T=150e-9); print(' Ku1=%.2e  tau=%.3f ns   formula tau0/(1-K/Kc)=%.3f'%(k,tt,0.5/(1-k/Kc)))
