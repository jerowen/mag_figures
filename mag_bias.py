import numpy as np
import matplotlib.pyplot as plt
import unicodedata
from scipy import integrate
from scipy import interpolate
import math
from itertools import compress

# ----------------------
# units and constants

c = 2.99792458e8 #m/s
G = 6.6743015e-11 # m^3 kg^-1 s^-2
H0 = 70.0 # km/s/Mpc
omega_L = 0.714
omega_m  = 0.286

Msun_to_kg = 1.988e30
yr_to_sec = 31556952.
Mpc_to_m = 3.086e22
Gpc_to_m = 3.086e25

PSD = np.loadtxt('aLIGO_noise.txt')
freq = PSD[:,0]
S = PSD[:,1]**2.

def lumdist(z):
    z_int = np.linspace(0.,z,1001)
    H = H0 * np.sqrt((1+z_int)**3.*omega_m + omega_L) * 1.e3 # so the Dl is returned in Gpc
    dl_int = 1/H
    step = np.diff(z_int, axis=0)
    dl = np.sum((dl_int[1:] + dl_int[:-1])*step/2., axis=0)
    return c * 1.e-3 * (1+z) * dl

def I_fact(Mc,z):
    fmax = 4397./2.**(6./5.)/(1+z)/Mc
    f = np.tile(freq,(fmax.shape[0],1)).transpose()
    T = G/c**3. * Msun_to_kg
    I_int = 1./ T**(1./3.) / freq**(7./3.) / S
    I_int = np.tile(I_int,(fmax.shape[0],1)).transpose()
    I_int[f>fmax] = 0.
    step = np.diff(freq)[:,np.newaxis]
    I = np.sum((I_int[1:,:]+I_int[:-1,:])*step/2., axis=0)
    return I

def Theta(Mc,z):
    rho_th = 5.
    I = I_fact(Mc,z)
    dl = lumdist(z)
    R = G/c**2. * Msun_to_kg
    rho0 = np.sqrt(5./96./np.pi**(4./3.)) * R / dl / Gpc_to_m * ((1+z)*Mc)**(5./6.) * np.sqrt(I)
    return rho_th/rho0

zlist = np.linspace(0,1,11)[1:]
Mclist = np.linspace(1,100,101)

#def Theta(Mc,z)
T = 1 #G/c**3. * Msun_to_kg
I_int = 1./ T**(1./3.) / freq**(7./3.) / S

for red in zlist:
    plt.plot(Mclist, Theta(Mclist,red), label='z='+str(red))
plt.hlines(4.,0.,Mclist.max())
plt.xlabel(r'$M_c\,[M_{\odot}]$')
plt.ylabel(r'$\theta$')
plt.savefig('theta.pdf')
plt.legend()
plt.show()
