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

# ----------------------
# Jeroen's code

def Dlum(z): #works with individual z, not with lists
  c   = 3*10**5. #SOL in natural units? km/s
  H0  = 70*1.e3#Hubble constant value in km/s/Gpc
  Ode = 0.714 #Omega DE
  Om  = 0.286 #Omega matter
  part1 = c*(1+z)/H0 #"Prefactor"

  #Numerically solve integral part of Dlum
  zlist = np.linspace(0,z,1000)
  dz = zlist[1]-zlist[0]
  #print(dz.shape, dz)
  Ez = Om*(1+zlist)**3+Ode
  integrand = 1/np.sqrt(Ez)
  part2 = np.sum(integrand*dz)
  tot = part1*part2
  return tot

def intI(z,Mc): #For Mc
  #cut noise data into pieces
  T = G/c**3. * Msun_to_kg
  flist = PSD[:,0]
  farr = np.ones((len(Mc),len(flist)))*flist #make frequency data into 2d array

  noise = PSD[:,1]**2. #square according to dimensions
  narr = np.ones((len(Mc),len(noise)))*noise #copy noise into 2d array

  fmax = 4397./((1.+z)*2.**(6./5.)*Mc) #M = 2^6/5 * Mc
  fmax = fmax[:,np.newaxis]

  summand = farr**(-7./3.) / narr / T**(1./3.) #calculate the function f^(-7/3)
  #psd = narr**(-1) #calculate psd^(-1)
  summand[farr>fmax] = 0. #set all values larger than fmax to zero to exclude from integral
  #psd[farr>fmax] = 0 #set all values larger than fmax to zero to exclude from integral

  dfarr = np.diff(farr,axis=1) #calculate multidimensional array of df
  #dfarr = np.diff(flist)
  #dfarr[dfarr<0] = 0 #get rid off the cutoff jump being large potentially

  #summand = ffunc*psd #define the function to integrate over
  res = np.sum((summand[:,1:]+summand[:,:-1])/2.*dfarr,axis=1) #trapezoid rule integral (might go wrong at the edges)
  return res

def rho0(z,Mc):
  #gcf = (1.99*10**30.)**(2./3.)/(3.085*10**(25.)) #conversion of G to Gpc^3 Msol^-1 yr^-2
  #gval = 6.67*10**(-11.) #value of G in si units
  cval = 3*10**8. #Value of SOL in m/s
  R = G/c**2. * Msun_to_kg #gval/cval**2.
  T = 1. #gval/cval**3.
  part1 = np.sqrt(5/(96*np.pi**(4./3.))) #prefactor
  part2 = (((1+z)*Mc)**(5./6.))/ Dlum(z) / Gpc_to_m
  part3 = np.sqrt(intI(z,Mc))
  rho0val = R*part1*part2*part3 #gcf*R*(T**(-1./3.))*part1*part2*part3
  return rho0val


# ----------------------
# Anna's code

def lumdist(z):
    z_int = np.linspace(0.,z,1001)
    H = H0 * np.sqrt((1+z_int)**3.*omega_m + omega_L) * 1.e3 # so the Dl is returned in Gpc
    dl_int = 1/H
    step = np.diff(z_int, axis=0)
    dl = np.sum((dl_int[1:] + dl_int[:-1])*step/2., axis=0)
    return c * 1.e-3 * (1+z) * dl


def I_fact(Mc,z):
    T = G/c**3. * Msun_to_kg
    fmax = 4397./2.**(5./6.)/(1+z)/Mc
    I_int = 1./ T**(1./3.) / freq**(7./3.) / S

    f = np.tile(freq,(fmax.shape[0],1)).transpose()
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


mc = np.linspace(1.,100.,101)

theta1 = Theta(mc, 1.)
theta2 = 5./rho0(1.,mc)

print(theta1[theta1<4.]/theta2[theta1<4.])

plt.plot(mc,theta1)
plt.plot(mc,theta2, linestyle='--')
plt.hlines(4,0.,100.)
plt.show()


