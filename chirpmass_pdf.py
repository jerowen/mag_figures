import numpy as np
import pickle
import copy
import scipy.interpolate as interpolate
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

def secondary_mass(Mc, m1):

    A = (1. - 4.*Mc**5 / 27./ m1**5)**(1./2.)
    S = Mc**5/2./m1**2 * (1.-A)
    T = Mc**5/2./m1**2 * (1.+A)

    return S**(1./3.) + T**(1./3.)

def dm2dMc (Mc, m1):

    A = (1. - 4.*Mc**5 / 27./ m1**5)**(1./2.)
    S = Mc**5/2./m1**2 * (1.+A)
    T = Mc**5/2./m1**2 * (1.-A)
    dSdMc = (5./Mc * S - 5.*Mc**9./27./m1**7/A)
    dTdMc = (5./Mc * T + 5.*Mc**9./27./m1**7/A)
    return 1./3. * S**(-2./3.) * dSdMc + 1./3. * T**(-2./3.) * dTdMc

def smoothing_function(m, mmin, delta):
    f = np.zeros(m.shape)
    f[(m>mmin+0.1) & (m<mmin+delta)] = 1./(np.exp(delta/(m[(m>mmin+0.1) & (m<mmin+delta)]-mmin) + delta/(m[(m>mmin+0.1) & (m<mmin+delta)]-mmin-delta)) + 1.)
    f[m>=mmin+delta] = 1.
    return f

mm1 = np.linspace(1.,100.,1001)

# ------ Power Law + Peak

def smoothed_gaussian(m1):
    mu = 33.07
    sigma = 5.69
    return np.exp(-(m1-mu)**2./2./sigma**2) * smoothing_function(m=m1, mmin=4.59, delta=4.82)

def smoothed_powerlaw(m1):
    alpha = -2.63
    mmax = 86.22
    y = m1**alpha * smoothing_function(m=m1, mmin=4.59, delta=4.82)
    y[m1>mmax] = 0.
    return y

SG = smoothed_gaussian(mm1)
NSG = np.sum( (SG[1:]+SG[:-1]) * np.diff(mm1, axis=0)/2. , axis=0)

SP = smoothed_powerlaw(mm1)
NSP = np.sum( (SP[1:]+SP[:-1]) * np.diff(mm1, axis=0)/2. , axis=0)

def primary_peak_pdf(m1):

    lambda_p = 0.10

    G = smoothed_gaussian(m1) / NSG
    P = smoothed_powerlaw(m1) / NSP

    return (1-lambda_p) * P + lambda_p * G

# ------ Broken Power Law

def smoothed_broken_powerlaw(m1):
    alpha1 = -1.58
    alpha2 = -5.59
    b = 0.43
    mmin = 3.96
    mmax = 87.14
    mbreak = mmin + b *(mmax - mmin)

    y = m1**alpha1
    y[m1>mbreak] = m1[m1>mbreak]**alpha2 * (mbreak**alpha1/mbreak**alpha2)
    y[m1>mmax] = 0.
    y = y * smoothing_function(m=m1, mmin=mmin, delta=4.83)
    return y

SB = smoothed_broken_powerlaw(mm1)
NSB = np.sum( (SB[1:]+SB[:-1]) * np.diff(mm1)/2. )

def primary_broken_pdf(m1):

    B = smoothed_broken_powerlaw(m1) / NSB
    return B

m1 = np.linspace(1., 100., 501)

# ------ Secondary PDF - Power Law + Peak

def norm_secondary_peak(m1):

    beta = 1.40

    mm2 = np.linspace(1.e-3,m1,1001)
    mm1 = np.tile(m1, (mm2.shape[0],1))
    SSP = (mm2/mm1)**beta / mm1 * smoothing_function(m=mm2, mmin=4.59, delta=4.82)

    P = np.sum( (SSP[1:]+SSP[:-1]) * np.absolute(np.diff(mm2, axis=0))/2. , axis=0)
    return P

norm_secondary_peak_pdf = interpolate.interp1d(m1, norm_secondary_peak(m1), kind='cubic')

def secondary_peak_pdf(Mc, m1):

    beta = 1.40

    m2 = secondary_mass(Mc, m1)
    SP = (m2/m1)**beta / m1 * smoothing_function(m=m2, mmin=4.59, delta=4.82)

    P = SP / norm_secondary_peak_pdf(m1)
    return P

# ------ Secondary PDF - Broken Power Law

def norm_secondary_broken(m1):

    beta = 1.26

    mm2 = np.linspace(1.e-3,m1,1001)
    mm1 = np.tile(m1, (mm2.shape[0],1))
    SSP = (mm2/mm1)**beta / mm1 * smoothing_function(m=mm2, mmin=3.96, delta=4.83)

    P = np.sum( (SSP[1:]+SSP[:-1]) * np.absolute(np.diff(mm2, axis=0))/2. , axis=0)
    return P

norm_secondary_broken_pdf = interpolate.interp1d(m1, norm_secondary_broken(m1), kind='cubic')

def secondary_broken_pdf(Mc, m1):

    beta = 1.26

    m2 = secondary_mass(Mc, m1)
    SP = (m2/m1)**beta / m1 * smoothing_function(m=m2, mmin=3.96, delta=4.83)

    P = SP / norm_secondary_broken_pdf(m1)
    return P

# ------ Chirp mass PDF

def chirpmass_peak_pdf(Mc):
    
    mm = np.linspace(2.**(1./5.)*Mc + 0.001, 100., 11)
    MM = np.tile(Mc, (mm.shape[0],1))

    primary = primary_peak_pdf(mm)
    secondary = secondary_peak_pdf(MM, mm)
    JJ = dm2dMc(MM, mm)

    yy = primary * secondary * JJ
    y = np.sum( (yy[1:]+yy[:-1]) * np.diff(mm, axis=0)/2. , axis=0)
    return y

def chirpmass_broken_pdf(Mc):
    
    mm = np.linspace(2.**(1./5.)*Mc + 0.01, 100., 101)
    MM = np.tile(Mc, (mm.shape[0],1))

    primary = primary_broken_pdf(mm)
    secondary = secondary_broken_pdf(MM, mm)
    JJ = dm2dMc(MM, mm)

    yy = primary * secondary * JJ
    y = np.sum( (yy[1:]+yy[:-1]) * np.diff(mm, axis=0)/2. , axis=0)
    return y

chirp_mass = np.linspace(1., 80., 1001)



primary_mass = np.linspace(1., 100., 1001)
#mm = chirpmass_peak_pdf(Mc)
#print(mm[10])
phi_peak = chirpmass_peak_pdf(chirp_mass)
NPhiP = np.sum((phi_peak[1:] + phi_peak[:-1]) * np.diff(chirp_mass)/2.)

phi_broken = chirpmass_broken_pdf(chirp_mass)
NPhiB = np.sum((phi_broken[1:] + phi_broken[:-1]) * np.diff(chirp_mass)/2.)

plt.figure()
plt.plot(chirp_mass, phi_peak/NPhiP)
plt.plot(chirp_mass, phi_broken/NPhiB)
plt.yscale('log')
plt.ylim(1.e-4, 1.)
plt.xlim(1.e-3,100.)


mm = np.linspace(2.**(1./5.)*chirp_mass + 0.001, 100., 1001)
MM = np.tile(chirp_mass, (mm.shape[0],1))

secondary = secondary_peak_pdf(MM, mm)
#secondary[secondary==0.] = 1.
#secondary[MM > (4.59)**(3/5) * mm**(2/5)] = 0.
plt.figure()
plt.imshow(secondary, norm=LogNorm(), extent=[1, 80., 32,0])


plt.show()
