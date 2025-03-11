import numpy as np
import const
#from sed_lc_fn import L_fn, L_fn_vec
from fpath import *
from scipy.interpolate import interp1d
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import griddata
from math import pi, log10, sqrt, log, exp, sqrt, atan, cos, sin, acos, asin

# adjustable parameters
nH0 = 1                # [cm^-3] H number density
tmax = 300.     # [s], duration of the source
LUV = 3e47      # [erg/s]  # source luminosity

# fixed parameters
tmin = 0.
n0_over_nH = 1.45e-15    # dust number density over H number density
lam0 = 2.       # [um] critical wavelength for Qabs_lambda
thej = pi/2     # [rad] jet opening angle
p = 2.2         # electron PL index => spectrum L_nu ~ nu^{(1-p)/2}
nuUVmax = 50/const.erg2eV/const.H_PLANCK   # maximum UV frequency 50 eV
LnuUVmax = (3-p)/2*LUV/nuUVmax   # Lnu at nuUVmax

amin, amax = 0.01, 0.3      # um, grain size limits
Na = 30
aarr = np.logspace(log10(amin), log10(amax), Na)
a_ratio = aarr[1]/aarr[0]

rmin, rmax = 0.4, 100.       # pc, radial layers
Nr = 150     # we need dr/r <~ thej to resolve the light curve of the echo
rarr = np.logspace(log10(rmin), log10(rmax), Nr)
r_ratio = rarr[1]/rarr[0]

#THIS IS NEW
themin, themax = 0.000001, pi    #theta array
Nthe = 50
thearr = np.linspace(themin, themax, Nthe)
Dthe = thearr[1]-thearr[0]

# min and max source frequencies
numin, numax = 0.1/(const.erg2eV*const.H_PLANCK), 50/(const.erg2eV*const.H_PLANCK)
Nnu = 40
nuarr = np.logspace(log10(numin), log10(numax), Nnu)    # frequency bins
nu_ratio = nuarr[1]/nuarr[0]

# jet emission time [dust local frame]
Nt = 20     # this is the dimension we interpolate over
tarr = np.linspace(tmin, tmax, Nt, endpoint=False)
dt = tarr[1] - tarr[0]
tarr += dt/2.

Tarr = np.zeros((Nt, Nr, Na, Nthe), dtype=float)   # to store the dust temperature
asubarr = np.zeros((Nt, Nr, Nthe), dtype=float)    # to store sublimation radii
taudarr = np.zeros((Nnu, Nr, Nthe), dtype=float)   # dust extinction optical depth at each nu
jdnuarr = np.zeros((Nt, Nr, Nthe), dtype=float)    # volumetric emissivity at lamobs


def func_Lnu(t, nu):     # source spectrum and light curve
    if t < tmax:
        return LnuUVmax*(nu/nuUVmax)**((1 - p)/2)  # spectrum L_nu ~ nu^{(1-p)/2}
    return 0.


def func_nH(r, the):      # gas density profile (r in pc)
    return nH0


def func_Qabs(nu, a):      # absorption efficiency for grain size a [um]
    lam = const.C_LIGHT/nu * 1e4    # wavelength in um
    return 1./(1 + (lam/lam0)**2 / a)


def func_T(qdot_h, aum):    # solve the heating and cooling balance for T
    y = qdot_h/(7.12576*aum**2)
    if y >= (31.5/aum)**2/12:
        return 3240/sqrt(aum)   # the grain should evaporate immediately
    xi = sqrt((31.5/aum)**2 - 12*y) + 31.5/aum
    T3 = sqrt((2*y*y/3/xi)**(1./3) + (xi*y/18)**(1./3))
    return 1000*T3


def jdnu_intp(t, j_r, j_the, jdnuarr):
    # linear interpolation in time for a given r, the (given by indexes j_r, j_the)
    i_floor = np.argmin(np.abs(t-tarr))
    if t < tarr[i_floor] and i_floor != 0:
        i_floor -= 1
    if i_floor == Nt - 1:
        i_floor -= 1
    slope = (jdnuarr[i_floor+1, j_r, j_the] - jdnuarr[i_floor, j_r, j_the])\
            /(tarr[i_floor+1] - tarr[i_floor])
    return max(jdnuarr[i_floor, j_r, j_the] + slope * (t - tarr[i_floor]), 0)


# compute cumulative number of H as a function of radius and angle
fine_grid = 10
Nr_fine = int(Nr*fine_grid)
Nthe_fine = int(Nthe*fine_grid)
rarr_fine = np.logspace(log10(rmin/fine_grid), log10(rmax), Nr_fine)
thearr_fine = np.linspace(themin, themax, Nthe_fine)
r_ratio_fine = rarr_fine[1]/rarr_fine[0]
Dthe_fine = thearr_fine[1]-thearr_fine[0]
NHarr_fine = np.zeros((Nthe_fine, Nr_fine), dtype=float)
NH = 0.     # total H number
interp_fns = np.zeros(Nthe_fine, dtype=object)
for i in range(Nthe_fine):
    NH = 0
    for j in range(Nr_fine):
        r = rarr_fine[j]
        the = thearr_fine[i]
        dr = r * (sqrt(r_ratio_fine) - 1/sqrt(r_ratio_fine))
        NH += 2*pi * r*r*np.sin(the)*Dthe_fine*dr*const.pc2cm**3 * func_nH(r, the)
        NHarr_fine[i][j] = NH
    rthe_NH_intp = interp1d(rarr_fine, NHarr_fine[i], fill_value='extrapolate')
    interp_fns[i] = rthe_NH_intp

for i in range(Nr_fine):
    print(np.sum(NHarr_fine[:, i]))
