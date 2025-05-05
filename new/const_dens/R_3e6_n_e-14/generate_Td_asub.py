import numpy as np
import numpy as np
import const
from fpath import *
from scipy.interpolate import interp1d
from scipy.interpolate import griddata
from scipy.interpolate import RegularGridInterpolator
from math import pi, log10, sqrt, log, exp, sqrt, atan, cos, sin, acos, asin

# adjustable parameters
nH0 = 1                # [cm^-3] H number density
tmax = 17159000.     # [s], duration of the source
LUV = 3e47      # [erg/s]  # source luminosity

# fixed parameters
tmin = 0.
n0_over_nH = 1.45e-14    # dust number density over H number density
lam0 = 2.       # [um] critical wavelength for Qabs_lambda
thej = pi/2     # [rad] jet opening angle
p = 2.2         # electron PL index => spectrum L_nu ~ nu^{(1-p)/2}
nuUVmax = 50/const.erg2eV/const.H_PLANCK   # maximum UV frequency 50 eV
LnuUVmax = (3-p)/2*LUV/nuUVmax   # Lnu at nuUVmax

amin, amax = 0.01, 0.3      # um, grain size limits
Na = 30
aarr = np.logspace(log10(amin), log10(amax), Na)
a_ratio = aarr[1]/aarr[0]

Rs = 3.888935148e-7 #SgrA* R_s in pc

rmin, rmax = 100*Rs, 3e6*Rs       # pc, radial layers
Nr = 100     # we need dr/r <~ thej to resolve the light curve of the echo
rarr = np.logspace(log10(rmin), log10(rmax), Nr)
r_ratio = rarr[1]/rarr[0]

#THIS IS NEW
themin, themax = 0.000001, pi    #theta array
Nthe = 50
thearr = np.linspace(themin, themax, Nthe)
dthe = thearr[1]-thearr[0]

# min and max source frequencies
numin, numax = 0.1/(const.erg2eV*const.H_PLANCK), 50/(const.erg2eV*const.H_PLANCK)
Nnu = 50
nuarr = np.logspace(log10(numin), log10(numax), Nnu)    # frequency bins
nu_ratio = nuarr[1]/nuarr[0]

# jet emission time [dust local frame]
Nt = 100     # this is the dimension we interpolate over
tarr = np.linspace(tmin, tmax, Nt, endpoint=False)
dt = tarr[1] - tarr[0]
tarr += dt/2.

Tarr = np.zeros((Nt, Nr, Nthe, Na), dtype=float)   # to store the dust temperature
asubarr = np.zeros((Nt, Nr, Nthe), dtype=float)    # to store sublimation radii
taudarr = np.zeros((Nnu, Nr, Nthe), dtype=float)   # dust extinction optical depth at each nu

seds_mosfit = np.genfromtxt('SEDs_mosfit.txt')
time_mosfit= np.genfromtxt('time_SEDs_mosfit.txt')
time_mosfit -= time_mosfit[0]
frequency = np.genfromtxt('frequency.txt')
time_mosfit = time_mosfit*24*60*60

intp = RegularGridInterpolator((time_mosfit, frequency), seds_mosfit)
    # def func_Lnu(time, freq):
    #     return intp((time, freq))
def func_Lnu_vec(time, freq):
    T, F = np.meshgrid(time, freq, indexing='ij')  # Shape: (len(time), len(freq))
    
    # Stack and reshape into a list of (time, freq) pairs for interpolation
    points = np.stack([T.ravel(), F.ravel()], axis=-1)
    
    # Interpolate and reshape back to grid shape
    L = intp(points).reshape(len(time), len(freq))
    return L

def func_Lnu(time, freq):
    # Create a grid of (time, freq) pairs
    T, F = np.meshgrid(time, freq, indexing='ij')  # Shape: (Nt, Nf)
    points = np.stack([T.ravel(), F.ravel()], axis=-1)  # Shape: (Nt*Nf, 2)
    return intp(points).reshape(len(time), len(freq))  # Shape: (Nt, Nf)

def func_nH(r, the):      # gas density profile (r in pc) - MAKE SURE THIS IS VECTORIZABLE
    return nH0

def func_Qabs(nu, a):      # absorption efficiency for grain size a [um]
    lam = const.C_LIGHT/nu * 1e4    # wavelength in um
    return 1./(1 + (lam/lam0)**2 / a)

def func_Qabs_vec(nu, a):
    nu = np.atleast_1d(nu)
    a = np.atleast_1d(a)

    # Create meshgrid for broadcasting
    nu_grid, a_grid = np.meshgrid(nu, a, indexing='ij')  # Shape: (len(nu), len(a))
    
    lam_grid = const.C_LIGHT / nu_grid * 1e4  # wavelength in microns (1e6 to convert from m to um)

    return 1.0 / (1.0 + (lam_grid / lam0) ** 2 / a_grid)

def func_T(qdot_h, aum):    # solve the heating and cooling balance for T
    y = qdot_h/(7.12576*aum**2)
    if y >= (31.5/aum)**2/12:
        return 3240/sqrt(aum)   # the grain should evaporate immediately
    xi = sqrt((31.5/aum)**2 - 12*y) + 31.5/aum
    T3 = sqrt((2*y*y/3/xi)**(1./3) + (xi*y/18)**(1./3))
    return 1000*T3

def func_T_vec(qdot_h, a_val):    # solve the heating and cooling balance for T
    y = qdot_h/(7.12576*a_val**2)
    thresh = (31.5/a_val)**2/12
    thresh = np.tile(thresh, (Nr, Nthe, 1))
    T_evap = 3240/np.sqrt(a_val) 
    T_evap = np.tile(T_evap, (Nr, Nthe, 1))
    xi = np.sqrt((31.5 / a_val)**2 - 12 * y) + 31.5 / a_val
    T3 = np.sqrt(((2 * y**2) / (3 * xi))**(1.0 / 3) + ((xi * y) / 18)**(1.0 / 3))
    T_non_evap = 1000 * T3
    return np.where(y >= thresh, T_evap, T_non_evap)

# compute cumulative number of H as a function of radius
fine_spacing = 10
Nr_fine = int(Nr*fine_spacing)
Nthe_fine = int(Nthe*fine_spacing)
rarr_fine = np.logspace(log10(rmin/fine_spacing), log10(rmax), Nr_fine)
thearr_fine = np.linspace(themin, themax, Nthe_fine)
r_ratio_fine = rarr_fine[1]/rarr_fine[0]
dthe_fine = dthe/fine_spacing
NHarr_fine = np.zeros((Nr_fine, Nthe_fine), dtype=float)
interp_fns = np.zeros(Nthe_fine, dtype=object)

#vectorized version
dr_arr = rarr_fine * (np.sqrt(r_ratio_fine) - 1 / np.sqrt(r_ratio_fine))
r2_dr = 2 * np.pi * rarr_fine**2 * dr_arr * const.pc2cm**3
r_grid, the_grid = np.meshgrid(rarr_fine, thearr_fine, indexing='ij')  # both shape: (Nr, Nthe)
sin_the_grid = np.sin(the_grid)  
nH_grid = func_nH(r_grid, the_grid)
dNH_grid = r2_dr[:, None] * sin_the_grid * dthe_fine * nH_grid
NHarr_fine = np.cumsum(dNH_grid, axis=0)
interp_fns = [interp1d(rarr_fine, NHarr_fine[:, i], fill_value='extrapolate') for i in range(Nthe_fine)]

# compute cumulative number of ionizing photons as a function of time
Nt_fine = int(Nt*20)
tarr_fine = np.linspace(tmin, tmax, Nt_fine)
dt_fine = tarr_fine[1] - tarr_fine[0]
tarr_fine += dt_fine/2.
Nionarr_fine = np.zeros(Nt_fine, dtype=float)   # cumulative number of ionizing photons
Nnu_fine = int(Nnu*5)
nu_ion_min, nu_ion_max = 13.6/(const.erg2eV*const.H_PLANCK), 100/(const.erg2eV*const.H_PLANCK)
nuarr_fine = np.logspace(log10(nu_ion_min), log10(nu_ion_max), Nnu_fine)
nu_ratio_fine = nuarr_fine[1]/nuarr_fine[0]

#vectorized
dnuarr_fine = nuarr_fine * (np.sqrt(nu_ratio_fine) - 1 / np.sqrt(nu_ratio_fine))  # shape (Nnu,)
Lnu_vals = func_Lnu_vec(tarr_fine, nuarr_fine)  # Should return shape (Nt_fine, Nnu)
Lion_arr = Lnu_vals@(dnuarr_fine/nuarr_fine) / (const.H_PLANCK)
Nionarr_fine = np.cumsum(Lion_arr * dt_fine)  # shape (Nt_fine,)
Nion_t_intp = interp1d(Nionarr_fine, tarr_fine, fill_value='extrapolate')

# calculate dust temperature at a given time
def calculate_all_T_vec(t, i_t, rion, Tarr, asubarr, taudarr):
    dnuarr = nuarr * (np.sqrt(nu_ratio) - 1./np.sqrt(nu_ratio))
    Lnu_t = func_Lnu_vec(np.asarray([t]), nuarr)  # (Nnu,)
    Qabs = func_Qabs_vec(nuarr, aarr)
    exp_taud = np.exp(-taudarr)
    
    nu_mask = (nuarr > nu_ion_min).reshape(-1, 1, 1, 1)     # (Nnu, 1, 1, 1)
    r_mask = (rarr[:,None] < rion).reshape(1, -1, len(rion), 1)             # (1, Nr, Nth, 1)
    
    # Combine masks — broadcasting gives shape (Nnu, Nr, Nthe, Na)
    exclude_mask = nu_mask & r_mask
    exclude_mask = np.tile(exclude_mask, (1, 1, 1, Na))
    
    dnu_val = dnuarr.reshape(-1, 1, 1, 1)
    Lnu_val = Lnu_t.reshape(-1, 1, 1, 1)
    Qabs_val = Qabs.reshape(-1, 1, 1, len(aarr))
    exptaud_val = exp_taud[:, :, :, np.newaxis]
    pia2_val = (np.pi*aarr**2).reshape(1, 1, 1, -1)
    denom = (4 * np.pi * rarr**2 * const.pc2cm**2).reshape(1, -1, 1, 1)
    
    qhdot_cont = (dnu_val * Lnu_val * exptaud_val * Qabs_val * pia2_val) / denom
    qhdot_cont[exclude_mask] = 0.
    
    qhdot = np.sum(qhdot_cont, axis=0)*1e-8
    
    a_val = aarr.reshape(1, 1, -1)
    a_val_n = np.tile(a_val, (Nr, Nthe, 1))
    
    Tsub = 2.33e3 * (1 - 0.033*np.log(t/100./a_val_n))
    
    Tvals = func_T_vec(qhdot, a_val)                   # (Nr, Nthe, Na)
    Tmask = (Tvals > Tsub)                             # grains that are *evaporated*
    
    # Mark evaporated grains as 0 in Tarr
    Tarr[i_t] = Tvals * (~Tmask)

    # Now calculate asubarr (min a that survives)
    survive_mask = ~Tmask                              # (Nr, Nthe, Na)
    a_survive = np.where(survive_mask, a_val_n, np.inf)
    asub_now = np.min(a_survive, axis=2)               # (Nr, Nthe)
    
    # Clamp to [amin, amax]
    asub_now = np.clip(asub_now, amin, amax)
    #asub_now
    if i_t == 0:
        asubarr[i_t] = asub_now
    else:
        asubarr[i_t] = np.maximum(asubarr[i_t - 1], asub_now)
    return asubarr, Tarr
    
def calculate_taudarr_vec(i_t, asubarr, taudarr, thearr):
    lamarr = (const.C_LIGHT/nuarr*1e4).reshape(-1, 1, 1)
    xmax = (amax * (lam0/lamarr)**2).reshape(-1, 1, 1)
    tau_pre_factor = (2*sqrt(2)*pi*1e-8*n0_over_nH*lam0/lamarr).reshape(-1, 1, 1)
    
    thearr_n = thearr.reshape(1, 1, -1)
    drarr = (rarr * (sqrt(r_ratio) - 1./sqrt(r_ratio))).reshape(1, -1, 1)
    
    xsub = asubarr[i_t][np.newaxis,:,:]*(lam0/lamarr)**2
    tau_cont = tau_pre_factor * drarr * const.pc2cm * func_nH(rarr, thearr) \
                            * (np.arctan(np.sqrt(0.5*xmax)) - np.arctan(np.sqrt(0.5*xsub)))
    taudarr = np.cumsum(tau_cont, axis=1)
    return taudarr

tol = 1e-10  # tolerance for asubarr
for i in range(Nt):
    # for k in range(Nthe):
    t = tarr[i]
    the = thearr
    Nion = Nion_t_intp(t)
    rion = np.array([interp_fns[int(k)](Nion) for k in range(len(the))])  # ionization radius
    taudarr.fill(0)    # first iteration, no dust extinction
    asubarr, Tarr = calculate_all_T_vec(t, i, rion, Tarr, asubarr, taudarr)
    frac_diff = np.ones(len(the))  # convergence criterion
    n_iter = 0.     # number of iterations
    while np.any(frac_diff > tol):
        n_iter += 1
        asubarr_old = np.copy(asubarr[i])
        # we go back to calculate dust extinction optical depth
        taudarr = calculate_taudarr_vec(i, asubarr, taudarr, thearr)
        # print(taudarr)
        # then calculate the whole temperature again
        asubarr, Tarr = calculate_all_T_vec(t, i, rion, Tarr, asubarr, taudarr)
        frac_diff = np.zeros(len(the))
        frac_diff = np.maximum(frac_diff, np.abs(asubarr_old/asubarr[i] - 1))
    print('t=%.1f' % t, '%d iterations' % n_iter)

# write the results into files: Tarr, asubarr
savelist = ['Td', 'asub']
for i_file in range(len(savelist)):
    fname = 'nH%.1e_' % nH0 + savelist[i_file]
    with open(savedir+fname + '.txt', 'w') as f:
        if savelist[i_file] == 'Td':
            f.write('tmin\ttmax\tNt\t%.8e\t%.8e\t%d\tlinear' % (tmin, tmax, Nt))
            f.write('\nrmin\trmax\tNr\t%.8e\t%.8e\t%d\tlog' % (rmin, rmax, Nr))
            f.write('\nthemin\tthemax\tNthe\t%.8e\t%.8e\t%d\tlinear' % (themin, themax, Nthe))
            f.write('\namin\tamax\tNa\t%.8e\t%.8e\t%d\tlog' % (amin, amax, Na))
            f.write('\n')
            for i in range(Nt):
                t = tarr[i]
                f.write('\ni=%d, t=%.8e' % (i, t))
                for j in range(Nr):
                    f.write('\n')
                    for l in range(Nthe):
                        f.write('\n')
                        for k in range(Na):
                            if k == 0:
                                f.write('%.8e' % Tarr[i, j, l, k])
                            else:
                                f.write('\t%.8e' % Tarr[i, j, l, k])
        elif savelist[i_file] == 'asub':
            f.write('tmin\ttmax\tNt\t%.8e\t%.8e\t%d\tlog' % (tmin, tmax, Nt))
            f.write('\nrmin\trmax\tNr\t%.8e\t%.8e\t%d\tlog' % (rmin, rmax, Nr))
            f.write('\nthemin\tthemax\tNthe\t%.8e\t%.8e\t%d\tlinear' % (themin, themax, Nthe))
            f.write('\n')
            for i in range(Nt):
                f.write('\n')
                for j in range(Nr):
                    f.write('\n')
                    for l in range(Nthe):
                        if l == 0:
                            f.write('%.8e' % asubarr[i, j, l])
                        else:
                            f.write('\t%.8e' % asubarr[i, j, l])
