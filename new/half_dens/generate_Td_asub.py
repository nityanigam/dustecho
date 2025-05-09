import numpy as np
import numpy as np
import const
from fpath import *
from scipy.interpolate import interp1d
from scipy.interpolate import griddata
from math import pi, log10, sqrt, log, exp, sqrt, atan, cos, sin, acos, asin

# adjustable parameters
nH0 = 0.5                # [cm^-3] H number density
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
Nthe = 5
thearr = np.linspace(themin, themax, Nthe)
dthe = thearr[1]-thearr[0]

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

Tarr = np.zeros((Nt, Nr, Nthe, Na), dtype=float)   # to store the dust temperature
asubarr = np.zeros((Nt, Nr, Nthe), dtype=float)    # to store sublimation radii
taudarr = np.zeros((Nnu, Nr, Nthe), dtype=float)   # dust extinction optical depth at each nu

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
for i in range(Nthe_fine):
    NH = 0.     # total H number
    the = thearr_fine[i]
    for j in range(Nr_fine):
        r = rarr_fine[j]
        dr = r * (sqrt(r_ratio_fine) - 1/sqrt(r_ratio_fine))
        NH += 2*pi*r*r*dr*np.sin(the)*dthe_fine*const.pc2cm**3 * func_nH(r, the)
        NHarr_fine[j][i] = NH
    r_NH_intp = interp1d(rarr_fine, NHarr_fine[:,i], fill_value='extrapolate')
    interp_fns[i] = r_NH_intp

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

Nion = 0.
for i in range(Nt_fine):
    t = tarr_fine[i]
    Lion = 0.
    for j in range(Nnu):
        nu = nuarr_fine[j]
        dnu = nu * (sqrt(nu_ratio_fine) - 1/sqrt(nu_ratio_fine))
        Lion += dnu * func_Lnu(t, nu)/(const.H_PLANCK*nu)
    Nion += Lion*dt_fine
    Nionarr_fine[i] = Nion
Nion_t_intp = interp1d(Nionarr_fine, tarr_fine, fill_value='extrapolate')

# calculate dust temperature at a given time
# calculate dust temperature at a given time
def calculate_all_T(t, i_t, rion, Tarr, asubarr, taudarr):
    for l in range(Nthe):
        for j in range(Nr):
            r = rarr[j]
            asub = amin
            for k in range(Na):
                a = aarr[k]
                # calculate heating rate divided by pi*a^2
                qdot_h_over_pia2 = 0.
                for m in range(Nnu):
                    nu = nuarr[m]
                    dnu = nu * (sqrt(nu_ratio) - 1./sqrt(nu_ratio))
                    if r < rion and nu > nu_ion_min:
                        break   # no photons above 13.6 eV ar r > rion
                    qdot_h_over_pia2 += dnu * func_Lnu(t, nu) * exp(-taudarr[m, j, l])\
                                        /(4*pi*r*r*const.pc2cm**2) * func_Qabs(nu, a)
                qdot_h = qdot_h_over_pia2 * pi * a**2 * 1e-8    # heating rate [cgs]
                T = func_T(qdot_h, a)
                Tsub = 2.33e3 * (1 - 0.033*log(t/100./a))
                if T > Tsub:    # grains in this size bin have already evaporated
                    asub = min(max(asub, a), amax)
                    Tarr[i_t, j, l, k] = 0.   # no emission from this size bin
                else:   # grains survive
                    Tarr[i_t, j, l, k] = T
            if i_t == 0:
                asubarr[i_t, j, l] = asub
            else:   # make sure asub does not decrease with time
                asubarr[i_t, j, l] = max(asub, asubarr[i_t-1, j, l])
    return      

def calculate_taudarr(i_t, asubarr, taudarr, thearr):
    for m in range(Nnu):
        nu = nuarr[m]
        lam = const.C_LIGHT/nu*1e4   # in um
        xmax = amax * (lam0/lam)**2
        tau_pre_factor = 2*sqrt(2)*pi*1e-8*n0_over_nH*lam0/lam
        for k in range(Nthe):
            taud = 0.
            for j in range(Nr):
                r = rarr[j]
                the = thearr[k]
                dr = r * (sqrt(r_ratio) - 1./sqrt(r_ratio))
                xsub = asubarr[i_t, j, k] * (lam0/lam)**2
                taud += tau_pre_factor * dr * const.pc2cm * func_nH(r, the) \
                        * (atan(sqrt(0.5*xmax)) - atan(sqrt(0.5*xsub)))
                taudarr[m, j, k] = taud
    return

tol = 0.01  # tolerance for asubarr
for i in range(Nt):
    for k in range(Nthe):
        t = tarr[i]
        the = thearr[k]
        Nion = Nion_t_intp(t)
        rion = interp_fns[int(k*fine_spacing)](Nion)  # ionization radius
        taudarr.fill(0)    # first iteration, no dust extinction
        calculate_all_T(t, i, rion, Tarr, asubarr, taudarr)
        frac_diff = 1.  # convergence criterion
        n_iter = 0.     # number of iterations
        while frac_diff > tol:
            n_iter += 1
            asubarr_old = np.copy(asubarr[i])
            # we go back to calculate dust extinction optical depth
            calculate_taudarr(i, asubarr, taudarr, thearr)
            # then calculate the whole temperature again
            calculate_all_T(t, i, rion, Tarr, asubarr, taudarr)
            frac_diff = 0.
            for j in range(Nr):
                frac_diff = max(frac_diff, abs(asubarr_old[j, k]/asubarr[i, j, k] - 1))
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
