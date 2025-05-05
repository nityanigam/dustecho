import numpy as np
import const
from fpath import *
from math import pi, log10, sqrt, log, exp, sqrt, atan, cos, sin, acos, asin
import sys

# take 3 arguments from the command line
nH0 = float(sys.argv[1])        # [cm^-3] H number density
lamobs = float(sys.argv[2])     # [um] observer's wavelength
theobs = float(sys.argv[3])      # [rad] observer's viewing angle wrt. jet axis

# adjustable parameters
thej = pi/2     # [rad] jet opening angle

# fixed ones
n0_over_nH = 1.45e-15    # dust number density over H number density
lam0 = 2.       # [um] critical wavelength for Qabs_lambda

# observer's time grid [sec]
Ntobs = 100
tobsmin = 0.1*const.pc2cm/const.C_LIGHT*(1-cos(max(theobs, thej)))
tobsmax = 365*3*60*60*24     # [sec]
tobsarr = np.logspace(log10(tobsmin), log10(tobsmax), Ntobs)
Ldnuarr = np.zeros(Ntobs, dtype=float)
xcentrarr = np.zeros(Ntobs, dtype=float)

def func_nH(r, the):      # gas density profile (r in pc)
    return nH0

def jdnu_intp(t, j_r, j_the, jdnuarr, tarr):
    # linear interpolation in time for a given r (given by index j_r)
    i_floor = np.argmin(np.abs(t-tarr))
    if t < tarr[i_floor] and i_floor != 0:
        i_floor -= 1
    if i_floor == Nt - 1:
        i_floor -= 1
    slope = (jdnuarr[i_floor+1, j_r, j_the] - jdnuarr[i_floor, j_r, j_the])\
            /(tarr[i_floor+1] - tarr[i_floor])
    return max(jdnuarr[i_floor, j_r, j_the] + slope * (t - tarr[i_floor]), 0)

# read the data for Td, asub (generated from 'generate_Td_asub')
savelist = ['Td', 'asub']   # no need for taud
for i_file in range(len(savelist)):
    fname = 'nH%.1e_' % nH0 + savelist[i_file]
    with open(savedir+fname + '.txt', 'r') as f:
        if savelist[i_file] == 'Td':
            row = f.readline().strip('\n').split('\t')
            tmin, tmax, Nt = float(row[3]), float(row[4]), int(row[5])
            row = f.readline().strip('\n').split('\t')
            rmin, rmax, Nr = float(row[3]), float(row[4]), int(row[5])
            row = f.readline().strip('\n').split('\t')
            themin, themax, Nthe = float(row[3]), float(row[4]), int(row[5])
            row = f.readline().strip('\n').split('\t')
            amin, amax, Na = float(row[3]), float(row[4]), int(row[5])

            Tarr = np.zeros((Nt, Nr, Nthe, Na), dtype=float)

            f.readline()    # skip this line
            for i in range(Nt):
                f.readline()    # skip this line
                for j in range(Nr):
                    row = f.readline().strip('\n').split('\t')
                    for l in range(Nthe):
                        row = f.readline().strip('\n').split('\t')
                        for k in range(Na):
                            Tarr[i, j, l, k] = float(row[k])
        else:   # asubarr
            row = f.readline().strip('\n').split('\t')
            tmin, tmax, Nt = float(row[3]), float(row[4]), int(row[5])
            row = f.readline().strip('\n').split('\t')
            rmin, rmax, Nr = float(row[3]), float(row[4]), int(row[5])
            row = f.readline().strip('\n').split('\t')
            themin, themax, Nthe = float(row[3]), float(row[4]), int(row[5])

            asubarr = np.zeros((Nt, Nr, Nthe), dtype=float)

            f.readline()  # skip this line
            for i in range(Nt):
                row = f.readline().strip('\n').split('\t')
                for j in range(Nr):
                    row = f.readline().strip('\n').split('\t')
                    for l in range(Nthe):
                        asubarr[i, j, l] = float(row[l])

rarr = np.logspace(log10(rmin), log10(rmax), Nr)
r_ratio = rarr[1]/rarr[0]

thearr = np.linspace(themin, themax, Nthe)
dthe = thearr[1]-thearr[0]

tarr = np.linspace(tmin, tmax, Nt, endpoint=False)
dt = tarr[1] - tarr[0]
tarr += dt/2.

aarr = np.logspace(log10(amin), log10(amax), Na)
a_ratio = aarr[1]/aarr[0]

jdnuarr = np.zeros((Nt, Nr, Nthe), dtype=float)    # volumetric emissivity at lamobs

# then we calculate the volumetric emissivity jdnuarr(t, r)
# by integrating over dust size distribution
nuobs = const.C_LIGHT/(lamobs*1e-4)  # [Hz], observer's frequency
h_nu_over_k = const.H_PLANCK*nuobs/const.K_B    # a useful constant

# Precompute logarithmic grain size array
num_a = int(np.floor(np.log(amax / amin) / np.log(a_ratio))) + 1
a_values = amin * a_ratio ** np.arange(num_a)
da_values = a_values * (np.sqrt(a_ratio) - 1. / np.sqrt(a_ratio))
k_values = np.round(np.log10(a_values / amin) / np.log10(a_ratio)).astype(int)

# Broadcast r and the arrays
r = rarr[:, None]  # shape (Nr, 1)
the = thearr[None, :]  # shape (1, Nthe)
#nH_vals = func_nH(r, the)  # shape (Nr, Nthe)
nH_vals = np.ones((Nr, Nthe)) * nH0 #TEMP
j_pre_factor = 2 * np.pi * const.H_PLANCK * nuobs / lamobs**2 * n0_over_nH * nH_vals  # shape (Nr, Nthe)

# Broadcast j_pre_factor to (Nt, Nr, Nthe)
j_pre_factor = j_pre_factor[None, :, :]  # shape (Nt, Nr, Nthe)

# Create container
j_integ = np.zeros((Nt, Nr, Nthe))

# Loop over precomputed grain sizes
for idx_a, (a, da, k) in enumerate(zip(a_values, da_values, k_values)):
    # Build a mask for where a >= asubarr[i, j, l]
    a_mask = a >= asubarr  # shape (Nt, Nr, Nthe)

    # Get T values at this grain size
    Tvals = Tarr[:, :, :, k]  # shape (Nt, Nr, Nthe)
    valid_mask = (Tvals >= 100) & a_mask

    term = np.zeros_like(Tvals)
    exp_term = np.exp(h_nu_over_k / Tvals[valid_mask]) - 1
    term[valid_mask] = da * a**-0.5 / (a + (lamobs / lam0)**2) / exp_term

    j_integ += term

# Final emission coefficient
jdnuarr = j_pre_factor * j_integ

# then we calculate observed lightcurve taking into account light-travel delay
Nmu = 100    # resolution of the bright stripe
percent = 0
for i_tobs in range(Ntobs):
    if 100*i_tobs/Ntobs > percent:
        print('%d %%' % percent)
        percent += 10
    tobs = tobsarr[i_tobs]
    Ldnu = 0.
    Ldnu_xcentr = 0.
    for j in range(Nr):
        r = rarr[j]
        dr = r * (sqrt(r_ratio) - 1/sqrt(r_ratio))
        if r*const.pc2cm < max(0, const.C_LIGHT*(tobs-tmax))/(1 - cos(theobs+thej)):
            continue    # light echo has already passed
        if theobs > thej and r*const.pc2cm > const.C_LIGHT*(tobs-tmin)/(1 - cos(theobs-thej)):
            # print('light echo hasnt arrived yet')
            continue    # light echo hasn't arrived yet
        mumin = max([cos(theobs+thej), 1 - const.C_LIGHT*(tobs-tmin)/(r*const.pc2cm)])
        mumax = min([cos(max(1e-10, theobs-thej)),
                     1 - const.C_LIGHT*(tobs-tmax)/(r*const.pc2cm)])
        if mumax < mumin:   # the region is outside the jet cone
            # print(mumax, mumin, 'mumax < mumin')
            continue
        # print('mumax-mumin', mumax-mumin)
        dmu = (mumax - mumin)/Nmu
        for k in range(Nthe):
            the = thearr[k]
            mu = mumin + dmu/2
            mu_integ = 0.
            mu_integ_xcentr = 0.
            while mu < mumax:
                if mu >= cos(max(1e-10, thej-theobs)):
                    tphimax = pi
                else:
                    phi = acos((mu - cos(theobs)*cos(thej))/(sin(theobs)*sin(thej)))
                    if abs(sin(thej)*sin(phi)/sqrt(1-mu*mu)) > 1:  # avoid round-off errors
                        tphimax = pi
                    else:
                        tphimax = asin(sin(thej)*sin(phi)/sqrt(1-mu*mu))
                t = tobs - r*const.pc2cm/const.C_LIGHT*(1-mu)
                jdnu = jdnu_intp(t, j, k, jdnuarr, tarr)
                mu_integ += dmu * jdnu * 2 * tphimax
                mu_integ_xcentr += dmu * r * sqrt(1-mu*mu) * jdnu * 2 * sin(tphimax)
                mu += dmu
            # print(mu_integ)
            Ldnu += 2*pi*dr*r*r * np.sin(the)*dthe*const.pc2cm**3 * mu_integ
            Ldnu_xcentr += 2*pi*dr*r*r * np.sin(the)*dthe*const.pc2cm**3 * mu_integ_xcentr
    Ldnuarr[i_tobs] = Ldnu
    if Ldnu < 1e10:    # ~zero flux
        xcentrarr[i_tobs] = np.nan
    else:
        xcentrarr[i_tobs] = Ldnu_xcentr/Ldnu
    # print('tobs=%.1e yr' % (tobs / const.yr2sec),
    #       'Ldnu=%.1e' % Ldnu, 'xcentr=%.1e' % xcentrarr[i_tobs])


# write the data into files
param = 'nH%.1e_lam%.2fum_theobs%.2f_' % (nH0, lamobs, theobs)
with open(savedir+param+'Ldnu_xcentr.txt', 'w') as f:
    f.write('tobsmin\ttobsmax\tNtobs\t%.8e\t%.8e\t%d' % (tobsmin, tobsmax, Ntobs))
    f.write('\nLdnu[cgs]\txcentr[pc]')
    f.write('\n')
    for i_tobs in range(Ntobs):
        if i_tobs == 0:
            f.write('%.8e' % Ldnuarr[i_tobs])
        else:
            f.write('\t%.8e' % Ldnuarr[i_tobs])
    f.write('\n')
    for i_tobs in range(Ntobs):
        if i_tobs == 0:
            f.write('%.8e' % xcentrarr[i_tobs])
        else:
            f.write('\t%.8e' % xcentrarr[i_tobs])
