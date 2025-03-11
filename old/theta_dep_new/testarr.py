import numpy as np
import const
from fpath import *
from math import pi, log10, sqrt, log, exp, sqrt, atan, cos, sin, acos, asin
import sys

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

# take 3 arguments from the command line
nH0 = 1        # [cm^-3] H number density
lamobs = 1.14    # [um] observer's wavelength
theobs = 20*pi/180     # [rad] observer's viewing angle wrt. jet axis

# adjustable parameters
thej = pi/2     # [rad] jet opening angle

# fixed ones
n0_over_nH = 1.45e-15    # dust number density over H number density
lam0 = 2.       # [um] critical wavelength for Qabs_lambda

themin, themax = 0.000001, pi/2    #theta array
Nthe = 10
thearr = np.linspace(themin, themax, Nthe)
Dthe = thearr[1]-thearr[0]

Nt = 20     # this is the dimension we interpolate over
tarr = np.linspace(tmin, tmax, Nt, endpoint=False)
dt = tarr[1] - tarr[0]
tarr += dt/2.

# observer's time grid [sec]
Ntobs = 100
tobsmin = 0.1*const.pc2cm/const.C_LIGHT*(1-cos(max(theobs, thej)))
tobsmax = 4e3*tobsmin     # [sec]
tobsarr = np.logspace(log10(tobsmin), log10(tobsmax), Ntobs)
Ldnuarr = np.zeros(Ntobs, dtype=float)
xcentrarr = np.zeros(Ntobs, dtype=float)


def func_nH(r, the):      # gas density profile (r in pc)
    return nH0


def jdnu_intp(t, j_r, jdnuarr, tarr, j_the):
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
            themin, themax, Nthe = float(row[3]), float(row[4]), int(row[5])
            row = f.readline().strip('\n').split('\t')
            rmin, rmax, Nr = float(row[3]), float(row[4]), int(row[5])
            row = f.readline().strip('\n').split('\t')
            amin, amax, Na = float(row[3]), float(row[4]), int(row[5])

            Tarr = np.zeros((Nt, Nr, Na, Nthe), dtype=float)

            f.readline()    # skip this line
            for i in range(Nt):
                f.readline()    # skip this line
                for l in range(Nthe):
                    f.readline()
                    for j in range(Nr):
                        row = f.readline().strip('\n').split('\t')
                        for k in range(Na):
                            Tarr[i, j, k, l] = float(row[k])
        # elif savelist[i_file] == 'taud':      # this file is not used for echo lightcurve
        #     row = f.readline().strip('\n').split('\t')
        #     numin, numax, Nnu = float(row[3]), float(row[4]), int(row[5])
        #     row = f.readline().strip('\n').split('\t')
        #     rmin, rmax, Nr = float(row[3]), float(row[4]), int(row[5])
        #
        #     taudarr = np.zeros((Nnu, Nr), dtype=float)
        #
        #     f.readline()    # skip this line
        #     for m in range(Nnu):
        #         row = f.readline().strip('\n').split('\t')
        #         for j in range(Nr):
        #             taudarr[m, j] = float(row[j])
        else:   # asubarr
            row = f.readline().strip('\n').split('\t')
            tmin, tmax, Nt = float(row[3]), float(row[4]), int(row[5])
            row = f.readline().strip('\n').split('\t')
            themin, themax, Nthe = float(row[3]), float(row[4]), int(row[5])
            row = f.readline().strip('\n').split('\t')
            rmin, rmax, Nr = float(row[3]), float(row[4]), int(row[5])

            asubarr = np.zeros((Nt, Nr, Nthe), dtype=float)

            f.readline()  # skip this line
            for i in range(Nt):
                f.readline()
                for l in range(Nthe):
                    row = f.readline().strip('\n').split('\t') #CHECK THIS
                    for j in range(Nr):
                        asubarr[i, j, l] = float(row[j])

# Create weights based on sin(theta) for solid angle integration
weights = np.sin(thearr)

# Normalize weights so that they sum to 1
weights /= np.sum(weights)

# Perform weighted average over the theta axis (the last axis in Tarr)
# This collapses the theta axis, resulting in a (Nt, Nr, Na) array
Tarr_avg = np.sum(Tarr * weights[None, None, None, :], axis=-1)


savelist = ['Tavg']
for i_file in range(len(savelist)):
    fname = 'nH%.1e_' % nH0 + savelist[i_file]
    with open(savedir + fname + '.txt', 'w') as f:
        if savelist[i_file] == 'Tavg':
            f.write('tmin\ttmax\tNt\t%.8e\t%.8e\t%d\tlinear' % (tmin, tmax, Nt))
            f.write('\nrmin\trmax\tNr\t%.8e\t%.8e\t%d\tlog' % (rmin, rmax, Nr))
            f.write('\namin\tamax\tNa\t%.8e\t%.8e\t%d\tlog' % (amin, amax, Na))
            f.write('\n')
            for i in range(Nt):
                t = tarr[i]
                f.write('\ni=%d, t=%.8e' % (i, t))
                for j in range(Nr):
                    f.write('\n')
                    for k in range(Na):
                        if k == 0:
                            f.write('%.8e' % Tarr_avg[i, j, k])
                        else:
                            f.write('\t%.8e' % Tarr_avg[i, j, k])
