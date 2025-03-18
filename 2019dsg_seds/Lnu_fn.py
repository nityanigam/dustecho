import numpy as np
import astropy.constants as c
import astropy.units as u

seds_EH = np.genfromtxt('SEDs_Hammerstein2023.txt')
time_EH = np.genfromtxt('time_SEDs_Hammerstein2023.txt')

seds_mosfit = np.genfromtxt('SEDs_mosfit.txt')
time_mosfit= np.genfromtxt('time_SEDs_mosfit.txt')

frequency = np.genfromtxt('frequency.txt')

tmin =  time_mosfit[0]
tmax = time_mosfit[-1]

from scipy.interpolate import RegularGridInterpolator

def L_fn(time, freq):
    intp = RegularGridInterpolator((time_mosfit, frequency), seds_mosfit)
    return intp((time, freq))

def L_fn_vec(pairs):
    intp = RegularGridInterpolator((time_mosfit, frequency), seds_mosfit)
    return intp(pairs)
