import numpy as np
from scipy.interpolate import RegularGridInterpolator

seds_mosfit = np.genfromtxt('SEDs_mosfit.txt')
time_mosfit= np.genfromtxt('time_SEDs_mosfit.txt')

frequency = np.genfromtxt('frequency.txt')

def L_fn(time, freq):
    intp = RegularGridInterpolator((time_mosfit, frequency), seds_mosfit)
    return intp((time, freq))

def L_fn_vec(pairs):
    intp = RegularGridInterpolator((time_mosfit, frequency), seds_mosfit)
    return intp(pairs)
