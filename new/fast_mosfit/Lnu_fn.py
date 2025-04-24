import numpy as np

seds_mosfit = np.genfromtxt('SEDs_mosfit.txt')
time_mosfit= np.genfromtxt('time_SEDs_mosfit.txt')
time_mosfit -= time_mosfit[0]
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
