import os
from math import pi
from fpath import *
import numpy as np

nH0_list = [1]        # [cm^-3]
lamobs_list = [1.14]    # micron
theobs_list = [0]   # radian
thej_list = pi*np.linspace(0.001, 0.5, 8)   #radian

for i in range(len(nH0_list)):
    nH0 = nH0_list[i]
    for j in range(len(thej_list)):
        thej = thej_list[j]
        for k in range(len(lamobs_list)):
            lamobs = lamobs_list[k]
            for l in range(len(theobs_list)):
                theobs = theobs_list[l]
                os.system('python ' + codedir + '/Ldnu_obs.py ' +
                          '%.1e ' % nH0 + '%.2f ' % lamobs + 
                          '%.3f ' % theobs + '%.4f ' % thej)
