import os
from math import pi
from fpath import *

nH0_list = [1]        # [cm^-3]
lamobs_list = [3.368, 4.618]    # micron
theobs_list = [0]   # radian


for i in range(len(nH0_list)):
    nH0 = nH0_list[i]
    for j in range(len(lamobs_list)):
        lamobs = lamobs_list[j]
        for k in range(len(theobs_list)):
            theobs = theobs_list[k]
            os.system('python ' + codedir + 'Ldnu_obs.py ' +
                      '%.1e ' % nH0 + '%.2f ' % lamobs + '%.3f ' % theobs)
