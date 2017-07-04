import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import Polygon
from matplotlib.patches import PathPatch
import pylab
import math as math
import corner

from scipy.optimize import curve_fit
import scipy.optimize as optimize

from os import sys

pylab.rc('font', family='serif', size=12)    #makes pretty plots


data= np.genfromtxt('./King_resample_cluster_data.txt')
tuc= np.genfromtxt('Tuc_Data.txt')

r= tuc[:,7]
r_sort= np.sort(r)

print(data.shape)
'''
(0)a_00_00, \
(1)Mc_00_00, \
(2)Mb_00_00, \
(3)a_00_BH,\
(4)Mc_00_BH, \
(5)Mb_00_BH, \
(6)a_BH_00, \
(7)Mc_BH_00, \
(8)Mb_BH_00, \
(9)a_BH_BH, \
(10)Mc_BH_BH, \
(11)MB_BH_BH'''

a_00_00_raw = data[:,0]
Mc_00_00_raw = data[:,1]
Mb_00_00_raw = data[:,2]

a_00_BH_raw = data[:,3]
Mc_00_BH_raw = data[:,4]
Mb_00_BH_raw = data[:,5]

a_BH_00_raw = data[:,6]
Mc_BH_00_raw = data[:,7]
Mb_BH_00_raw = data[:,8]

a_BH_BH_raw = data[:,9]
Mc_BH_BH_raw = data[:,10]
Mb_BH_BH_raw = data[:,11]

lnL_00_00_raw= data[:,12]
suc_00_00= data[:,13]

lnL_00_BH_raw= data[:,14]
suc_00_BH= data[:,15]

lnL_BH_00_raw= data[:,16]
suc_BH_00= data[:,17]

lnL_BH_BH_raw= data[:,18]
suc_BH_BH= data[:,19]


a_00_00 = a_00_00_raw #[np.where(suc_00_00==1)]
Mc_00_00 = Mc_00_00_raw #[np.where(suc_00_00==1)]
Mb_00_00 = Mb_00_00_raw #[np.where(suc_00_00==1)]
lnL_00_00= lnL_00_00_raw #[np.where(suc_00_00==1)]

a_00_BH = a_00_BH_raw[np.where(suc_00_BH==1)]
Mc_00_BH = Mc_00_BH_raw[np.where(suc_00_BH==1)]
Mb_00_BH = Mb_00_BH_raw[np.where(suc_00_BH==1)]
lnL_00_BH= lnL_00_BH_raw[np.where(suc_00_BH==1)]

a_BH_00 = a_BH_00_raw[np.where(suc_BH_00==1)]
Mc_BH_00 = Mc_BH_00_raw[np.where(suc_BH_00==1)]
Mb_BH_00 = Mb_BH_00_raw[np.where(suc_BH_00==1)]
lnL_BH_00= lnL_BH_00_raw[np.where(suc_BH_00==1)]

a_BH_BH = a_BH_BH_raw[np.where(suc_BH_BH==1)]
Mc_BH_BH = Mc_BH_BH_raw[np.where(suc_BH_BH==1)]
Mb_BH_BH = Mb_BH_BH_raw[np.where(suc_BH_BH==1)]
lnL_BH_BH= lnL_BH_BH_raw[np.where(suc_BH_BH==1)]

print('a_00_00_raw: ', len(a_00_00_raw))
print('a_00_00____: ', len(a_00_00))


def fitfunc(R, rc, Mc, Mb):
    # cluster radius
    x = R / rc
    s2c = (3*math.pi/64) / np.sqrt(1+x**2)
    s2b = x**(-1.0/3.0)*3*3.1415/64*(2.5+x**2)**(0.2)*((3*3.1415/64)**(15.0/8.0)+x**2)**(-8.0/15.0)
    return (Mc * s2c + Mb * s2b)

a_00_00_mean = np.mean(a_00_00)
a_00_BH_mean = np.mean(a_00_BH)
a_BH_00_mean = np.mean(a_BH_00)
a_BH_BH_mean = np.mean(a_BH_BH)

Mc_00_00_mean = np.mean(Mc_00_00)
Mc_00_BH_mean = np.mean(Mc_00_BH)
Mc_BH_00_mean = np.mean(Mc_BH_00)
Mc_BH_BH_mean = np.mean(Mc_BH_BH)

Mb_00_00_mean = np.mean(Mb_00_00)
Mb_00_BH_mean = np.mean(Mb_00_BH)
Mb_BH_00_mean = np.mean(Mb_BH_00)
Mb_BH_BH_mean = np.mean(Mb_BH_BH)



#plt.plot(r_sort, fitfunc(r_sort, a0, Mc0, Mb0), label='00-model')
plt.plot(r_sort, fitfunc(r_sort, a_00_00_mean, Mc_00_00_mean, Mb_00_00_mean), label='00-fit')
plt.plot(r_sort, fitfunc(r_sort, a_00_BH_mean, Mc_00_BH_mean, Mb_00_BH_mean), label='BH-fit')
plt.legend()
plt.semilogx()
#plt.show()
plt.savefig('./Plots/fitfunc_00.png')
plt.close()

#plt.plot(r_sort, fitfunc(r_sort, aBH, McBH, MbBH), label='BH-model')
plt.plot(r_sort, fitfunc(r_sort, a_BH_00_mean, Mc_BH_00_mean, Mb_BH_00_mean), label='00-fit')
plt.plot(r_sort, fitfunc(r_sort, a_BH_BH_mean, Mc_BH_BH_mean, Mb_BH_BH_mean), label='BH-fit')
plt.legend()
plt.semilogx()
#plt.show()
plt.savefig('./Plots/fitfunc_BH.png')
plt.close()

print(len(a_BH_BH[np.where(a_BH_BH>300)]))
print(a_BH_00.max())

plt.hist(a_00_00, bins=50, alpha=0.3, label='00-fit')
plt.hist(a_00_BH, bins=50, alpha=0.3, label='BH_fit')
plt.legend()
plt.title('a_00-fit Histogram')
plt.savefig('./Plots/hist_a_00.png')
#plt.show()
plt.close()

plt.hist(a_BH_00, bins=50, alpha=0.3, label='00-fit')

plt.hist(a_BH_BH[np.where(a_BH_BH<300)], bins=50, alpha=0.3, label='BH_fit')
plt.legend()
plt.title('a_BH-fit Histogram')
plt.savefig('./Plots/hist_a_BH.png')
#plt.show()
plt.close()

plt.hist(Mc_00_00, bins=50, alpha=0.3, label='00-fit')
plt.hist(Mc_00_BH, bins=200, alpha=0.3, label='BH_fit')
plt.legend()
plt.title('Mc_00-fit Histogram')
plt.savefig('./Plots/hist_Mc_00.png')
#plt.show()
plt.close()

plt.hist(Mc_BH_00, bins=50, alpha=0.3, label='00-fit')
plt.hist(Mc_BH_BH, bins=200, alpha=0.3, label='BH_fit')
plt.legend()
plt.title('Mc_BH-fit Histogram')
plt.savefig('./Plots/hist_Mc_BH.png')
#plt.show()
plt.close()

#plt.hist(Mb_00_00, bins=50, alpha=0.3, label='00-fit')
plt.hist(Mb_00_BH, bins=50, alpha=0.3, label='BH_fit')
plt.legend()
plt.title('Mb_00-fit Histogram')
plt.savefig('./Plots/hist_Mb_00_BH.png')
#plt.show()
plt.close()

#plt.hist(Mb_BH_00, bins=50, alpha=0.3, label='00-fit')
plt.hist(Mb_BH_BH, bins=50, alpha=0.3, label='BH_fit')
plt.legend()
plt.title('Mb_BH-fit Histogram')
plt.savefig('./Plots/hist_Mb_BH_BH.png')
#plt.show()
plt.close()

#plt.hist(lnL_00_00, bins=50, alpha=0.3, label='00-fit')
plt.hist(lnL_00_BH, bins=50, alpha=0.3, label='BH_fit')
plt.legend()
plt.title('Likelihood_00 Histogram')
plt.savefig('./Plots/hist_lnL_00.png')
#plt.show()
plt.close()

#plt.hist(lnL_BH_00, bins=50, alpha=0.3, label='00-fit')
plt.hist(lnL_BH_BH, bins=50, alpha=0.3, label='BH_fit')
plt.axvline(x=246041, label='47 Tuc Value')
plt.legend()
plt.title('Likelihood_BH Histogram')
plt.savefig('./Plots/hist_lnL_BH_BH.png')
#plt.show()
plt.close()

#plt.hist(Mb_00_00, bins=50, alpha=0.3, label='00-fit')
plt.hist(Mb_00_BH/Mc_00_BH, bins=50, alpha=0.3, label='BH Plummerfit')
plt.legend()
plt.title(r'$Mb/Mc$ for $N=10^6$ randomly generated no black hole clusters.')
plt.savefig('./Plots/hist_Mb-Mc_00.png')
#plt.show()
plt.close()

#plt.hist(Mb_BH_00/Mc_BH_00, bins=50, alpha=0.3, label='00-fit')
plt.hist(Mb_BH_BH/Mc_BH_BH, bins=50, alpha=0.3, label='BH Plummer Fit')
plt.legend()
plt.title(r'$Mb/Mc$ for $N=10^6$ randomly generated black hole clusters.')
plt.savefig('./Plots/hist_Mb-Mc_BH.png')
#plt.show()
plt.close()