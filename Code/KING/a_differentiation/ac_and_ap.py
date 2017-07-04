import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import Polygon
from matplotlib.patches import PathPatch
import pylab
import math as math
from mpl_toolkits.mplot3d import Axes3D

from scipy.optimize import curve_fit
import scipy.optimize as optimize
from scipy.integrate import quad

pylab.rc('font', family='serif', size=12)               #makes pretty plots

#### Define Functions
#  rho(r)
def rho(r, a): ###
    rho= ((r ** 2 + a ** 2) ** (-3 / 2) - (rt ** 2 + a ** 2) ** (-3 / 2))
    if rho<0:
        return 0
    else:
        return rho          # rho(r)

def IntMcr(r,a): ###
    return (4*np.pi *rho(r,a)*r**2)

# M(r)
def M(rp,a): ###
    Mc = quad(IntMcr, 0, rp, args=(a))[0]
    return Mc                                           # M(r)


def Intsigma0(r,ac, ap): ###
    return (rho(r,ap)*M(r,ac)*G/r**2)

def IntsigmaBH(r, ap, Mbh): ###
    return rho(r,ap )*Mbh *G/ r**2


# sigma_0^2(r)
def sigma0(rp, ac, ap): ###
    sigma = quad(Intsigma0, rp, rt,  args=(ac, ap))[0]
    #sigma=sigmaA/rho(rp, ap)
    return sigma

# sigma_BH^2(r)
def sigmaBH(rp, ap, Mbh): ###
    sigma =quad(IntsigmaBH,rp, rt, args=( ap, Mbh))[0]
    #sigma= sigmaB/rho(rp, ap)
    return sigma

def Int_sigmaR0(r, R, ac, ap): ###
    return (sigma0(r, ac, ap)*r/np.sqrt(r**2-R**2))

def norm_int_sigma0(r, R, ap): ###
    return rho(r,ap)*r/np.sqrt(r**2-R**2)

# sigma_0^2(R)
def sigmaR0(R, ac, ap): ###
    above = quad(Int_sigmaR0, R, np.inf, args=(R, ac, ap)) [0]
    below = quad(norm_int_sigma0, R, np.inf, args=(R, ap)) [0]
    sigma2=  above / below
    return sigma2

def Int_sigmaRBH(r, R, ap, Mbh): ###
    return (sigmaBH(r, ap, Mbh)*r/np.sqrt(r**2-R**2))

# sigma_BH^2(R)
def sigmaRBH(R, ap, Mbh): ###
    above = quad(Int_sigmaRBH, R, np.inf, args=(R, ap, Mbh))[0]
    below = quad(norm_int_sigma0, R, np.inf, args=(R, ap))[0]
    sigma2 = above/below
    return sigma2

pi=np.pi
def s2cl_3d_analyt(r,a):
    # print "Running analytic s2 for cluster (3D) ..."
    K=1
    return -6*(2*(r**2+1)**(3/2)*r*(rt**2+1)**(7/2)*np.log(rt*(rt**2+1)**(1/2)+rt**2+1)-2*(r**2+1)**(3/2)*r*(rt**2+1)**(7/2)*np.log(r**2+1+r*(r**2+1)**(1/2))+2*r*(r**2+1)**(3/2)*(rt**2+1)**2*(rt**2*(rt**2+1)**(1/2)+(rt**2+1)**(1/2)-1)*np.log(rt+(rt**2+1)**(1/2)-1)+2*r*(r**2+1)**(3/2)*(rt**2+1)**2*(rt**2*(rt**2+1)**(1/2)+(rt**2+1)**(1/2)+1)*np.log(rt+(rt**2+1)**(1/2)+1)-2*(r**2+1)**(3/2)*r*(rt**2+1)**2*(rt**2*(rt**2+1)**(1/2)+(rt**2+1)**(1/2)-1)*np.log(r+(r**2+1)**(1/2)-1)-2*(r**2+1)**(3/2)*r*(rt**2+1)**2*(rt**2*(rt**2+1)**(1/2)+(rt**2+1)**(1/2)+1)*np.log(r+(r**2+1)**(1/2)+1)-2*r*(r**2+1)**(3/2)*(rt**2+1)**2*np.arctanh(1/(rt**2+1)**(1/2))+2*(r**2+1)**(3/2)*r*(rt**2+1)**2*np.arctanh(1/(r**2+1)**(1/2))+(r*((rt**2+1)**3*(r**2+1)*np.log(rt**2+1)-(rt**2+1)**3*(r**2+1)*np.log(r**2+1)+4*(rt**2+1)**3*(r**2+1)*np.arcsinh(r)-4*(rt**2+1)**3*(r**2+1)*np.arcsinh(rt)-2*(rt**2+1)**3*(r**2+1)*np.log(rt)+2*(rt**2+1)**3*(r**2+1)*np.log(r)-1/3*r**4+(-rt**4-rt**2-2/3)*r**2+rt**6+2*rt**4+2*rt**2+2/3)*(rt**2+1)**(1/2)-4*(r**2+1)*(rt**2+1)**2*(1/2*np.arcsinh(r)+np.arcsinh(rt)*rt*r*(rt**2+3/2)))*(r**2+1)**(1/2)+4*((rt**2+1)**(3/2)*np.arcsinh(r)*(r**2+1/2)-1/6*r)*(r**2+1)*(rt**2+1)**2)*K*pi/(3*(r**2+1)**(3/2)*r*(rt**2+1)**2-3*(rt**2+1)**(7/2)*r)

def s2bh_3d_analyt(r,a,Mbh):
    # print "Running analytic s2 for BH (3D) ..."
    pref = Mbh*(r**2+a**2)
    num1 = -1*(rt**2+a**2)**(3/2) *(2*r**2+a**2)
    num2 = (a**4 + 3*a**2 *r*rt + 2*r*rt**3) *(r**2+a**2)**(1/2)
    denom = a**4 *r *( (r**2+a**2)**(3/2) - (rt**2+a**2)**(3/2) )
    return pref*(num1+num2)/denom


### Import Data

data= np.genfromtxt('../Tuc_Data.txt')

#fit0_params= np.genfromtxt('../Tuc_fit0.txt')
#fitBH_params= np.genfromtxt('../Tuc_fitBH.txt')

Vlim=50

D= Dist = 4.29*3.086e16 # (kpc->km)
Dm= D*1000
G = 6.67e-11

x= data[:, 0]
y= data[:, 1]
vx= data[:, 5]
vy=data[:, 6]
r= data[:, 7]
v_total = data[:, 8]
evx= data[:, 9]
evy= data[:, 10]
weight= data[:,11]

x=x[np.where(v_total<=50)]
y=y[np.where(v_total<=50)]
vx=vx[np.where(v_total<=50)]
vy=vy[np.where(v_total<=50)]
r=r[np.where(v_total<=50)]
evx=evx[np.where(v_total<=50)]
evy=evy[np.where(v_total<=50)]
weight=weight[np.where(v_total<=50)]
v_total=v_total[np.where(v_total<=50)]

'''
Mc0= fit0_params[0]
Mb0= fit0_params[1]
a0 = fit0_params[2]

McBH= fitBH_params[0]
MbBH= fitBH_params[1]
aBH = fitBH_params[2]
'''
rt=2500
R= np.logspace(-2, np.log10(.95*rt), num=10)

a=1
ac=1
#ap=1

Mc= M(rt, ac)
Mbh= Mc

### Fit Surface density to find ap (a_pm)
Rbins = np.arange(0, 120, 5)
binN = len(Rbins) - 1
vol = np.zeros(binN)


def surf_king(R, K, aPM):
    rt = 57 * aPM
    return K*(R**2-rt**2)** 2 /(-(aPM**2+rt**2) * (R**2-rt**2))**(1/2)/(aPM**2+R**2)/(aPM**2+rt**2)


Rmid = Rbins[:-1] + 0.5 * (Rbins[1:] - Rbins[:-1])
histR, binR, patches = plt.hist(r, Rbins)
indR = np.digitize(r, Rbins)
count = np.zeros(len(Rbins) - 1)

for i in range(binN):
    vol[i] = (2.) * math.pi * (Rbins[i + 1] ** 2. - Rbins[i] ** 2)
    count[i] = np.sum(weight[np.where(indR == (i + 1))])

density = count / vol
plt.close()

ap_fit = curve_fit(surf_king, Rmid, density, p0=(5, 40), bounds=([-np.inf, 0], [np.inf, np.inf]))
print(ap_fit[0])
plt.scatter(Rmid, density)
plt.plot(Rmid, surf_king(Rmid, *ap_fit[0]))
ap_params = ap_fit[0]
plt.text(1.2, 2.3, r'$a_p=${0:.2f}  $K=${1:.2f}'.format(ap_params[1], ap_params[0]))
plt.semilogx()
plt.savefig('surface_density_ap.png')
plt.close()

ap= ap_params[1]


### make King Model

mr, sigma_0, sigma_BH= np.zeros(len(R)), np.zeros(len(R)), np.zeros(len(R))
sigma0_R= np.zeros(len(R))
sigmaR_BH= np.zeros(len(R))

N = len(R)
'''
print('start integration loop...')
for i in range(N):
    mr[i]= M(R[i],ac)
    sigma_0[i]=sigma0(R[i], ac, ap)/rho(R[i], ap)
    sigma_BH[i]= sigmaBH(R[i], ap, Mbh)/rho(R[i], ap)
    sigmaR_BH[i]=sigmaRBH(R[i], ap,  Mbh)
    sigma0_R[i] =sigmaR0( R[i], ac, ap)
    print(i)

plt.plot(R, sigma_0/G, label= 'sigma0')
plt.plot(R, sigma_BH/G, label= 'sigma_BH')
plt.plot(R, sigmaR_BH/G, label='sigmaR_BH')
plt.plot(R, sigma0_R/G, label= 'sigmaR_0')
plt.legend()
plt.loglog()
#plt.savefig('king_model_ap37.png')
#plt.show()
plt.close()
'''

ac_loop = np.arange(1, 151, 15)
#'''

sigma_a=[]
sigmaBH_a=[]


for AC in ac_loop:
    print('loop ac={0}'.format(AC))
    factor= G*M(rt, AC)
    Mbh= M(rt, AC)
    for i in range(N):
        sigma_loop = np.zeros(len(R))
        sigmaBH_loop = np.zeros(len(R))
        #print('sigma loop {0}'.format(i))
        sigma_loop[i]= sigmaR0(R[i], AC, ap)/factor
        sigmaBH_loop[i]= sigmaRBH(R[i], ap, Mbh)/factor
        print(sigma_loop[i], sigmaBH_loop[i])

    sigma_a.append(sigma_loop)
    sigmaBH_a.append(sigmaBH_loop)

np.savetxt('sigma_a.txt', np.transpose(sigma_a), header='vertigal: rising a, horizontal: rising r')
np.savetxt('sigmaBH_a.txt', np.transpose(sigmaBH_a), header='vertical: rising a, horizontal: rising r')
np.savetxt('a_values.txt', ac_loop)
np.savetxt('R_values.txt', np.log10(R))

#'''

s2cl_a= np.genfromtxt('sigma_a.txt')
s2bh_a= np.genfromtxt('sigmaBH_a.txt')

Xbh, Ybh = np.meshgrid(np.log10(R), ac_loop)

fig = plt.figure()
ax = fig.gca(projection='3d')

ax.plot_wireframe(Xbh, Ybh, np.transpose(np.log10(s2cl_a)))
ax.plot_wireframe(Xbh, Ybh, np.transpose(np.log10(s2bh_a)))
plt.savefig('sigma(ac, r).png')
plt.show()
plt.close()






#plt.plot(R, sigma1/G, label='ac=1')
#plt.plot(R, sigma5/G, label='ac=5')
#plt.plot(R, sigma10/G, label='ac=10')
#plt.plot(R, sigma50/G, label='ac=50')
#plt.plot(R, sigma100/G, label='ac=100')
#plt.legend()
#plt.loglog()
#plt.show()