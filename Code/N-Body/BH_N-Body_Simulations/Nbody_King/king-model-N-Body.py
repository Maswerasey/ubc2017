import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import Polygon
from matplotlib.patches import PathPatch
import pylab
import math as math

from scipy.optimize import curve_fit
import scipy.optimize as optimize
from scipy.integrate import quad

pylab.rc('font', family='serif', size=12)               #makes pretty plots

#  rho(r)
def rho(r):
    ''' 
    :param r: distance to core
    :param rt: tidal radius
    :param a: a
    :return: density rho(r) 
    '''
    return ((r**2+a**2)**(-3/2) - (rt**2+a**2)**(-3/2))             # rho(r)

def IntMcr(r):
    '''
    :param r: r
    :return: Integral function for cluster mass (encompassed at r)
    '''
    return (4*np.pi *rho(r)*r**2)

def Intsigma0(r):
    '''
    :param r: 
    :return: Integral function for velocity dispersion squared at r with no black hole
    '''
    return (rho(r)*M(r)/r**2)

def IntsigmaBH(r, Mbh):
    '''
    :param r: 
    :param Mbh: 
    :return: Integral function for velocity dispersin squared incl black hole
    '''
    return (rho(r)*Mbh / r**2)

# M(r)
def M(rp):
    '''
    :param rp: r (distance of star from centre)
    :return: Enclosed Mass ar r=rp
    '''
    Mc , Mc_err= quad(IntMcr, 0, rp)

    return Mc                                           # M(r)

# sigma_0^2(r)
def sigma0(rp):
    '''
    :param rp: r (distance to cluster centre)
    :return: velocity dispersion squared at r=rp
    '''
    sigmaA = -(quad(Intsigma0, 0, rp)[0] - quad(Intsigma0, 0, rt)[0])
    sigma = sigmaA /rho(rp)
    return sigma

# sigma_BH^2(r)
def sigmaBH(rp, Mbh):
    '''
    :param rp: r distance to cluster centre
    :param Mbh: Black Hole Mass
    :return: velocity dispersion squared due to Black hole
    '''
    sigma =-(quad(IntsigmaBH,0, rp, args=( Mbh))[0] - quad(IntsigmaBH,0, rt, args=(Mbh))[0])
    sigmaB = sigma /rho(rp)
    return sigmaB

def Int_sigmaR0(r, R):
    '''
    :param r: integration parameter r  
    :param R: projected distance to core
    :return: part of integral function for projected velocity dispersion at projected radius R
    '''
    return (s2cl_3d_analyt(r,a)*rho(r)*r/(r**2-R**2)**0.5)

def norm_int_sigma0(r, R):
    '''
    :param r: 
    :param R: 
    :return: integral function to normalize velocity dispersion
    '''
    return rho(r)*r/(r**2-R**2)**0.5

# sigma_0^2(R)
def sigmaR0(R):
    '''
    :param R: projected distance R
    :return: projected velocity dispersion at R for no black hole
    '''
    above, above_err= quad(Int_sigmaR0, R, rt, args=(R))
    below, below_err= quad(norm_int_sigma0, R, rt, args=(R))
    sigma2=  above / below
    return sigma2

def Int_sigmaRBH(r, R, Mbh):
    '''
    :param r: integration parameter r  
    :param R: projected distance to core
    :param Mbh:
    :return: part of integral function for projected velocity dispersion at projected radius R
    '''
    return (s2bh_3d_analyt(r,a,Mbh)*rho(r)*r/(r**2-R**2)**0.5)

# sigma_BH^2(R)
def sigmaRBH(R, Mbh):
    '''
    :param R: projected distance R
    :return: projected velocity dispersion at R for no black hole
    '''
    above = quad(Int_sigmaRBH, R, rt, args=(R, Mbh))[0]
    below = quad(norm_int_sigma0, R, rt, args=(R))[0]
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


BH00= np.genfromtxt('nbodymod0.0imbh_mod_comp')
BH05= np.genfromtxt('nbodymod0.5imbh_mod_comp')
BH10= np.genfromtxt('nbodymod1.0imbh_mod_comp')

Dmod= 5.3232200/(277.9728924*(1/3600)*math.pi/180)
print(Dmod)                                            #Distance to cluster from simulation
D= 4.57                                                #Our assumed distance

r00_mod = BH00[:,2]
v00 = BH00[:,5]

r05_mod = BH05[:,2]
v05 = BH05[:,5]

r10_mod = BH10[:,2]
v10 = BH10[:,5]

#'''
r00= r00_mod*Dmod/D
r05= r05_mod*Dmod/D
r10= r10_mod*Dmod/D


#data= np.genfromtxt('../Tuc_Data.txt')

fit0_params= np.genfromtxt('./Tuc_fit0.txt')
fitBH_params= np.genfromtxt('./Tuc_fitBH.txt')

Vlim=50

D= Dist = 4.29*3.086e16 # (kpc->km)
Dm= D*1000

#x= data[:, 0]
#y= data[:, 1]
vx= v10
vy=np.zeros(len(vx))
r= r10
v_total = v10
evx= np.zeros(len(vx))
evy= np.zeros(len(vy))
weight= 1

#x=x[np.where(v_total<=50)]
#y=y[np.where(v_total<=50)]
#vx=vx[np.where(v_total<=50)]
#vy=vy[np.where(v_total<=50)]
#r=r[np.where(v_total<=50)]
#evx=evx[np.where(v_total<=50)]
#evy=evy[np.where(v_total<=50)]
#weight=weight[np.where(v_total<=50)]
#v_total=v_total[np.where(v_total<=50)]


Mc0= fit0_params[0]
Mb0= fit0_params[1]
a0 = fit0_params[2]

McBH= fitBH_params[0]
MbBH= fitBH_params[1]
aBH = fitBH_params[2]

rt= 57 #864500
R= np.logspace(-2, np.log10(.95*rt), num=100)
a=1
Mc= M(rt)
Mbh= Mc


mr, sigma_0, sigma_BH= np.zeros(len(R)), np.zeros(len(R)), np.zeros(len(R))
sigma0_R= np.zeros(len(R))
sigmaR_BH= np.zeros(len(R))
N = len(R)

for i in range(N):
    mr[i]= M(R[i])
 #   sigma_0[i]=sigma0(R[i])
 #   sigma_BH[i]= sigmaBH(R[i], Mbh)
    sigmaR_BH[i]=sigmaRBH(R[i], Mbh)
    sigma0_R[i] =sigmaR0( R[i])
    print(i)


'''
# plot rho(r)
plt.plot(R, rho(R), label= r'$r_t=30$')
plt.semilogx()
plt.title('King Model density')
plt.xlabel(r'$r$')
plt.ylabel(r'$\rho (r)$')
plt.legend()
plt.savefig('./Plots/rho(r).png')
plt.close()

#plot M(r)
plt.plot(R, mr, label=r'$r_t=30$')
plt.semilogx()
plt.title('King-Model M(r)')
plt.xlabel(r'$r$')
plt.ylabel(r'$M(r)$')
plt.legend()
plt.savefig('./Plots/M(r).png')
plt.close()

#plot sigmaR
plt.plot(R, sigma_0, label=r'$r_t=30, M_{BH}=00\%$')
plt.plot(R, sigma_BH, label=r'$r_t=30, M_{BH}=100\%$')
plt.loglog()
plt.title(r'King-Model $\sigma^2(R)$')
plt.xlabel(r'$R$')
plt.ylabel(r'$\sigma^2(R)$')
plt.legend()
plt.savefig('./Plots/sigma0.png')
plt.close()

plt.plot(R, sigma0_R, label=r'$r_t=30$, no BH')
plt.plot(R, sigmaR_BH, label= r'$r_t=30$, 100% BH')
plt.semilogx()
plt.title(r'King-Model projected $\sigma(R)^2$')
plt.xlabel(r'$R$')
plt.ylabel(r'$\sigma^2(R)')
plt.legend()
plt.savefig('./Plots/sigmaR.png')
plt.close()

plt.plot(R, sigma0_R, label=r'$\sigma^2(R)$, no BH')
plt.plot(R, sigmaR_BH, label= r'$\sigma^2(R)$, 100% BH')
plt.plot(R, sigma_0, label=r'$\sigma^2(r)$, no BH')
plt.plot(R, sigma_BH, label=r'$\sigma^2(r)$, 100% BH')
plt.loglog()
plt.title(r'King-Model projected $\sigma^2$')
plt.xlabel(r'$R$')
plt.ylabel(r'$\sigma^2(R)')
plt.legend()
plt.savefig('./Plots/sigmaR_loglog.png')
plt.close()
'''

def fitfunc_king(R, rc, Mc, Mb):
    # cluster radius
    x = R / rc
    S2c = s2c(x, *s2c_params)
    S2b = s2b(x, *s2b_params)
    return (Mc * S2c + Mb * S2b)

def s2c(x, k, n, h, j, p, s):
    return x**n*k*(h+x**2)**j *( p  + x**2)**s

def s2b(x, k, n, h, j, p, s):
    return x**n*k*(h+x**2)**j *( p  + x**2)**s

def lnf_king(X, vx, vy, R, vxerr, vyerr, weight):
    '''
       a=X[0]
       Mc=X[1]
       Mb=X[2]'''

    x= R/X[0]
    sigma2=fitfunc_king(R, X[0], X[1], X[2])
    minfunc= -weight*( -vx**2/(2*sigma2+ 2*vxerr*2) - vy**2/(2*sigma2+ 2*vyerr**2) - 0.5*np.log(sigma2 + vxerr**2) - 0.5*np.log(sigma2 + vyerr**2) )
    return(np.sum(minfunc))

def lnf0_king(X, vx, vy, R, vxerr, vyerr, weight):
    '''
       a=X[0] 
       Mc= X[1]'''

    x= R/X[0]
    sigma2=fitfunc_king(R, X[0], X[1], 0)
    minfunc= -weight*( -vx**2/(2*sigma2+ 2*vxerr*2) - vy**2/(2*sigma2+ 2*vyerr**2) - 0.5*np.log(sigma2 + vxerr**2) - 0.5*np.log(sigma2 + vyerr**2) )
    return(np.sum(minfunc))

R_fit= R[R<=10]
sigmaR0_fit= sigma0_R[R<=10]
sigmaRBH_fit=sigmaR_BH[R<=10]

fit_s2c = curve_fit(s2c, R_fit, sigmaR0_fit, maxfev=40000, p0=(1/9, -1/3, 5/2, 1/5, 15/8, -8/15))
print('s2c fit: k, l:')
print(fit_s2c[0])
s2c_params= fit_s2c[0]
k = s2c_params[0]



fit_s2b = curve_fit(s2b, R_fit, sigmaRBH_fit, maxfev=40000, p0=(1/9, -1/3, 5/2, 1/5, 15/8, -8/15))
print('s2b fit:')
print(fit_s2b[0])
s2b_params= fit_s2b[0]

plt.plot(R, sigma0_R, label=r'King Model $M_b/M_c=0$', color='b')
plt.plot(R, s2c(R, *fit_s2c[0]), label= 's2c-fit', lw=2, ls='--', color='b')
plt.plot(R, sigmaR_BH, label='King Model $M_b/M_c=1$', color='r')
plt.plot(R, s2b(R, *fit_s2b[0]), label= 's2b-fit', lw=2, ls='--', color='r')
plt.axvline(x=10)
plt.loglog()
plt.legend()
plt.xlabel(r'$R  [1/a]$')
plt.ylabel(r'$\sigma^2(R)  [GM/a]$')
#plt.show()
plt.savefig('./Plots/King_fit_final.png')
plt.close()



XBH=np.array([30, 1000, 2])

result= optimize.minimize(lnf_king, XBH, args= (vx, vy, r, evx, evy, weight), bounds= [ (1.0, None), (0,None), (0,None)])
X0=([30, 1000])
res_0= optimize.minimize(lnf0_king, X0,  args= (vx, vy, r, evx, evy, weight), bounds= [ (1,None), (0,None)])
print('no BH Fit:')
print(res_0)
print('\nBH fit:')
print(result)
RES_BH=result.x
RES_0 =res_0.x




###  Find Bins with 10 stars close to center, 100 in middle and 1000 outside
r_sort= np.sort(r)                #sort radii  by increasing size
r_iter = len(r_sort)
borders= np.zeros(1)
k, l, m  =1, 1, 1

for i in range(r_iter):             # Find
    if ((i<100) & ((i/10.)== k)):
        borders= np.append(borders, r_sort[i])
        k +=1

    if ((i<1000) & ((i/100.)== l)):
        borders= np.append(borders, r_sort[i])
        l+= 1

    if ((i>= 1000) & ((i/1000.)== m)):
        borders= np.append(borders, r_sort[i])
        m+= 1
borders= np.append(borders, r_sort[len(r_sort)-1])

ind_N= np.digitize(r, borders)





logbins = np.r_[0, np.logspace(-1.5, 2.)]
Rbins= borders
iter=np.arange(0,len(Rbins)-1)
indR= np.digitize(r, Rbins)

R_bin_mid= Rbins[:-1] + 0.5 * (Rbins[1:] - Rbins[:-1])
Vmean=np.zeros(len(Rbins)-1)
Vdev= np.zeros(len(Rbins)-1)
vol= np.zeros(len(Rbins)-1)
Vmean_err= np.zeros(len(Rbins)-1)
Vdev_error= np.zeros(len(Rbins)-1)
histR, binR, patches= plt.hist(r, Rbins)

'''
def bootstrap_Vdev(N, vx, vy):
    ind = np.arange( 0, len(vx)-1)
    Vdev_boot = np.zeros(N)
    for i in range(N):
        sample = np.random.choice(ind, size=len(vx), replace=True)
        vx_boot= vx[sample]
        vy_boot= vy[sample]
        Vdev_boot[i]= np.sqrt((np.std(vx_boot)**2+np.std(vy_boot)**2)/2)
    V_boot_error = np.std(Vdev_boot)
    return V_boot_error

# find mean and std dev/ Qn of data bins depending on distance to core
#'''
for i in np.nditer(iter):
    Vmean[i]=np.mean(vx[np.where(indR==i+1)])
    Vmean_err[i] = np.mean(evx[np.where(indR==i+1)]**2+ evy[np.where(indR==i+1)]**2)
    Vdev[i]= (np.std(vx[np.where((indR==i+1) )])**2+np.std(vy[np.where((indR==i+1))])**2)/2
    #Vdev_error[i]= bootstrap_Vdev(1000, vx[np.where((indR==i+1) )], vy[np.where((indR==i+1))])
    vol[i]= (2.)*math.pi*(Rbins[i+1]**2.-Rbins[i]**2)
plt.close()


###plummer model
def fitfunc(R, rc, Mc, Mb):
    x = R / rc
    s2c = (3 * math.pi / 64) / np.sqrt(1 + x ** 2)
    s2b = x ** (-1.0 / 3.0) * 3 * 3.1415 / 64 * (2.5 + x ** 2) ** (0.2) * ((3 * 3.1415 / 64) ** (15.0 / 8.0) + x ** 2) ** (-8.0 / 15.0)
    return (Mc * s2c + Mb * s2b)

def lnf(X, vx, vy, R, vxerr, vyerr, weight):
    '''
       Mc=X[0]
       Mb=X[1]
       a= X[2]'''
    sigma2=fitfunc(R, X[2], X[0], X[1])
    minfunc= -( -vx**2/(2*sigma2+ 2*vxerr*2) - vy**2/(2*sigma2+ 2*vyerr**2) - 0.5*np.log(sigma2 + vxerr**2) - 0.5*np.log(sigma2 + vyerr**2) )
    return(np.sum(minfunc))

def lnf0(X, vx, vy, R, vxerr, vyerr, weight):
    '''
       Mc=X[0] 
       a= X[1]'''
    sigma2=fitfunc(R, X[1], X[0], 0)
    minfunc= -( -vx**2/(2*sigma2+ 2*vxerr*2) - vy**2/(2*sigma2+ 2*vyerr**2) - 0.5*np.log(sigma2 + vxerr**2) - 0.5*np.log(sigma2 + vyerr**2) )
    return(np.sum(minfunc))

XBH=np.array([1800, 0, 50])
result_plummer= optimize.minimize(lnf, XBH, args= (vx, vy, r, evx, evy, weight), bounds= [ (0.0, None), (0,None), (1,None)])
X0=([1600,50])
res_0_plummer= optimize.minimize(lnf0, X0,  args= (vx, vy, r, evx, evy, weight), bounds= [ (0,None), (1,None)])
RES_P=result_plummer.x
RES0_P=res_0_plummer.x


print('Plummer no BH')
print(res_0_plummer, '\n')
print('Plummer BH')
print(result_plummer)

def arcsec_to_m(a, Dm):  #D: distance to star, a arcsec
    a_deg= a/3600
    a_rad= a_deg*math.pi/180
    m= a_rad*Dm          #m: arcsec -> m
    return m

def M_to_M_solar(M_code, a, Dm, Mc):
    M_solar=2e30
    a_m= arcsec_to_m(a, Dm)
    G=6.67408e-11
    M_bh_solar= M_code*Mc*1000**2*a_m/(M_solar*G)
    return M_bh_solar

def plummer_M(M_code, a, Dm):
    M_solar=2e30
    a_m= arcsec_to_m(a, Dm)
    G=6.67408e-11
    return M_code*1000**2*a_m/(G*M_solar)

M_bh_king= M_to_M_solar(RES_BH[2], RES_BH[0], Dm, Mc)
Mc_bhmod_king= M_to_M_solar(RES_BH[1], RES_BH[0], Dm,Mc)
Mc_king= M_to_M_solar(RES_0[1], RES_0[0], Dm, Mc)
print('Mc: ', Mc)
print ('real Mbh King-Model:   ', M_bh_king)
print('real Mc King Model:  ', Mc_king)
print('BHmod Mc King Model: ', Mc_bhmod_king)
print('RATIO Mb/Mc=', M_bh_king/Mc_bhmod_king)

print('BH fit Plummer: Mc= ', int(plummer_M(RES_P[0], RES_P[2],Dm)), ' Mb= ', int(plummer_M(RES_P[1], RES_P[2], Dm)))
print('00 fit Plummer: Mc= ', int(plummer_M(RES0_P[0], RES0_P[1], Dm)))


plt.plot(R_bin_mid, np.sqrt(Vdev-Vmean_err) , label= 'Binned Data', color= 'k', marker= 'o')
plt.plot(R_bin_mid, np.sqrt(fitfunc_king(R_bin_mid, *RES_BH)), label='King: Black Hole Fit', color= 'r', linewidth= 2)
plt.plot(R_bin_mid, np.sqrt(fitfunc_king(R_bin_mid, *RES_0, 0)), label='King: No Black Hole Fit', color= 'b', linewidth=2)
plt.plot(R_bin_mid, np.sqrt(fitfunc(R_bin_mid, RES_P[2], RES_P[0], RES_P[1])), label='Plummer: Black Hole Fit', linewidth= 2, ls='--')
plt.plot(R_bin_mid, np.sqrt(fitfunc(R_bin_mid, RES0_P[1], RES0_P[0], 0)), label='Plummer: No Black Hole Fit', linewidth=2, ls='--')
plt.text(8,16.7, r'King: $M_b$={0}$M_s$, $M_c$={1} $M_s$'.format(int(M_bh_king), int(Mc_king)))
plt.text(8,16, r'King LnL: No BH:{0}, BH:{1}'.format(int(res_0.fun), int(result.fun)))
plt.legend()
plt.semilogx()
#plt.xlim(0.5,200)
#plt.ylim(220, 380)
plt.xlabel('R in arcsec')
plt.ylabel('Velocity Dispersion in km/s')
plt.title('King- and Plummer Model Fits for the velocity dispersion of the 47 Tuc')
plt.savefig('./Plots/king+plummer_fit.png')
plt.close()


np.savetxt('fitfunc_params_BH.txt', s2b_params)
np.savetxt('fitfunc_params_00.txt', s2c_params)

np.savetxt('king_params_BH.txt', RES_BH)
np.savetxt('king_params_00.txt', RES_0)


