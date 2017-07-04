import numpy as np

import math as math

import scipy.optimize as optimize

from os import sys

# pylab.rc('font', family='serif', size=14)               #makes pretty plots

N = int(sys.argv[1])



#### Import Data
#### import data
data= np.genfromtxt('Tuc_Data.txt')
s2b_params= np.genfromtxt('fitfunc_params_BH.txt')
s2c_params= np.genfromtxt('fitfunc_params_00.txt')
model_params_BH= np.genfromtxt('king_params_BH.txt')
model_params_00= np.genfromtxt('king_params_00.txt')


a0, Mc0, Mb0 = model_params_00[0], model_params_00[1], 0
aBH, McBH, MbBH= model_params_BH[0], model_params_BH[1], model_params_BH[2]





Vlim=50

D= Dist = 4.29*3.086e16 # (kpc->km)
Dm= D*1000

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

# fit velocity dispersion to data
def fitfunc0(R, Mc, rc):
                  #cluster radius
    x=R/rc
    s2c= (3*math.pi/64)/np.sqrt(1+x**2)
    return (Mc*s2c)

def fitfunc(R, rc, Mc,Mb):
                      #cluster radius
    x=R/rc
    s2c = (3 * math.pi / 64) / np.sqrt(1 + x ** 2)
    s2b= x**(-1.0/3.0)*3*3.1415/64*(2.5+x**2)**(0.2)*((3*3.1415/64)**(15.0/8.0)+x**2)**(-8.0/15.0)
    return (Mc*s2c+Mb*s2b)



def lnf(X, vx, vy, R, vxerr, vyerr):
    '''
       Mc=X[0]
       Mb=X[1]
       a= X[2]'''
    sigma2=fitfunc(R, X[2], X[0], X[1])
    minfunc= -( -vx**2/(2*sigma2+ 2*vxerr*2) - vy**2/(2*sigma2+ 2*vyerr**2) - 0.5*np.log(sigma2 + vxerr**2) - 0.5*np.log(sigma2 + vyerr**2) )
    return(np.sum(minfunc))

def lnf0(X, vx, vy, R, vxerr, vyerr):
    '''
       Mc=X[0] 
       a= X[1]'''
    sigma2=fitfunc(R, X[1], X[0], 0)
    minfunc= -( -vx**2/(2*sigma2+ 2*vxerr*2) - vy**2/(2*sigma2+ 2*vyerr**2) - 0.5*np.log(sigma2 + vxerr**2) - 0.5*np.log(sigma2 + vyerr**2) )
    return(np.sum(minfunc))

##### curve fit functions and max. likelihood function

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




#### Find random velocities from distribution

def random_v( r, a, Mc, Mb):
    random_x = np.random.normal(size=len(r))
    random_y = np.random.normal(size=len(r))
    sigma2 = fitfunc_king(r, a, Mc, Mb)
    sigma =  np.sqrt(sigma2)
    vx = sigma * random_x
    vy = sigma * random_y
    return vx, vy

a_00_00= np.zeros(N)
a_00_BH= np.zeros(N)

a_BH_00= np.zeros(N)
a_BH_BH= np.zeros(N)

Mc_00_00= np.zeros(N)
Mc_00_BH= np.zeros(N)

Mc_BH_00= np.zeros(N)
Mc_BH_BH= np.zeros(N)

Mb_00_00= np.zeros(N)
Mb_00_BH= np.zeros(N)

Mb_BH_00= np.zeros(N)
Mb_BH_BH= np.zeros(N)

lnL_00_00= np.zeros(N)
lnL_00_BH= np.zeros(N)
lnL_BH_00= np.zeros(N)
lnL_BH_BH= np.zeros(N)

suc_00_00 = np.zeros(N)
suc_00_BH = np.zeros(N)
suc_BH_00 = np.zeros(N)
suc_BH_BH = np.zeros(N)


for i in range(N):
    vx_00, vy_00 = random_v( r, a0, Mc0, Mb0)
    vx_BH, vy_BH = random_v( r, aBH, McBH, MbBH)

    XBH=np.array([30, 1000, 0])
    X0=([30, 1000])

    #result_distribution_fit
    result_BH_BH= optimize.minimize(lnf_king, XBH, args= (vx_BH, vy_BH, r, evx, evy, 1), bounds= [ (1, None), (0,None), (0,None)])
    result_BH_00= optimize.minimize(lnf0_king, X0,  args= (vx_BH, vy_BH, r, evx, evy, 1), bounds= [ (1,None), (0,None)])
    res_BH_BH, res_BH_00 = result_BH_BH.x, result_BH_00.x
    a_BH_BH[i], Mc_BH_BH[i], Mb_BH_BH[i], lnL_BH_BH[i], suc_BH_BH[i]= res_BH_BH[0], res_BH_BH[1], res_BH_BH[2], result_BH_BH.fun, result_BH_BH.success
    a_BH_00[i], Mc_BH_00[i], Mb_BH_00[i], lnL_BH_00[i], suc_BH_00[i]= res_BH_00[0], res_BH_00[1], 0, result_BH_00.fun, result_BH_00.success

    result_00_BH= optimize.minimize(lnf_king, XBH, args= (vx_00, vy_00, r, evx, evy, 1), bounds= [ (1, None), (0,None), (0,None)])
    result_00_00= optimize.minimize(lnf0_king, X0,  args= (vx_00, vy_00, r, evx, evy, 1), bounds= [ (1,None), (0,None)])
    res_00_BH, res_00_00 = result_00_BH.x, result_00_00.x
    a_00_BH[i], Mc_00_BH[i], Mb_00_BH[i], lnL_00_BH[i], suc_00_BH[i]= res_00_BH[0], res_00_BH[1], res_00_BH[2], result_00_BH.fun, result_00_BH.success
    a_00_00[i], Mc_00_00[i], Mb_00_00[i], lnL_00_00[i], suc_00_00[i]= res_00_00[0], res_00_00[1], 0, result_00_00.fun, result_00_00.success



np.savetxt(sys.stdout, np.transpose([a_00_00, Mc_00_00, Mb_00_00, a_00_BH, Mc_00_BH, Mb_00_BH, a_BH_00, Mc_BH_00, Mb_BH_00, a_BH_BH, Mc_BH_BH, Mb_BH_BH, lnL_00_00, suc_00_00, lnL_00_BH, suc_00_BH, lnL_BH_00, suc_BH_00, lnL_BH_BH, suc_BH_BH]), header='(0)a_00_00,    (1)Mc_00_00,    (2)Mb_00_00,    (3)a_00_BH,    (4)Mc_00_BH,    (5)Mb_00_BH,    (6)a_BH_00,    (7)Mc_BH_00,    (8)Mb_BH_00,    (9)a_BH_BH,    (10)Mc_BH_BH,    (11)Mb_BH_BH,    (12)lnN_00_00,    (13)suc_00_00,    (14)lnN_00_BH,    (15)suc_00_BH,    (16)lnN_BH_00,    (17)suc_BH_00,    (18)lnL_BH_BH,    (19)suc_BH_BH')





