import numpy as np
import matplotlib.pyplot as plt
import math as math
import scipy.optimize as optimize

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

plt.scatter(r00_mod, v00)
plt.title('0% BH N-Body Simulation')
plt.xlabel('r [arcsec]')
plt.ylabel('v [km/s]')
plt.xlim(0.01,1100)
plt.ylim(1,200)
plt.loglog()
plt.savefig('00BH.png')
#plt.show()
plt.close()


plt.scatter(r05_mod, v05)
plt.title('0.5% BH N_Body Simulation')
plt.xlabel('r [arcsec]')
plt.ylabel('v [km/s]')
plt.xlim(0.01,1100)
plt.ylim(1,200)
plt.loglog()
plt.savefig('05BH.png')
#plt.show()
plt.close()

plt.scatter(r10_mod, v10)
plt.title('1% BH N_Body Simulation')
plt.xlabel('r [arcsec]')
plt.ylabel('v [km/s]')
plt.xlim(0.01,1100)
plt.ylim(1,200)
plt.loglog()
plt.savefig('10BH.png')
#plt.show()
plt.close()


plt.scatter(r00_mod, v00,marker='x', c='b', label='0%')

plt.scatter(r05_mod, v05,marker='^', c='y', label='0.5%')

plt.scatter(r10_mod, v10, c='r', label='1%')
plt.xlabel('r [arcsec]')
plt.ylabel('v [km/s]')
plt.xlim(0.01,1100)
plt.ylim(1,200)
plt.title('N_Body Simulation velocities')
plt.legend()
plt.loglog()
plt.savefig('allBH.png')
#plt.show()
plt.close()
#'''


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

def lnf(X, v, R):
    '''
       Mc=X[0]
       Mb=X[1]
       a= X[2]'''

    x= R/X[2]
    #print(Mc)
    sigma2=fitfunc(R, X[2], X[0], X[1])
    minfunc= -(-(v**2)/(2*sigma2) - 0.5*np.log(sigma2))
    return(np.sum(minfunc))

def lnf0(X, v, R):
    '''
       Mc=X[0]
       a= X[1]'''

    x= R/X[1]
    sigma2=fitfunc(R, X[1], X[0], 0)
    minfunc= -(-(v**2)/(2*sigma2) - 0.5*np.log(sigma2))
    return(np.sum(minfunc))

X0=([100,30])
XBH= ([1500,1,30])

resBH_00= optimize.minimize(lnf, XBH, args= (v00, r00_mod), bounds= [ (0.0, None), (0, None), (1, None)])
#res0_00= optimize.minimize(lnf0, X0,  args= (v00, r00_mod), bounds= [ (0,None), (1,None)])
print('\n00 BH:')
print(resBH_00)
#print(res0_00)

resBH_05= optimize.minimize(lnf, XBH, args= (v05, r05_mod), bounds= [ (0.0, None), (0, None), (1, None)])
#res0_05= optimize.minimize(lnf0, X0,  args= (v05, r05_mod), bounds= [ (0,None), (1,None)])
print('\n05 BH:')
print(resBH_05)
#print(res0_05)

resBH_10= optimize.minimize(lnf, XBH, args= (v10, r10_mod), bounds= [ (0.0, None), (0, None), (1, None)])
#res0_10= optimize.minimize(lnf0, X0,  args= (v10, r10_mod), bounds= [ (0,None), (1,None)])
print('\n10 BH:')
print(resBH_10)
#print(res0_10)

print( '0.0% BH: Mb/Mc= ', resBH_00.x[1]/resBH_00.x[0])
print( '0.5% BH: Mb/Mc= ', resBH_05.x[1]/resBH_05.x[0])
print( '1.0% BH: Mb/Mc= ', resBH_10.x[1]/resBH_10.x[0])

#plt.scatter(R_bin_mid, Vdev)
plt.plot(np.linspace(-1, 3, 50), fitfunc(np.linspace(-1, 3, 50), resBH_00.x[2], resBH_00.x[0], resBH_00.x[1]), label='00 Black Hole')
plt.plot(np.linspace(-1, 3, 50), fitfunc(np.linspace(-1, 3, 50), resBH_05.x[2], resBH_05.x[0], resBH_05.x[1]), label='05 Black Hole')
plt.plot(np.linspace(-1, 3, 50), fitfunc(np.linspace(-1, 3, 50), resBH_10.x[2], resBH_10.x[0], resBH_10.x[1]), label='10 Black Hole')
#plt.plot(R_bin_mid, fitfunc(R_bin_mid, RES0[1], RES0[0], 0), label=' no Black Hole')
plt.legend()
plt.semilogx()
#plt.xlim(7,180)
#plt.ylim(180, 310)
plt.xlabel('$R$ [arcsec]')
plt.ylabel(r'$\sigma^2$ [km/s]')
plt.title('Velocity dispersion^2 vs projected radial distance to centre.')
plt.savefig('fit_model.png')
#plt.show()
plt.close()

