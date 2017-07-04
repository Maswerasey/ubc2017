import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import Polygon
from matplotlib.patches import PathPatch
import pylab
import math as math
import qn_calc
from scipy.optimize import curve_fit
import scipy.optimize as optimize

pylab.rc('font', family='serif', size=14)               #makes pretty plots

#### import data
data = np.genfromtxt('pm_v6_ePMcut_comp_vel+pm.dat')

u, b, v = data[:, 0], data[:, 1], data[:, 2]            #magnitudes
x, y = data[:, 3]*0.04, data[:, 4]*0.04                 # turns pixels into arcsec
vx, vy = data[:, 5], data[:, 6]                         # v in km/s
mem = data[:, 19]                                       # = 1 if in Tuc, =2 if in SMC, =3 if field star
epmx, epmy = data[:,15], data[:,16]                     # propper motion error (pix/12yr)
weight= data[:,23]

#### convert data into usable units
Dist = 4.29*3.086e16                                    # kpc*(km/kpc) = km    #Holger's value
conv_t = 60*60*24*365.25                                # yr -> s
conv_ang = (180/math.pi)*60*60                          # radians -> arcsec
conv_vel = (1/conv_ang)*Dist/conv_t                     # arcsec/yr -> km/s

evx = epmx/12*conv_vel*0.04                             #convert errors
evy = epmy/12*conv_vel*0.04


#### isolating CMD Branches (MS, WD, Giants) by drawing boxes around them.

###  Main sequence
##   Find stars
points = np.c_[u-b, u]
#           upper left  lower left    lower right   upper right
verts = ([1.68, 20], [3.8, 25.5], [4.5, 25.5], [2.05, 20], [0,0])
codes = (1, 2, 2, 2, 79)
path = Path(verts, codes)
ms = path.contains_points(points)                                   # 1 if star in MS
ind = np.where(ms==1)
##filter main sequence stars
msX, msY, msU, msB, msV = x[ind], y[ind], u[ind], b[ind], v[ind]
msVX, msVY = vx[ind], vy[ind]
msMEM= mem[ind]
msEvx, msEvy = evx[ind], evy[ind]
msWeight= weight[ind]


#### Giant Branch
##   Find Giants
points = np.c_[u-b, u]
#           upper left  lower left    lower right   upper right
vertsL = ([2.15, 20], [3.2, 21.3], [3.2, 21.15], [2.25, 19.95], [0,0])
codesL = (1, 2, 2, 2, 79)
pathL = Path(vertsL, codesL)
GB = pathL.contains_points(points)
indL = np.where(GB==1)
##   Filter Giants
lmX, lmY, lmU, lmB, lmV = x[indL], y[indL], u[indL], b[indL], v[indL]
lmVX, lmVY = vx[indL], vy[indL]
lmMEM= mem[indL]
lmEvx, lmEvy= evx[indL], evy[indL]
lmweight= weight[indL]

### White Dwarf Branch
##  Find White Dwarfs
points = np.c_[u-b, u]
#           upper left  lower left    lower right   upper right
vertsWD = ([-0.95, 17.5], [-0.90, 20], [-0.45,24], [0.15 ,24], [-0.5, 21.6], [-0.95, 17.5], [0,0])
codesWD = (1, 2, 2, 2, 2, 2, 79)
pathWD = Path(vertsWD, codesWD)
wd = pathWD.contains_points(points)
indWD = np.where(wd==1)
##  Filter white dwarfs
wdX, wdY, wdU, wdB, wdV = x[indWD], y[indWD], u[indWD], b[indWD], v[indWD]
wdVX, wdVY = vx[indWD], vy[indWD]
wdMEM = mem[indWD]
wdEvx, wdEvy = evx[indWD], evy[indWD]
wdweight=weight[indWD]

#### Plots of CMD and Velocities
'''
# CMD incl SMC
#plt.scatter(u-b, u, marker=".", s=1)
#plt.show()
#plt.close()

#plot of all velocities relative to center including SMC
#plt.scatter(vx, vy, marker=".", s=1)
#plt.show()
#plt.close()

#plot of all velocities incl SMC
#plt.scatter( np.sqrt(x**2+y**2), np.sqrt(vx**2+vy**2), marker=".", s=1)
#plt.loglog()
#plt.show()

#plot velocities relative to centre (only TUC)
plt.scatter(vx[mem==1], vy[mem==1], marker=".", s=1)
plt.xlabel('vx (km/s)')
plt.ylabel('vy (km/s)')
plt.title(' cluster star velocities relative to centre of 47 Tuc.')
#plt.show()
plt.savefig('../Plots/Tuc_vx_vy.png')
plt.close()

#plot of velocities vs radius no high vel. stars
plt.scatter( np.sqrt(x**2+y**2)[mem==1], np.sqrt(vx**2+vy**2)[mem==1] , marker=".", s=1)
plt.xlabel('r (arcsec)')
plt.ylabel('v (km/s)')
plt.title('velocities vs distance to centre for cluster stars')
plt.loglog()
#plt.show()
plt.savefig('../Plots/Tuc_centre_vel.png')
plt.close()

#plot of CMD with boxes
axes=plt.gca()
poly=PathPatch(path, alpha=0.4)
axes.add_patch(poly)
polyL=PathPatch(pathL, alpha=0.4)
axes.add_patch(polyL)
polyWD=PathPatch(pathWD, alpha=0.4)
axes.add_patch(polyWD)
plt.ylim(15,27)
plt.xlabel('Colour (U_B)')
plt.ylabel('U magnitude')
plt.title('CMD of 47 Tuc.')
plt.scatter((u-b)[mem==1], u[mem==1], marker=".", s=1)
plt.gca().invert_yaxis()
#plt.show()
plt.savefig('../Plots/TUC_CMD_Boxes.png')
plt.close()

#plot velocity of MS and WD Stars
pylab.scatter(np.sqrt(msX**2+msY**2), np.sqrt(msVX**2+msVY**2))
pylab.scatter(np.sqrt(wdX**2+wdY**2), np.sqrt(wdVX**2+wdVY**2))
plt.xlabel('r (arcsec)')
plt.ylabel('v (km/s)')
plt.title('velocity vs distance to centre for stars in the MS and WD bands of the CMD.')
plt.loglog()
#plt.show()
plt.savefig('../Plots/v_r_MS_and_WD.png')
plt.close()
''' #expand here (curently inactive)


#### find fast stars
msR, msV = np.sqrt(msX**2+msY**2), np.sqrt(msVX**2+msVY**2)
wdR, wdV = np.sqrt(wdX**2+wdY**2), np.sqrt(wdVX**2+wdVY**2)
pointsV = np.c_[msR, msV]
pointsVwd= np.c_[wdR,wdV]
#           upper left  lower left    lower right   upper right
vertsV = ([2, 50], [1000, 50], [1000,1000], [2, 1000], [0,0])
codesV = (1, 2, 2, 2, 79)
pathV = Path(vertsV, codesV)
fast = pathV.contains_points(pointsV)
fastwd=pathV.contains_points(pointsVwd)
indV = np.where(fast==1)
indVwd=np.where(fastwd==1)

###filter  fast stars
vX, vY, vU, vB, vV = msX[indV], msY[indV], msU[indV], msB[indV], msV[indV]
vVX, vVY = msVX[indV], msVY[indV]
vMEM= msMEM[indV]
vEvx, vEvy= evx[indV], evy[indV]
vWeight= weight[indV]

wdvX, wdvY, wdvU, wdvB, wdvV = wdX[indVwd], wdY[indVwd], wdU[indVwd], wdB[indVwd], wdV[indVwd]
wdvVX, wdvVY = wdVX[indVwd], wdVY[indVwd]
wdvMEM= wdMEM[indVwd]
wdvEvx, wdvEvy = wdEvx[indVwd], wdEvy[indVwd]
wdvWeight= wdweight[indVwd]

vX= np.append(vX,wdvX)                                  #put all fast stars into one array
vY= np.append(vY,wdvY)
vU= np.append(vU,wdvU)
vB= np.append(vB,wdvB)
vV= np.append(vV,wdvV)
vVX= np.append(vVX,wdvVX)
vVY= np.append(vVY,wdvVY)
vEvx= np.append(vEvx, wdvEvx)
vEvy= np.append(vEvy, wdvEvy)
vWeight= np.append(vWeight, wdvWeight)
#### Plots
'''
## CMD incl SMC and fast
plt.scatter(u-b, u, marker=".", s=1)
plt.scatter(vU-vB, vU, c='r')
plt.xlabel('Colour (U_B)')
plt.ylabel('U magnitude')
plt.title('CMD of 47 Tuc including field stars and SMC')
plt.gca().invert_yaxis()
#plt.show()
plt.savefig('../Plots/CMD_all.png')
plt.close()

## plot of all velocities relative to center including SMC and fast
plt.scatter(vx, vy, marker=".", s=1)
plt.scatter(vVX, vVY, marker="o", c='r')
plt.xlabel('r (arcsec)')
plt.ylabel('v (km/s)')
plt.title('velocities vs distance to centre incl. SMC and field stars')
#plt.show()
plt.savefig('../Plots/vx_vy_incl_fast_stars.png')
plt.close()

## plot velocity of MS and WD Stars
pylab.scatter(np.sqrt(msX**2+msY**2), np.sqrt(msVX**2+msVY**2), marker=".", s=1)
pylab.scatter(np.sqrt(wdX**2+wdY**2), np.sqrt(wdVX**2+wdVY**2), marker=".", s=1)
plt.loglog()
plt.scatter(np.sqrt(vX**2+vY**2), np.sqrt(vVX**2+vVY**2), marker='o', c='r')
polyV=PathPatch(pathV, alpha=0.0)
axes2=plt.gca()
axes2.add_patch(polyV)
plt.xlabel('r (arcsec)')
plt.ylabel('v (km/s)')
plt.title('velocities vs distance for MS and WD Bands ')
#plt.show()
plt.savefig('../Plots/v_r_incl_fast.png')
plt.close()

## plot of all velocities incl SMC and fast
plt.scatter( np.sqrt(x**2+y**2), np.sqrt(vx**2+vy**2), marker=".", s=1)
plt.scatter(np.sqrt(vX**2+vY**2), np.sqrt(vVX**2+vVY**2), marker='o', c='r')
polyV=PathPatch(pathV, alpha=0.0)
axes2=plt.gca()
axes2.add_patch(polyV)
plt.xlabel('r (arcsec)')
plt.ylabel('v (km/s)')
plt.title('velocity vs distance for all stars in CMD')
plt.loglog()
#plt.show()
plt.savefig('../Plots/v_r_incl_SMC_and_fast.png')
plt.close()

''' # expand here (curently inactive)


#### find velocity dispersion of sample

###combine all TUC data (MS, WD, Giants) into one array per component
Tx= np.r_[msX, wdX, lmX]
Ty= np.r_[msY, wdY, lmY]
TU= np.r_[msU, wdU, lmU]
TB= np.r_[msB, wdB, lmB]
TV= np.r_[msV, wdV, lmV]
Tvx= np.r_[msVX, wdVX, lmVX]
Tvy= np.r_[msVY, wdVY, lmVY]
T_R= np.sqrt(Tx**2+Ty**2)           #radial distance (Projection)
T_V= np.sqrt(Tvx**2+Tvy**2)         # absolute velocity
T_evx = np.r_[msEvx, wdEvx, lmEvx]  # velocity errors
T_evy = np.r_[msEvy, wdEvy, lmEvy]
T_weight= np.r_[msWeight, lmweight, wdweight]

Vlimit=50                           # max speed to be considered
Tvx_clean = Tvx[T_V<=Vlimit]        # cleaned from very fast false stars)
Tvy_clean = Tvy[T_V<=Vlimit]
T_R_clean = T_R[T_V<=Vlimit]
T_evx_clean = T_evx[T_V<=Vlimit]
T_evy_clean = T_evy[T_V<=Vlimit]
T_weight_clean= T_weight[T_V<=Vlimit]
T_V_clean= T_V[T_V<=Vlimit]

###  Find Bins with 10 stars close to center, 100 in middle and 1000 outside
r_sort= np.sort(T_R)                #sort radii  by increasing size
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

ind_N= np.digitize(T_R_clean, borders)





logbins = np.r_[0, np.logspace(-1.5, 2.)]
#print(logbins)
Rbins= borders

#Rbins=np.percentile(T_R_clean, logbins)
#Rbins=np.logspace(-1,2,40)
iter=np.arange(0,len(Rbins)-1)
indR= np.digitize(T_R_clean, Rbins)

R_bin_mid= Rbins[:-1] + 0.5 * (Rbins[1:] - Rbins[:-1])
Vmean=np.zeros(len(Rbins)-1)
Vdev= np.zeros(len(Rbins)-1)
vol= np.zeros(len(Rbins)-1)
Vmean_err= np.zeros(len(Rbins)-1)
Vdev_error= np.zeros(len(Rbins)-1)
histR, binR, patches= plt.hist(T_R_clean, Rbins)

#plt.show()
plt.close()


#print(Rbins)

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
#f, axarr = plt.subplots(len(Rbins)-1, sharex=True)

for i in np.nditer(iter):
    Vmean[i]=np.mean(Tvx_clean[np.where(indR==i+1)])
    Vmean_err[i] = np.mean(T_evx_clean[np.where(indR==i+1)]**2+ T_evy_clean[np.where(indR==i+1)]**2)
    #Vdev[i] =qn_calc.qn_calc(Tvx_clean[np.where(indR==i+1)])**2+qn_calc.qn_calc(Tvy_clean[np.where(indR==i+1)])**2
    Vdev[i]= (np.std(Tvx_clean[np.where((indR==i+1) )])**2+np.std(Tvy_clean[np.where((indR==i+1))])**2)/2
    Vdev_error[i]= bootstrap_Vdev(1000, Tvx_clean[np.where((indR==i+1) )], Tvy_clean[np.where((indR==i+1))])
   # axarr[i].hist(Tvx_clean[np.where(indR==i+1)], bins=40)
    vol[i]= (2.)*math.pi*(Rbins[i+1]**2.-Rbins[i]**2)
plt.xlim(0,50)
#plt.show()
plt.close()

plt.plot(R_bin_mid, Vdev**2-Vmean_err**2)
#plt.plot(R_bin_mid, Vmean)
plt.title('Qn^2 of v versus distance to cluster centre.')
plt.xlabel('r in arcsec')
plt.ylabel('Qn^2')
#plt.show()
plt.savefig('../Plots/Qn2.png')
plt.close()


# find density rho(r)
rho= histR/vol
def density(R, a, M):
    return(M*a**2/math.pi * 1/(R**2+a**2)**2)

popt_rho, pcov_rho= curve_fit(density, R_bin_mid, rho)
a, M= abs(popt_rho[0]), abs(popt_rho[1])
print('density \n         a        M\n', popt_rho)

plt.scatter(R_bin_mid, rho)
plt.plot(R_bin_mid, density(R_bin_mid, *popt_rho))
plt.semilogx()
plt.xlim(5,130)
plt.xlabel('r in arcsec')
plt.ylabel('density rho')
plt.title('Star density vs distance to cluster centre.')
plt.savefig('../Plots/density.png')
#plt.show()
plt.close()


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


popt_data, pcov_data= curve_fit(fitfunc, R_bin_mid, Vdev)
popt_0, pcov_o=curve_fit(fitfunc0, R_bin_mid, Vdev)
plt.scatter(R_bin_mid, Vdev)
plt.plot(R_bin_mid, fitfunc(R_bin_mid, *popt_data), label='Black Hole')
#plt.plot(R_bin_mid, fitfunc(R_bin_mid, (popt_data[0]+ pcov_data[0][0]), (popt_data[1]+ pcov_data[1][1]), (popt_data[2]+ pcov_data[2][2]) ), label='Black Hole max')
#plt.plot(R_bin_mid, fitfunc(R_bin_mid, popt_data[0]- pcov_data[0][0], popt_data[1]- pcov_data[1][1], popt_data[2]- pcov_data[2][2] ), label='Black Hole min')
plt.plot(R_bin_mid, fitfunc(R_bin_mid, (popt_data[0]), (popt_data[1]), (popt_data[2]+ pcov_data[2][2]) ), label='Black Hole  BH max')
plt.plot(R_bin_mid, fitfunc(R_bin_mid, popt_data[0], popt_data[1], popt_data[2]- pcov_data[2][2] ), label='Black Hole BH min')
plt.plot(R_bin_mid, fitfunc0(R_bin_mid, *popt_0), label='no Black Hole')
plt.legend()
plt.semilogx()
#plt.xlim(7,180)
#plt.ylim(180, 300)
plt.xlabel('R in arcsec')
plt.ylabel('Qn^2')
plt.title('Velocity dispersion^2 vs projected radial distance to centre.')
#plt.show()
plt.savefig('../Plots/Qn_fit_min_max_BH.png')
plt.close()
print('       rc                 Mc                  Mb \n',
      popt_data, '\n')

print('       Mc                 rc \n',
      popt_0, '\n')

print('BH quotient Mb/Mc= ', popt_data[2]/popt_data[1])

#X=np.array(a, Mc, Mb)

def lnf(X, vx, vy, R, vxerr, vyerr, weight):
    '''
       Mc=X[0]
       Mb=X[1]
       a= X[2]'''

    x= R/X[2]
    #print(Mc)
    sigma2=fitfunc(R, X[2], X[0], X[1])
    #s2c = (3 * math.pi / 64) / np.sqrt(1 + x ** 2)
    #s2b = x**(-1.0 / 3.0) *3*3.1415/64*(2.5 + x**2)**(0.2)*((3*3.1415/64)**(15.0/8.0)+x**2)**(-8.0/15.0)
    #sigma2= X[0]*s2c+ X[1]*s2b
    #print('\n sigma:', sigma2)
    minfunc= -weight*( -vx**2/(2*sigma2+ 2*vxerr*2) - vy**2/(2*sigma2+ 2*vyerr**2) - 0.5*np.log(sigma2 + vxerr**2) - 0.5*np.log(sigma2 + vyerr**2) )
    #minfunc= -(-(vx**2+vy**2)/(2*sigma2) - 0.5*np.log(sigma2))
    return(np.sum(minfunc))

def lnf0(X, vx, vy, R, vxerr, vyerr, weight):
    '''
       Mc=X[0] 
       a= X[1]'''

    x= R/X[1]
    #print(Mc)
    sigma2=fitfunc(R, X[1], X[0], 0)
    #s2c = (3 * math.pi / 64) / np.sqrt(1 + x ** 2)
    #s2b = x**(-1.0 / 3.0) *3*3.1415/64*(2.5 + x**2)**(0.2)*((3*3.1415/64)**(15.0/8.0)+x**2)**(-8.0/15.0)
    #sigma2= X[0]*s2c+ X[1]*s2b
    #print('\n sigma:', sigma2)
    minfunc= -weight*( -vx**2/(2*sigma2+ 2*vxerr*2) - vy**2/(2*sigma2+ 2*vyerr**2) - 0.5*np.log(sigma2 + vxerr**2) - 0.5*np.log(sigma2 + vyerr**2) )
    #minfunc= -(-(vx**2+vy**2)/(2*sigma2) - 0.5*np.log(sigma2))
    return(np.sum(minfunc))





XBH=np.array([1800, 0, 50])
sigma2=8
Mc=160

result= optimize.minimize(lnf, XBH, args= (Tvx_clean, Tvy_clean, T_R_clean, T_evx_clean, T_evy_clean, T_weight_clean), bounds= [ (0.0, None), (0,None), (1,None)])
X0=([1600,50])
res_0= optimize.minimize(lnf0, X0,  args= (Tvx_clean, Tvy_clean, T_R_clean, T_evx_clean, T_evy_clean, T_weight_clean), bounds= [ (0,None), (1,None)])
print(res_0)


print('\n')
print(result)
RES= result.x
RES0= res_0.x
plt.errorbar(R_bin_mid, np.sqrt(Vdev-Vmean_err), yerr= Vdev_error, label= 'Binned Data', color= 'k', marker= 'o', linewidth= 1)
plt.plot(R_bin_mid, np.sqrt(fitfunc(R_bin_mid, RES[2], RES[0], RES[1])), label='Black Hole Fit', color= 'r', linewidth= 2)
plt.plot(R_bin_mid, np.sqrt(fitfunc(R_bin_mid, RES0[1], RES0[0], 0)), label=' No Black Hole Fit', color= 'b', linewidth=2)
plt.legend()
plt.semilogx()
plt.xlim(0.5,200)
#plt.ylim(220, 380)
plt.xlabel('R in arcsec')
plt.ylabel('Velocity Dispersion in km/s')
plt.title('Maximal Likelihood Fit of the Velocity Dispersion.')
plt.savefig('../Plots/fit_sigma2_incl_errors.png')
#plt.show()
plt.close()

print( 'Mb/Mc= ', RES[1]/RES[0])

print(pcov_data)

np.savetxt('Tuc_fitBH.txt', np.transpose(RES), header='# Mc, Mb, a')
np.savetxt('Tuc_fit0.txt', np.transpose([res_0.x[0], 0, res_0.x[1]]), header='# Mc, Mb, a')

def bootstrap(N, vx, vy, r, evx, evy, weight):
    Mc, Mb, a = np.zeros(N), np.zeros(N), np.zeros(N)
    ind = np.arange(0, len(vx)-1)
    XBH = np.array([1800, 0, 50])
    for i in range(N):
        sample= np.random.choice(ind,size= len(vx),  replace=True)
        vx_sample = vx[sample]
        vy_sample = vy[sample]
        r_sample  = r[sample]
        evx_sample= evx[sample]
        evy_sample= evy[sample]
        weight_sample= weight[sample]

        result = optimize.minimize(lnf, XBH, args=(vx_sample, vy_sample, r_sample, evx_sample, evy_sample, weight_sample),
                                   bounds=[(0.0, None), (0, None), (1, None)])
        Mc[i]= result.x[0]
        Mb[i]= result.x[1]
        a[i] = result.x[2]
        XBH= result.x
        #print('Bootstrap ', i, ' out of ', N, 'Iterations.')
    return Mc, Mb, a

Mc_boot, Mb_boot, a_boot = bootstrap (10, Tvx_clean, Tvy_clean, T_R_clean, T_evx_clean, T_evy_clean, T_weight_clean)


Mc_hist, Mc_bins, Mc_patches = plt.hist(Mc_boot, bins= 10)
plt.title('Bootstrap Mc histogram')
#plt.show()
plt.close()

Mb_hist, Mb_bins, Mb_patches = plt.hist(Mb_boot, bins= 10)
plt.title('Bootstrap Mb histogram')
#plt.show()
plt.close()

a_hist, a_bins, a_patches = plt.hist(a_boot, bins= 10)
plt.title('Bootstrap a histogram')
#plt.show()
plt.close()

np.savetxt('Tuc_Data.txt', np.transpose([Tx, Ty, TU, TB, TV, Tvx, Tvy, T_R, T_V, T_evx, T_evy , T_weight]) , header='# Tuc Data \n# (0)x,   (1)y,    (2)mU,   (3)mB,    (4)mV,    (5)vx,    (6)vy,   (7)r,   (8)v,   (9)evx,    (10)evy,   (11)weight')

