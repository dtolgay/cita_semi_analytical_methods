import os
import h5py as h5
import numpy as np
import scipy as sp
import glob

# -*- coding: utf-8 -*- 

M_sun      = 1.99e33   # in gms 
kpc        = 3.08e21   # in cms 
pc         = kpc/1.e3
k_b        = 1.380e-16 # in erg K^(-1)
m_proton   = 1.672e-24 # in gms 
gamma      = 5./3      # adiabatic index 
omega_m    = 0.272
omega_b    = 0.0455
omega_c    = omega_m - omega_b
sigma_dust = 5.e-22    # cm^(2) g^(-1)]] 
h          = 0.7       # Hubble_parameter
solar_metallicity = 0.02 #solar mass fraction

#############DEFINING KERNELS#########################

def cubic_spline(q, smoothing_length):
        q = abs(q/smoothing_length)
        factor = 8./(np.pi*smoothing_length**3)
        #print "We are in the cubic spline" 
        if 0 <= q < 1./2.:
                return factor*(1. + 6.*(q - 1.)*q**2)
        if 1./2 <= q < 1.:
                return factor*(2.*(1. - q)**3)
        if q >= 1.:
                return 0.

def quintic_spline(q, smoothing_length):
        q = abs(q/smoothing_length)
        factor = 3.**7/(40*np.pi*smoothing_length**3)
        #print "We are in the quintic spline"
        if 0 <= q < 1./3:
                return factor * ((1. - q)**5 - 6.*(2./3 - q)**5 + 15.*(1./3 - q)**5)
        if 1./3 <= q < 2./3:
                return factor * ((1. - q)**5 - 6.*(2./3 - q)**5)
        if 2./3 <= q < 1.:
                return factor * (1. - q)**5
        if q >= 1.:
                return 0.

#######################################################

def normalization(dr, smoothing_length):
        #Equation-7
        sum_W = 0.

        for i in range(0, len(dr)):
                sum_W += quintic_spline(dr[i], smoothing_length[i])

        return sum_W

def phi(dr, smoothing_length, norm):
        #Equation-6
        return quintic_spline(dr, smoothing_length)/norm


def construct_E(dxx, dyy, dzz, phi_i):
        #Equation-13, 14  
        #This constructs the matrix E for particle i and then returns the inverse (= B) from the paper 

        E = np.zeros(shape=(3,3))

        #print dxx, dyy, dzz, phi_i, E 

        E[0][0] = dxx*dxx*phi_i
        E[0][1] = dxx*dyy*phi_i
        E[0][2] = dxx*dyy*phi_i
        E[1][1] = dyy*dyy*phi_i
        E[2][2] = dzz*dzz*phi_i
        E[1][2] = dyy*dzz*phi_i

        E[1][0] = E[0][1]
        E[2][0] = E[0][2]
        E[2][1] = E[1][2]

        return E




def construct_gradient(dparameter, dxx, dyy, dzz, dr, smoothing_length):
        # Equation 12 
        # This constructs the gradient for dparameter
        norm_i = normalization(dr, smoothing_length)

        #print "DEBUG: NORM I", norm_i  
        E = np.zeros(shape=(3,3))

        for j in range(len(dr)):

                phi_j = phi(dr[j], smoothing_length[j], norm_i)
                #print construct_E(dxx[j], dyy[j], dzz[j], phi_j)
                E += construct_E(dxx[j], dyy[j], dzz[j],  phi_j)

        B = np.linalg.inv(E)

        gradient = np.zeros(3)
        for j in range(0, len(dr)):
                phi_j = phi(dr[j], smoothing_length[j], norm_i)

                gradient[0] += dparameter[j]*phi_j*( B[0][0]*dxx[j] + B[0][1]*dyy[j] + B[0][2]*dzz[j] )
                gradient[1] += dparameter[j]*phi_j*( B[1][0]*dxx[j] + B[1][1]*dyy[j] + B[1][2]*dzz[j] )
                gradient[2] += dparameter[j]*phi_j*( B[2][0]*dxx[j] + B[2][1]*dyy[j] + B[2][2]*dzz[j] )

        return gradient

####################################################


def change_coordinates(xc, yc, zc, x, y, z):
    xnew = x-xc
    ynew = y-yc
    znew = z-zc
    return xnew, ynew, znew

def cut_particles_index(x,y,z,R_max,cut):
	# spherical or cylindrical cut
	if cut==0: #spherical
		R_part = x**2+y**2+z**2
		ind    = np.where(R_part < R_max**2)
	if cut==1: #cube
		ind    = np.where((abs(x) < R_max) & (abs(y) < R_max) & (abs(z) < R_max))
		
	return ind
	
def rotation(theta, phi, x, y, z):
    xnew = np.cos(theta)*np.cos(phi)*x + np.cos(theta)*np.sin(phi)*y + np.sin(theta)*z
    ynew = -np.sin(phi)*x + np.cos(phi)*y
    znew = -np.sin(theta)*np.cos(phi)*x - np.sin(theta)*np.sin(phi)*y + np.cos(theta)*z
    return xnew, ynew, znew

def angular_momentum(r_disk, gas_mass, x_gas, y_gas, z_gas, vx_gas, vy_gas, vz_gas):
        lx = gas_mass*(y_gas*vz_gas - z_gas*vy_gas)
        ly = gas_mass*(z_gas*vx_gas - x_gas*vz_gas)
        lz = gas_mass*(x_gas*vy_gas - y_gas*vx_gas)

        L_x = np.sum(lx)
        L_y = np.sum(ly)
        L_z = np.sum(lz)

        L_old = np.sqrt(L_x**2 + L_y**2 + L_z**2)

        e_L_x = L_x/L_old
        e_L_y = L_y/L_old
        e_L_z = L_z/L_old

        theta = np.arccos(e_L_z)
        phi = np.arctan(e_L_y/e_L_x)

#        print '\nComponents of total angular momentum vector before rotation are: ', (L_x, L_y, L_z), '\n'
    
#        print '(e_L_x, e_L_y, e_L_z):', (e_L_x, e_L_y, e_L_z), '\n'

#        print 'Total angular momentum before rotation is: ', L_old, '\n'
    
#        print 'theta, phi: ', (theta, phi), '\n'
    
        return theta, phi




############################################################################################################

def find_mostmassive(file_dir, snapshot):
	snapshot = '%03d' % snapshot
        filein = glob.glob(file_dir+"/snap"+str(snapshot)+"Rpep*_halos")
        data   = np.loadtxt(filein[0])
        M      = data[:,3]
        indmax = 0 #M.argmax()

        xc    = data[indmax,5]
        yc    = data[indmax,6]
        zc    = data[indmax,7]
        Mmax  = M[indmax]

        return xc,yc,zc,Mmax

###########################################################################################################

def return_h2_fraction(gas_density, smoothing_gas, Z_prime, clumping_factor):
	local_column_density            =  gas_density*smoothing_gas*1.e10/1.e6
	tau_c                           = clumping_factor*local_column_density*2.e3*M_sun/(pc**2.)*Z_prime
	chi                             = 3.1*(1.+3.1*Z_prime**0.365)/4.1
	s                               = np.log(1.+0.6*chi+0.01*chi**2)/(0.6*tau_c)
	h2_fraction                     = 1. - (3.*s)/(4.*(1.+ 0.25*s))
	h2_fraction[h2_fraction < 0. ]  = 0.
	
	return h2_fraction 





 
