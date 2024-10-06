import os
import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import *
from scipy.spatial import *
from scipy.linalg import *
import matplotlib
from matplotlib import rcParams
import glob
import funcs
import readsnap



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



plt.rcParams.update({'axes.labelsize': 20, 'font.size': 20,
                    'legend.fontsize': 16, 'xtick.labelsize': 20, 
                    'ytick.labelsize': 20,
                    'axes.linewidth': 6,
                    'xtick.labelsize': 20})
    
font = {'family' : 'normal',
        'weight' : 'normal', 
        'size'   : 20}
matplotlib.rc('font', **font)


#change font
plt.rc('text', usetex=True)
plt.rc('font',**{'family':'serif','serif':['Palatino']})

#####################################################

snapshot_i = 599
snapshot_f = 599
nsteps = 1 #SPECIFY THE LOOP STEP FOR SNAPSHOTS

file_dir = '/mnt/scratch-lustre/lakhlani/m12b_res56000/output/' 

#file_dir = '/mnt/scratch-lustre/lakhlani/m13_mr_Dec16_2013/'

#file_dir = '/mnt/scratch-lustre/lakhlani/m12qq_hr_Dec16_2013/'

out_dir = '/home/lakhlani/Figures/m12b_res56000/'

#out_dir = '/home/lakhlani/Figures/m12qq_hr_Dec16_2013/'

#out_dir = '/home/lakhlani/Figures/m13_mr_Dec16_2013/'

for snapshot in range(snapshot_i, snapshot_f+1, nsteps):
        snapshot = '%03d' %snapshot
        P = readsnap.readsnap(file_dir, snapshot, 0, cosmological=1, loud = 1 )
        P_star = readsnap.readsnap(file_dir, snapshot, 4, cosmological = 1)
        P_dm = readsnap.readsnap(file_dir, snapshot, 1, cosmological = 1 )
        header_info = readsnap.readsnap(file_dir, snapshot, 0 ,cosmological=1,  header_only = 1)
        scale_factor = header_info['time']
        hubble_run = header_info['hubble']
        redshift = header_info['redshift']
	
        xc, yc, zc, Mhalo = funcs.find_mostmassive(file_dir, int(snapshot), 0)

        x_gas, y_gas, z_gas = P['p'][:,0], P['p'][:,1], P['p'][:,2]
        x_star, y_star, z_star = P_star['p'][:,0], P_star['p'][:,1], P_star['p'][:,2]
        x_dm, y_dm, z_dm = P_dm['p'][:,0], P_dm['p'][:,1], P_dm['p'][:,2]
        gas_mass = P['m']
        stellar_mass = P_star['m']
        dm_mass = P_dm['m']
        gas_density = P['rho']
        smoothing_gas = P['h']
        internal_energy = P['u']
        electron_abundance = P['ne']
        SFR = P['sfr']
        total_metallicity = P['z'][:,0]
        helium_mass_fraction = P['z'][:,1]
        vx_gas, vy_gas, vz_gas = P['v'][:,0], P['v'][:,1], P['v'][:,2]

        x_dm, y_dm, z_dm = funcs.change_coordinates(xc/hubble_run*scale_factor, yc/hubble_run*scale_factor, zc/hubble_run*scale_factor, x_dm, y_dm, z_dm)

        x_gas, y_gas, z_gas = funcs.change_coordinates(xc/hubble_run*scale_factor, yc/hubble_run*scale_factor, zc/hubble_run*scale_factor, x_gas, y_gas, z_gas)

        x_star, y_star, z_star = funcs.change_coordinates(xc/hubble_run*scale_factor, yc/hubble_run*scale_factor, zc/hubble_run*scale_factor, x_star, y_star, z_star)


        R_index = 20. #The max distance from the galactic center to which we keep the particles, done to speed up the calculations.

        R_gas = np.sqrt(x_gas**2 + y_gas**2 + z_gas**2)

	index_gas = np.where(R_gas < R_index)

        R_star = np.sqrt(x_star**2 + y_star**2 + z_star**2)
        index_star = np.where(R_star < R_index)


        Total_gas_mass_fraction = np.sum(gas_mass)/(np.sum(gas_mass) + np.sum(stellar_mass))

        theta, phi = funcs.angular_momentum(R_index, gas_mass, x_gas, y_gas, z_gas, vx_gas, vy_gas, vz_gas)

        x_gas, y_gas, z_gas = funcs.rotation(theta, phi, x_gas, y_gas, z_gas)

        vx_gas, vy_gas, vz_gas = funcs.rotation(theta, phi, vx_gas, vy_gas, vz_gas)

        theta_new, phi_new = funcs.angular_momentum(R_index, gas_mass, x_gas, y_gas, z_gas, vx_gas, vy_gas, vz_gas)

        x_gas, y_gas, z_gas = x_gas[index_gas], y_gas[index_gas], z_gas[index_gas]
        gas_mass, smoothing_gas, gas_density = gas_mass[index_gas], smoothing_gas[index_gas], gas_density[index_gas]
        R_gas = R_gas[index_gas]
        total_metallicity = total_metallicity[index_gas]


	SFR = SFR[index_star]
	R_star = R_star[index_star] 

##########################################################################
	n_bins = 2*R_index + 1.
	
	
        radius_bins = np.linspace(0., R_index, n_bins)

        radius_center_bins = np.linspace(0, R_index, n_bins-1)

        annulus_area = np.pi*(radius_bins[1:]**2 - radius_bins[:-1]**2)


        digitized_gas = np.digitize(R_gas, radius_bins)

	digitized_star = np.digitize(R_star, radius_bins) 

        gas_mass_digitized = np.asarray([gas_mass[digitized_gas == i].sum() for i in range(1, len(radius_bins))])

        sigma_gas = np.asarray([gas_mass[digitized_gas == i].sum()/annulus_area[i-1] for i in range(1, len(radius_bins))])*1.e10/1.e6
	
	sigma_SFR = np.asarray([SFR[digitized_star == i].sum()/annulus_area[i-1] for i in range(1, len(radius_bins))])

 ####################################################################################################

	
	#clumping_factor = np.array([1., 2., 5., 10.])
	#for i in range(0., clumping_factor+1.):

	Z_prime = total_metallicity/solar_metallicity

	#Z_prime_annulus_mean	 	= np.asarray([np.mean(Z_prime[digitized_gas == i]) for i in range(1, len(radius_bins))])
	#local_column_density           =  gas_density*smoothing_gas*1.e10/1.e6
	#local_column_density 		= sigma_gas 
	#tau_c                		= 6.*local_column_density*2.e3*M_sun/(pc**2.)*Z_prime
	#tau_c 				= 4.*local_column_density*2.e3*Z_prime_annulus_mean*M_sun/(pc**2.)
	#chi                  		= 3.1*(1.+3.1*Z_prime**0.365)/4.1
	#chi_annulus_averaged		= 3.1*(1.+3.1*Z_prime_annulus_mean**0.365)/4.1
	#chi_annulus_averaged 		= np.asarray([tau_c[digitized_gas == i].sum()/annulus_area[i-1] for i in range(1, len(radius_bins))])
	#s                   		= np.log(1.+0.6*chi+0.01*chi**2)/(0.6*tau_c)
	#s 				= np.log(1.+0.6*chi_annulus_averaged+0.01*chi_annulus_averaged**2.)/(0.6*tau_c) 
	#h2_fraction          		= 1. - (3.*s)/(4.*(1.+ 0.25*s))
	#h2_fraction[h2_fraction < 0. ]  = 0.
	
	h2_fraction_1  			= funcs.return_h2_fraction(gas_density, smoothing_gas, Z_prime, 1.0)
	h2_fraction_2 			= funcs.return_h2_fraction(gas_density, smoothing_gas, Z_prime, 2.0)
	h2_fraction_6			= funcs.return_h2_fraction(gas_density, smoothing_gas, Z_prime, 6.0)
	h2_fraction_10			= funcs.return_h2_fraction(gas_density, smoothing_gas, Z_prime, 10.0)
	mass_h2_1      			= gas_mass*h2_fraction_1
	mass_h2_2			= gas_mass*h2_fraction_2
	mass_h2_6			= gas_mass*h2_fraction_6
	mass_h2_10			= gas_mass*h2_fraction_10
	
	#total_h2_mass 			= np.sum(mass_h2)
	#gas_mass_total 			= np.sum(gas_mass)  #gas mass in R_index kpc from the center (in 10^10 M_sun)
	#h2_mass_fraction 		= total_h2_mass/ gas_mass_total
	#SFR_total			= np.sum(SFR) #SFR in R_index kpc from the center (in units of M_sun/year )  




	#h2_mass_digitized 	= np.asarray([mass_h2[digitized_gas == i].sum() for i in range(1, len(radius_bins))])

	sigma_h2_1 		= np.asarray([mass_h2_1[digitized_gas == i].sum()/annulus_area[i-1] for i in range(1, len(radius_bins))])*1.e10/1.e6

	sigma_h2_2              = np.asarray([mass_h2_2[digitized_gas == i].sum()/annulus_area[i-1] for i in range(1, len(radius_bins))])*1.e10/1.e6
	sigma_h2_6              = np.asarray([mass_h2_6[digitized_gas == i].sum()/annulus_area[i-1] for i in range(1, len(radius_bins))])*1.e10/1.e6
	sigma_h2_10              = np.asarray([mass_h2_10[digitized_gas == i].sum()/annulus_area[i-1] for i in range(1, len(radius_bins))])*1.e10/1.e6
	
	#sigma_h2_particles 	= mass_h2/(np.pi*smoothing_gas**2.)*1.e4

	#sigma_h2_gas_average 	= np.asarray([np.sum(gas_mass[digitized_gas == i]*sigma_h2_particles[digitized_gas == i])/np.sum(gas_mass[digitized_gas == i]) for i in range(1, len(radius_bins))])

	#sigma_h2_no_gas 	= np.asarray([np.sum(sigma_h2_particles[digitized_gas == i])/sigma_h2_particles[digitized_gas == i].size for i in range(1, len(radius_bins))])



	#mass_h1 		= gas_mass - mass_h2

	#sigma_h1 		= np.asarray([mass_h1[digitized_gas == i].sum()/annulus_area[i-1] for i in range(1, len(radius_bins))])*1.e10/1.e6

	
	f_h2_mass_average 	= np.asarray([np.mean(gas_mass[digitized_gas ==i]*h2_fraction_6[digitized_gas == i]/gas_mass[digitized_gas == i]) for i in range(1, len(radius_bins))]) 

	f_h2_annulus_1            = sigma_h2_1/(sigma_gas)
	
	f_h2_annulus_2            = sigma_h2_2/(sigma_gas)
	f_h2_annulus_6            = sigma_h2_6/(sigma_gas)
	f_h2_annulus_10           = sigma_h2_10/(sigma_gas)
#####################################################################################################################
# 	NOW WE BEGIN THE PLOTTING ROUTINES 
#
#
#####################################################################################################################
	

	'''
	#####DATA FROM MARC-ANTOINE, MURRAY, LEE ET AL 2017. 
	unit_x_1 = (450. - 112.)/5.
	unit_y_1 = 359. - 166. 
	
	unit_x_2 = unit_x_1 
	unit_y_2 = 654. - 349.
	
	txt_load = np.loadtxt('murray_et_al_plot_coordinates.txt')  
	x0, y0, x01, y01 = txt_load[0][0], txt_load[0][1], txt_load[0][4], txt_load[0][5]
	x_sigma_h2, y_sigma_h2, x_sigma_h1, y_sigma_h1 = txt_load[:,0], txt_load[:,1], txt_load[:,2], txt_load[:,3]

	x_fh2, y_fh2 				       = txt_load[:,4], txt_load[:,5]

	x_sigma_h2, y_sigma_h2, x_sigma_h1, y_sigma_h1 = np.delete(x_sigma_h2, 0), np.delete(y_sigma_h2, 0),np.delete(x_sigma_h1, 0), np.delete(y_sigma_h1, 0)

	x_fh2, y_fh2 				       = np.delete(x_fh2, 0), np.delete(y_fh2, 0)

	
	####WE TRANSFORM THE COORDINATES TO WHAT WE WILL USE TO PLOT 
	x_sigma_h2_to_use = (x_sigma_h2 - x0)/unit_x_1 
	y_sigma_h2_to_use = 10**(np.abs(y_sigma_h2 - y0)/unit_y_1 - 3.) 
	
	x_sigma_h1_to_use = (x_sigma_h1 - x0)/unit_x_1
	y_sigma_h1_to_use = 10**(np.abs(y_sigma_h1 - y0)/unit_y_1 - 3.) 
	
	
	x_fh2_to_use 	  = (x_fh2 - x01)/unit_x_2
	
	y_fh2_to_use 	  = 10**(np.abs(y_fh2 - y01)/unit_y_2 - 3.)	


	#CONVERT REDSHIFT TO REQUIRED PRECISION 
        redshift =  '%1.3f' %redshift

	fig = plt.figure(figsize=(8,5))
        ax = plt.subplot(111)
        for axis in ['top','bottom','left','right']:
                ax.spines[axis].set_linewidth(2)

        ax.tick_params('both', length=10, width=2, which='major', color='k')
        ax.tick_params('both', length=6, width=1.5, which='minor', color='k')

        ax.plot(radius_center_bins, gas_mass_digitized*1.e2, 'o')
	ax.plot(radius_center_bins, gas_mass_digitized*1.e2) 
        ax.set_xlabel(r'$R_{ \rm g}$[Kpc]')
        ax.set_ylabel(r'$M_{\rm gas}$[$10^{8} \rm M_{\odot}$]')
        plt.savefig(out_dir+"Gas_Mass_vs_R_plot_"+str(snapshot)+"_"+str(redshift)+"_.pdf", bbox_inches="tight")
        ax.cla()


        ax.plot(radius_center_bins, sigma_gas, 'o')
        ax.set_xlim(0., 20.)
        ax.set_yscale('log')
        ax.set_xlabel(r'$R_{\rm g}$[Kpc]')
        ax.set_ylabel(r'$\Sigma_{\rm gas}\ [\rm M_{\odot} pc^{-2} \rm]$')
        plt.savefig(out_dir+"Sigma_gas_vs_R_plot_"+str(snapshot)+"_"+str(redshift)+"_.pdf", bbox_inches="tight")
        ax.cla()

        #ax.plot(radius_center_bins, sigma_h2_1, 'o', label = r'$\Sigma_{H2}$ This Study, 1.')
        #ax.plot(radius_center_bins, sigma_h2_2, 'o', label = r'$\Sigma_{H2}$ This Study, 2.')
	ax.plot(radius_center_bins, sigma_h2_6, 'o', label = r'This Study')
	#ax.plot(radius_center_bins, sigma_h2_10, 'o', label = r'$\Sigma_{H2}$ This Study, 10.')
	ax.plot(radius_center_bins, sigma_gas, 'o', label=r'$\Sigma_{gas}$')	
	#ax.plot(radius_center_bins, sigma_h2_gas_average, 'ro')
        #ax.plot(radius_center_bins, sigma_h2_no_gas, 'go')
        ax.plot(x_sigma_h2_to_use, y_sigma_h2_to_use, '*', label = r'$\Sigma_{H2}$, Milville-Deschenes et al. 2017')
	#ax.plot(radius_center_bins, sigma_h1, 'o', label = r'$\Sigma_{HI}$ This Study')
	ax.plot(x_sigma_h1_to_use, y_sigma_h1_to_use+y_sigma_h2_to_use, '*', label = r'$\Sigma{gas}$, Milville-Deschenes et al. 2017') 
	ax.set_xlim(0., 20.)
        ax.set_ylim(1.e-3, 1.e2)
        ax.set_yscale('log')
        ax.set_xlabel(r'$R_{\rm g}$[Kpc]')
        ax.set_ylabel(r'$\Sigma_{\rm H_{2}}\ [\rm M_{\odot} pc^{-2} \rm]$')
	ax.legend(frameon=False, loc='best')
        plt.savefig(out_dir+"Sigma_H2_vs_R_plot_"+str(snapshot)+"_"+str(redshift)+"_.pdf", bbox_inches="tight")
	ax.cla()

        #ax.plot(radius_center_bins, f_h2_annulus_1, 'o', label='1.')
        #ax.plot(radius_center_bins, f_h2_annulus_2, 'o', label='2.')
	#ax.plot(radius_center_bins, f_h2_annulus_6, 'o', label='This Study')
	ax.plot(x_fh2_to_use, y_fh2_to_use, '*', label = 'Milville-Deschenes et al. 2017')
	ax.plot(radius_center_bins, f_h2_mass_average, 'o', label= 'This Study.')
	#ax.plot(radius_center_bins, f_h2_annulus_10, 'o', label='10.')
	
	ax.set_xlim(0., 20.)
        ax.set_ylim(1.e-3, 1.e0)
        ax.set_yscale('log')
        ax.set_xlabel(r'$R_{\rm g}$[Kpc]')
        ax.set_ylabel(r'$\rm f_{\rm H_{2}}$')
        ax.legend(frameon=False, loc='best')
	plt.savefig(out_dir+"f_H2_vs_R_plot_"+str(snapshot)+"_"+str(redshift)+"_.pdf", bbox_inches="tight")
        ax.cla()


	plt.close(fig)

'''


















