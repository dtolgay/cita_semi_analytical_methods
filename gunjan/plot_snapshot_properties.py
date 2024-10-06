import os
import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import *
from scipy.spatial import *
from scipy.linalg import *
from scipy.ndimage.filters import gaussian_filter
import matplotlib
from matplotlib import rcParams
import glob
import funcs
import readsnap
from matplotlib.gridspec import GridSpec


plt.rcParams.update({'axes.labelsize': 20, 'font.size': 20,
                    'legend.fontsize': 16, 'ytick.labelsize': 25,
                    'axes.linewidth': 1,
                    'xtick.labelsize': 25, 'xtick.major.size':10, 'ytick.major.size':10, 'xtick.major.width':2, 'ytick.major.width':2})

plt.rcParams['text.latex.preamble']=[r'\usepackage{lmodern}']

params = {'text.usetex': True, 'font.size': 20, 'font.family': 'lmodern', 'text.latex.unicode': True}

plt.rcParams.update(params)

##########################################################
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
##########################################################

#file_dir = '/mnt/raid-project/murray/lakhlani/FIRE2_core/m12c_res56000/output/'
#out_dir = '/home/lakhlani/Figures/m12c_res56000/'

file_dir = '/mnt/raid-project/murray/lakhlani/FIRE2_core/m12b_res56000/output/'
out_dir = '/home/lakhlani/Figures/m12b_res56000/'

snapshot_i = 100
snapshot_f = 200 
nsteps     = 1 #SPECIFY THE LOOP STEP FOR SNAPSHOTS

for snapshot in range(snapshot_i, snapshot_f+1, nsteps):
        snapshot = '%03d' %snapshot #SNAPSHOT NUMBER AS THREE DIGIT INTEGER
        
	P = readsnap.readsnap(file_dir, snapshot, 0, cosmological=1, loud = 1 ) #READING IN THE GAS PARTICLES AS DICTIONARY
        P_star = readsnap.readsnap(file_dir, snapshot, 4, cosmological = 1) #READING IN THE STAR PARTICLES AS DICTIONARY 
        P_dm = readsnap.readsnap(file_dir, snapshot, 1, cosmological = 1 ) #READING IN THE DARK MATTER PARTICLES AS DICTIONARY 
        header_info = readsnap.readsnap(file_dir, snapshot, 0 ,cosmological=1,  header_only = 1) #READING IN THE HEADER INFO AS DICTIONARY
        scale_factor = header_info['time'] 
        hubble_run = header_info['hubble'] #HUBBLE CONSTANT FOR THE RUN 
        redshift = header_info['redshift']
#################################################################################################################################

        xc, yc, zc, Mhalo = funcs.find_mostmassive(file_dir, int(snapshot), 0) #LOADING IN THE CENTER OF THE HALO FOR A GIVEN SNAPSHOT 

        x_gas, y_gas, z_gas = P['p'][:,0], P['p'][:,1], P['p'][:,2]
        x_star, y_star, z_star = P_star['p'][:,0], P_star['p'][:,1], P_star['p'][:,2]
        x_dm, y_dm, z_dm = P_dm['p'][:,0], P_dm['p'][:,1], P_dm['p'][:,2]
        gas_mass = P['m']
        stellar_mass = P_star['m']
        star_age = P_star['age']
        dm_mass = P_dm['m']
        gas_density = P['rho']
        smoothing_gas = P['h']
        internal_energy = P['u']
        electron_abundance = P['ne']
        SFR = P['sfr']
        total_metallicity = P['z'][:,0]
        helium_mass_fraction = P['z'][:,1]
        vx_gas, vy_gas, vz_gas = P['v'][:,0], P['v'][:,1], P['v'][:,2]
#############################################################################################################
	#CHANGING THE COORDINATES OF THE CENTER OF THE HALO, FOR STAR, DM AND GAS PARTICLES 
        x_dm, y_dm, z_dm = funcs.change_coordinates(xc/hubble_run*scale_factor, yc/hubble_run*scale_factor, zc/hubble_run*scale_factor, x_dm, y_dm, z_dm) 
	x_gas, y_gas, z_gas = funcs.change_coordinates(xc/hubble_run*scale_factor, yc/hubble_run*scale_factor, zc/hubble_run*scale_factor, x_gas, y_gas, z_gas)
	x_star, y_star, z_star = funcs.change_coordinates(xc/hubble_run*scale_factor, yc/hubble_run*scale_factor, zc/hubble_run*scale_factor, x_star, y_star, z_star)
#############################################################################################################

	#CUT PARTICLES

        R_index = 100. #The max distance from the galactic center to which we keep the particles, done to speed up the calculations.
	
	R_index_1 = 20. 

        index_gas  = funcs.cut_particles_index(x_gas,y_gas,z_gas,R_index,1)
        index_dm   = funcs.cut_particles_index(x_dm,y_dm,z_dm,R_index,1)
        index_star = funcs.cut_particles_index(x_star,y_star,z_star,R_index,1)

        x_gas  = x_gas[index_gas]
        y_gas  = y_gas[index_gas]
        z_gas  = z_gas[index_gas]
        vx_gas = vx_gas[index_gas]
        vy_gas = vy_gas[index_gas]
        vz_gas = vz_gas[index_gas]
        gas_mass      = gas_mass[index_gas]
        smoothing_gas = smoothing_gas[index_gas]
        gas_density   = gas_density[index_gas]
	helium_mass_fraction = helium_mass_fraction[index_gas]
	internal_energy = internal_energy[index_gas]
	electron_abundance = electron_abundance[index_gas]


        x_dm    = x_dm[index_dm]
        y_dm    = y_dm[index_dm]
        z_dm    = z_dm[index_dm]
        dm_mass = dm_mass[index_dm]

        x_star = x_star[index_star]
        y_star = y_star[index_star]
        z_star = z_star[index_star]
        stellar_mass = stellar_mass[index_star]

        R_gas             = np.sqrt(x_gas**2 + y_gas**2 + z_gas**2)
        total_metallicity = total_metallicity[index_gas]

        Total_gas_mass_fraction = np.sum(gas_mass)/(np.sum(gas_mass) + np.sum(stellar_mass))

	theta, phi              = funcs.angular_momentum(R_index, gas_mass, x_gas, y_gas, z_gas, vx_gas, vy_gas, vz_gas)
        print theta, phi

	x_gas, y_gas, z_gas     = funcs.rotation(theta, phi, x_gas, y_gas, z_gas)
        vx_gas, vy_gas, vz_gas  = funcs.rotation(theta, phi, vx_gas, vy_gas, vz_gas)

        x_dm, y_dm, z_dm        = funcs.rotation(theta, phi, x_dm, y_dm, z_dm)
        x_star, y_star, z_star  = funcs.rotation(theta, phi, x_star, y_star, z_star)

        theta_new, phi_new      = funcs.angular_momentum(R_index, gas_mass, x_gas, y_gas, z_gas, vx_gas, vy_gas, vz_gas)
		

        Z_prime                 = total_metallicity/solar_metallicity

        h2_mass                 = gas_mass*funcs.return_h2_fraction(gas_density, smoothing_gas, Z_prime, 1)
	
	temperature 		= funcs.internal_energy_to_temperature(internal_energy, helium_mass_fraction, electron_abundance) 
	
	gas_density_cgs 	= gas_density*1.e10*M_sun/(kpc)**3.

	number_density 		= funcs.mass_density_to_number_density(gas_density_cgs) 

		
	number_density_binned = binned_statistic_2d(x_gas, y_gas, number_density, statistic='sum', bins=1024, range=[[-R_index,R_index],[-R_index,R_index]]) 

	
	temperature_binned    = binned_statistic_2d(x_gas, y_gas, temperature, statistic='sum', bins=1024, range=[[-R_index, R_index],[-R_index, R_index]]) 


	gas_mass_binned = binned_statistic_2d(x_gas, y_gas, gas_mass*1.e10, statistic='sum', bins=1024, range=[[-R_index, R_index],[-R_index, R_index]])	

	gas_mass_binned_1 = binned_statistic_2d(x_gas, y_gas, gas_mass*1.e10, statistic='sum', bins=1024, range=[[-R_index_1, R_index_1],[-R_index_1, R_index_1]])	

	number_density_filtered = np.log10(gaussian_filter(number_density_binned[0], sigma=2.0)+1e-9)
	
	temperature_filtered = np.log10(gaussian_filter(temperature_binned[0], sigma=15.0)+1.e-9) 
	
	gas_mass_filtered = np.log10(gaussian_filter(gas_mass_binned[0], sigma=1.5)+1.e-8) 
		
	gas_mass_filtered_1 = np.log10(gaussian_filter(gas_mass_binned_1[0], sigma=1.5)+1.e-9)

	fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 15)) 

	cax = ax1.imshow(gas_mass_filtered,vmin=4, vmax=7, interpolation='spline16',origin='lower',extent=[-R_index,R_index,-R_index,R_index]) 

	cax = ax2.imshow(gas_mass_filtered_1, vmin=4, vmax=7, interpolation='spline16', origin='lower', extent=[-R_index_1,R_index_1,-R_index_1,R_index_1])

	ax1.set_xlabel(r'x [kpc]')
	ax1.set_ylabel(r'y [kpc]') 
	ax1.tick_params(axis='both', colors='white')
	plt.setp(ax1.get_xticklabels(), color="black")
	plt.setp(ax1.get_yticklabels(), color="black")
	
	redshift = '%1.1f' %redshift
	
	ax2.text(12,-14, r"z = "+str(redshift), color="w", fontsize=20, fontweight='bold')

	cbar = fig.colorbar(cax) 
	
	cbar.ax.set_ylabel(r' log($\rm{\Sigma}_{gas} \left[ \rm{M}_{\odot}/\rm{Kpc}^{2} \right ]$)') 

	line = np.linspace(-9, -8, 1000)

	#plt.gca().xaxis.set_major_locator(plt.NullLocator())
        #plt.gca().yaxis.set_major_locator(plt.NullLocator())

	#ax.plot(line, line*0+8, 'w', linewidth=4)

	#ax.text(-9,8.5 , r'Kpc', color='w', fontsize=30, fontweight='bold')
	ax2.text(-15, -15, r'm12b FIRE2', color='w', fontsize=20, fontweight='bold') 


	ax2.set_xlabel(r'x [kpc]')
        ax2.set_ylabel(r'y [kpc]')
        ax2.tick_params(axis='both', colors='white')
        plt.setp(ax2.get_xticklabels(), color="black")
        plt.setp(ax2.get_yticklabels(), color="black")

	#cbar = fig.colorbar(cax1) 
	#cbar.ax.set_ylabel(r' log($\rm{\Sigma}_{gas} \left[ \rm{M}_{\odot}/\rm{Kpc}^{2} \right ]$)')

	
	plt.savefig(out_dir+"_gas_mass_imshow_20_100_kpc_"+str(snapshot)+".pdf", bbox_inches="tight")

	plt.close(fig)

'''
	
	fig1, ax1 = plt.subplots()
	
	cax1 = ax1.imshow(temperature_filtered, vmin=1, interpolation='spline16', origin='lower', extent=[-R_index,R_index,-R_index,R_index])
	ax1.set_xlabel(r'Kpc')
        ax1.set_ylabel(r'Kpc')
	
	ax1.text(4.5,8, r"z = "+str(redshift), color="black", fontsize=30, fontweight='bold')

        cbar1 = fig1.colorbar(cax1)
	cbar1.ax.set_ylabel(r' log(T/K)')

	ax1.text(-8, -8, r'm12b FIRE2', color='black', fontsize=30, fontweight='bold')
	


	plt.close(fig1) 

'''
	

	###################################################################################################
	##################FANCY HISTOGRAM #################################################################
'''
	plt.rcParams.update({'axes.labelsize': 12, 'font.size': 12,
                    'legend.fontsize': 12, 'ytick.labelsize': 12,
                    'axes.linewidth': 1,
                    'xtick.labelsize': 12, 'xtick.major.size':1, 'ytick.major.size':1, 'xtick.major.width':1, 'ytick.major.width':1})

	binsx = np.linspace(-5, 4, 512)
	binsy = np.linspace(1, 7, 512) 
	
	bins_x_hist = np.linspace(-5, 4, 25) 
	bins_y_hist = np.linspace(1, 7, 25) 
	
	nd_T_histogram = np.histogram2d(np.log10(number_density), np.log10(temperature), weights = gas_mass, bins=(binsx, binsy)) 
	
	hist_up = np.histogram(np.log10(number_density), weights=gas_mass, bins=bins_x_hist) 
	hist_side = np.histogram(np.log10(temperature), weights=gas_mass, bins=bins_y_hist) 

	np.savez("/home/lakhlani/histogram_arrays.npz", nd_T_histogram_array=nd_T_histogram, hist_up_array=hist_up, hist_side_array=hist_side) 
	
	fig2 = plt.figure() 
	gs = GridSpec(4, 4) 
	
	ax2_joint  = fig2.add_subplot(gs[1:3, 0:2])
	ax2_marg_x = fig2.add_subplot(gs[0, 0:2])
	ax2_marg_y = fig2.add_subplot(gs[1:4, 2])

	c = ax2_joint.imshow(np.log10(nd_T_histogram[0].T), origin='lower', extent=[-5, 4, 1, 7]) 
	#cax2 = ax2_joint.hist2d(np.log10(number_density), np.log10(temperature), weights=gas_mass, bins=(binsx, binsy)) 
	cbar2= fig2.colorbar(c, orientation='horizontal', pad=0.15, aspect=50 )  
	cbar2.ax.set_xlabel(r'log baryon mass fraction')	

	ax2_marg_x.bar(bins_x_hist[:-1], hist_up[0]/np.sum(hist_up[0]), width=0.35)
	ax2_marg_x.set_xlim(-5, 4)
	ax2_marg_x.xaxis.set_major_locator(plt.NullLocator()) 



	ax2_marg_y.barh(bins_y_hist[:-1], hist_side[0]/np.sum(hist_side[0]), align='center', height=0.25) 
	ax2_marg_y.set_ylim(1., 7.) 
	ax2_marg_y.yaxis.set_major_locator(plt.NullLocator())

	
       	plt.savefig('/home/lakhlani/Desktop/nd_T_histogram.pdf', bbox_inches='tight')  
'''














