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
                    'legend.fontsize': 16, 'ytick.labelsize': 20,
                    'axes.linewidth': 2,
                    'xtick.labelsize': 20})
    
plt.rcParams['text.latex.preamble']=[r'\usepackage{lmodern}']

params = {'text.usetex': True, 'font.size': 20, 'font.family': 'lmodern', 'text.latex.unicode': True}

plt.rcParams.update(params) 

cmapF = plt.get_cmap('inferno') 


#####################################################

snapshot_i = 440
snapshot_f = 440
nsteps = 1 #SPECIFY THE LOOP STEP FOR SNAPSHOTS

#file_dir = '/mnt/scratch-lustre/lakhlani/m12b_res56000/output/' 

#file_dir = '/mnt/scratch-lustre/lakhlani/m13_mr_Dec16_2013/'

file_dir = '/mnt/raid-project/murray/lakhlani/FIRE1_core/m12qq_hr_Dec16_2013/'
#file_dir = '/mnt/raid-project/murray/lakhlani/FIRE2_core/m12c_res56000/output/'

#file_dir = '/mnt/raid-project/murray/lakhlani/FIRE1_core/m11v_lr_Jan2014/'
#out_dir = '/home/lakhlani/Figures/m11v_lr_Jan2014/'

#file_dir = '/mnt/raid-project/murray/lakhlani/FIRE1_core/m12v_mr_Dec5_2013_3/'
#out_dir = '/home/lakhlani/Figures/m12v_mr_Dec5_2013_3/'

#file_dir = '/mnt/raid-project/murray/FIRE/FIRE_1/m09_hr_Dec16_2013/'
#out_dir = '/home/lakhlani/Figures/m09_hr_Dec16_2013/'
#out_dir = '/home/lakhlani/Figures/m12c_res56000/'
#out_dir = '/home/lakhlani/Figures/m12b_res56000/'

out_dir = '/home/lakhlani/Figures/m12qq_hr_Dec16_2013/'

#out_dir = '/home/lakhlani/Figures/m13_mr_Dec16_2013/'

#file_dir = '/mnt/raid-project/murray/FIRE/FIRE_2/m11a/output/'

#out_dir = '/home/lakhlani/Figures/m11a/'

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

        x_dm, y_dm, z_dm = funcs.change_coordinates(xc/hubble_run*scale_factor, yc/hubble_run*scale_factor, zc/hubble_run*scale_factor, x_dm, y_dm, z_dm)

        x_gas, y_gas, z_gas = funcs.change_coordinates(xc/hubble_run*scale_factor, yc/hubble_run*scale_factor, zc/hubble_run*scale_factor, x_gas, y_gas, z_gas)

        x_star, y_star, z_star = funcs.change_coordinates(xc/hubble_run*scale_factor, yc/hubble_run*scale_factor, zc/hubble_run*scale_factor, x_star, y_star, z_star)


        R_index = 20 #The max distance from the galactic center to which we keep the particles, done to speed up the calculations.

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

	SFR = SFR[index_gas]

	internal_energy = internal_energy[index_gas] 
	helium_mass_fraction = helium_mass_fraction[index_gas]
	electron_abundance = electron_abundance[index_gas] 
	
	
	R_star = R_star[index_star] 

	stellar_mass = stellar_mass[index_star]	
	
	star_age = star_age[index_star] 


	print 'Total Gas Mass: ', np.sum(gas_mass)
##########################################################################
	n_bins = 2*R_index + 1.
	
	
        radius_bins = np.linspace(0., R_index, n_bins)

        radius_center_bins = np.linspace(0, R_index, n_bins-1)

        annulus_area = np.pi*(radius_bins[1:]**2 - radius_bins[:-1]**2)


        digitized_gas = np.digitize(R_gas, radius_bins)

	digitized_star = np.digitize(R_star, radius_bins) 

        gas_mass_digitized = np.asarray([gas_mass[digitized_gas == i].sum() for i in range(1, len(radius_bins))])

        sigma_gas = np.asarray([gas_mass[digitized_gas == i].sum()/annulus_area[i-1] for i in range(1, len(radius_bins))])*1.e10/1.e6
	
	sigma_SFR = np.asarray([SFR[digitized_gas == i].sum()/annulus_area[i-1] for i in range(1, len(radius_bins))])

	radial_velocity = vx_gas*np.cos(phi)*np.sin(theta)+vy_gas*np.sin(phi)*np.sin(theta)+vz_gas*np.cos(theta)


 ####################################################################################################

	
	Z_prime = total_metallicity/solar_metallicity


	h2_fraction_1  			= funcs.return_h2_fraction(gas_density, smoothing_gas, Z_prime, 1.0)
	h2_fraction_2 			= funcs.return_h2_fraction(gas_density, smoothing_gas, Z_prime, 2.0)
	h2_fraction_6			= funcs.return_h2_fraction(gas_density, smoothing_gas, Z_prime, 6.0)
	h2_fraction_10			= funcs.return_h2_fraction(gas_density, smoothing_gas, Z_prime, 10.0)
	mass_h2_1      			= gas_mass*h2_fraction_1
	mass_h2_2			= gas_mass*h2_fraction_2
	mass_h2_6			= gas_mass*h2_fraction_6
	mass_h2_10			= gas_mass*h2_fraction_10

	h2_mass_total_1 		= np.sum(mass_h2_1)
	h2_mass_total_2                 = np.sum(mass_h2_2)
	h2_mass_total_6                 = np.sum(mass_h2_6)
	h2_mass_total_10                 = np.sum(mass_h2_10)
	
	gas_mass_total 			= np.sum(gas_mass)  #gas mass in R_index kpc from the center (in 10^10 M_sun)
	SFR_total			= np.sum(SFR) #SFR in R_index kpc from the center (in units of M_sun/year )  



	#h2_mass_digitized 	= np.asarray([mass_h2[digitized_gas == i].sum() for i in range(1, len(radius_bins))])


	
	sigma_h2_1 		= np.asarray([mass_h2_1[digitized_gas == i].sum()/annulus_area[i-1] for i in range(1, len(radius_bins))])*1.e10/1.e6

	sigma_h2_2              = np.asarray([mass_h2_2[digitized_gas == i].sum()/annulus_area[i-1] for i in range(1, len(radius_bins))])*1.e10/1.e6
	sigma_h2_6              = np.asarray([mass_h2_6[digitized_gas == i].sum()/annulus_area[i-1] for i in range(1, len(radius_bins))])*1.e10/1.e6
	sigma_h2_10              = np.asarray([mass_h2_10[digitized_gas == i].sum()/annulus_area[i-1] for i in range(1, len(radius_bins))])*1.e10/1.e6
	
	sigma_h2_1[sigma_h2_1 == 0. ] = 1.e-30 
	sigma_h2_2[sigma_h2_2 == 0. ] = 1.e-30
	sigma_h2_6[sigma_h2_6 == 0. ] = 1.e-30
	sigma_h2_10[sigma_h2_10 == 0. ] = 1.e-30

	sigma_gas[sigma_gas == 0.] = 1.e-30

	#mass_h1 		= gas_mass - mass_h2

	#sigma_h1 		= np.asarray([mass_h1[digitized_gas == i].sum()/annulus_area[i-1] for i in range(1, len(radius_bins))])*1.e10/1.e6

	mass_h2_1_digitized  = np.asarray([np.sum(mass_h2_1[digitized_gas == i] ) for i in range(1, len(radius_bins))])

	mass_h2_2_digitized  = np.asarray([np.sum(mass_h2_2[digitized_gas == i] ) for i in range(1, len(radius_bins))])

	mass_h2_6_digitized  = np.asarray([np.sum(mass_h2_6[digitized_gas == i] ) for i in range(1, len(radius_bins))])

	mass_h2_10_digitized = np.asarray([np.sum(mass_h2_10[digitized_gas == i] ) for i in range(1, len(radius_bins))])
	

	mass_h2_1_digitized[mass_h2_1_digitized == 0.] = 1.e-30 
	mass_h2_2_digitized[mass_h2_2_digitized == 0.] = 1.e-30
	mass_h2_6_digitized[mass_h2_6_digitized == 0.] = 1.e-30
	mass_h2_10_digitized[mass_h2_10_digitized == 0.] = 1.e-30 

	flux_digitized = np.asarray([np.sum(gas_density[digitized_gas==i]*radial_velocity[digitized_gas == i] )*annulus_area[i-1] for i in range(1, len(radius_bins))])

	Z_prime_mass_averaged_1 = np.asarray([np.sum(mass_h2_1[digitized_gas == i]*Z_prime[digitized_gas == i])/mass_h2_1_digitized[i-1] for i in range(1, len(radius_bins))])


	Z_prime_mass_averaged_2 = np.asarray([np.sum(mass_h2_2[digitized_gas == i]*Z_prime[digitized_gas == i])/mass_h2_2_digitized[i-1] for i in range(1, len(radius_bins))])

	Z_prime_mass_averaged_6 = np.asarray([np.sum(mass_h2_6[digitized_gas == i]*Z_prime[digitized_gas == i])/mass_h2_6_digitized[i-1] for i in range(1, len(radius_bins))])

	Z_prime_mass_averaged_10 = np.asarray([np.sum(mass_h2_10[digitized_gas == i]*Z_prime[digitized_gas == i])/mass_h2_10_digitized[i-1] for i in range(1, len(radius_bins))])



	Z_prime_mass_averaged_1[Z_prime_mass_averaged_1 == 0]   = 1.e-30
	Z_prime_mass_averaged_2[Z_prime_mass_averaged_2 == 0]   = 1.e-30
	Z_prime_mass_averaged_6[Z_prime_mass_averaged_6 == 0]   = 1.e-30
	Z_prime_mass_averaged_10[Z_prime_mass_averaged_10 == 0] = 1.e-30


	#f_h2_mass_average 	= np.asarray([np.mean(gas_mass[digitized_gas ==i]*h2_fraction_6[digitized_gas == i]/gas_mass[digitized_gas == i]) for i in range(1, len(radius_bins))]) 

	f_h2_annulus_1            = sigma_h2_1/(sigma_gas)
	f_h2_annulus_2            = sigma_h2_2/(sigma_gas)
	f_h2_annulus_6            = sigma_h2_6/(sigma_gas)
	f_h2_annulus_10           = sigma_h2_10/(sigma_gas)
	
	gas_density_cgs 	  = gas_density*1.e10*M_sun/(kpc)**3.

	index_gas_mass 		  = np.where(gas_density_cgs > 1.e-24)

	gas_mass_total_indexed    = np.sum(gas_mass[index_gas_mass])
	
	X_co_annulus_1		  = 1.3e21/(Z_prime_mass_averaged_1*sigma_h2_1**(0.5))
	X_co_annulus_2            = 1.3e21/(Z_prime_mass_averaged_2*sigma_h2_2**(0.5))
	X_co_annulus_6            = 1.3e21/(Z_prime_mass_averaged_6*sigma_h2_6**(0.5))
	X_co_annulus_10           = 1.3e21/(Z_prime_mass_averaged_10*sigma_h2_10**(0.5)) 

	alpha_co_annulus_1        = X_co_annulus_1/6.3e19
	alpha_co_annulus_2        = X_co_annulus_2/6.3e19
	alpha_co_annulus_6        = X_co_annulus_6/6.3e19
	alpha_co_annulus_10       = X_co_annulus_10/6.3e19

	L_co_annulus_1 		  = np.sum(mass_h2_1)/alpha_co_annulus_1
	L_co_annulus_2            = np.sum(mass_h2_2)/alpha_co_annulus_2
	L_co_annulus_6            = np.sum(mass_h2_6)/alpha_co_annulus_6
	L_co_annulus_10           = np.sum(mass_h2_10)/alpha_co_annulus_10

	L_co_total_1		  = np.sum(L_co_annulus_1)*4.9e-5*1.e10 #L_CO in units of  L_sun. 
	L_co_total_2              = np.sum(L_co_annulus_2)*4.9e-5*1.e10
	L_co_total_6              = np.sum(L_co_annulus_6)*4.9e-5*1.e10
	L_co_total_10             = np.sum(L_co_annulus_10)*4.9e-5*1.e10

	alpha_co_mw 		  = 3.17	

	L_co_fixed_1 		  = np.sum(mass_h2_1)/alpha_co_mw*4.9e-5*1.e10
	L_co_fixed_2              = np.sum(mass_h2_2)/alpha_co_mw*4.9e-5*1.e10
	L_co_fixed_6              = np.sum(mass_h2_6)/alpha_co_mw*4.9e-5*1.e10
	L_co_fixed_10             = np.sum(mass_h2_10)/alpha_co_mw*4.9e-5*1.e10

'''
	if snapshot==snapshot_i:
                with open(out_dir+'_newest_L_co_arrays_'+str(snapshot_i)+"-"+str(snapshot_f)+"_.txt",'w') as filein:
                        filein.write(str(snapshot)+' '+str(redshift)+' '+str(Mhalo)+' '+str(h2_mass_total_1)+' '+str(h2_mass_total_2)+' '+str(h2_mass_total_6)+' '+str(h2_mass_total_10)+' '+str(SFR_total)+' '+str(gas_mass_total)+' '+str(gas_mass_total_indexed)+' '+str(Total_gas_mass_fraction)+' '+str(L_co_total_1)+' '+str(L_co_total_2)+' '+str(L_co_total_6)+' '+str(L_co_total_10)+' '+str(L_co_fixed_1)+' '+str(L_co_fixed_2)+' '+str(L_co_fixed_6)+' '+str(L_co_fixed_10))
                        filein.write('\n')

        else:
                with open(out_dir+'_newest_L_co_arrays_'+str(snapshot_i)+"-"+str(snapshot_f)+"_.txt",'a') as filein:
			filein.write(str(snapshot)+' '+str(redshift)+' '+str(Mhalo)+' '+str(h2_mass_total_1)+' '+str(h2_mass_total_2)+' '+str(h2_mass_total_6)+' '+str(h2_mass_total_10)+' '+str(SFR_total)+' '+str(gas_mass_total)+' '+str(gas_mass_total_indexed)+' '+str(Total_gas_mass_fraction)+' '+str(L_co_total_1)+' '+str(L_co_total_2)+' '+str(L_co_total_6)+' '+str(L_co_total_10)+' '+str(L_co_fixed_1)+' '+str(L_co_fixed_2)+' '+str(L_co_fixed_6)+' '+str(L_co_fixed_10))
                        filein.write('\n')








#####################################################################################################################
# 	NOW WE BEGIN THE PLOTTING ROUTINES 
#
#
#####################################################################################################################
	
	#####DATA FROM MARC-ANTOINE, MURRAY, LEE ET AL 2017. 
	unit_x_1 = (450. - 112.)/5.
	unit_y_1 = 359. - 166. 
	
	unit_x_2 = unit_x_1 
	unit_y_2 = 654. - 349.

	unit_x_3 = (444.-108.)/5.
	unit_y_3 = 973. - 509.
	
	txt_load = np.loadtxt('murray_et_al_plot_coordinates.txt')  
	x0, y0, x01, y01 = txt_load[0][0], txt_load[0][1], txt_load[0][4], txt_load[0][5]
	
	x02, y02 = txt_load[0][6], txt_load[0][7]	

	x_sigma_h2, y_sigma_h2, x_sigma_h1, y_sigma_h1 = txt_load[:,0], txt_load[:,1], txt_load[:,2], txt_load[:,3]

	x_fh2, y_fh2 				       = txt_load[:,4], txt_load[:,5]

	x_sigma_h2, y_sigma_h2, x_sigma_h1, y_sigma_h1 = np.delete(x_sigma_h2, 0), np.delete(y_sigma_h2, 0),np.delete(x_sigma_h1, 0), np.delete(y_sigma_h1, 0)

	x_fh2, y_fh2 				       = np.delete(x_fh2, 0), np.delete(y_fh2, 0)

	x_gmc1, y_gmc1 = txt_load[:,6], txt_load[:,7]	
	x_gmc1, y_gmc1 = np.delete(x_gmc1, 0), np.delete(y_gmc1, 0)

	####WE TRANSFORM THE COORDINATES TO WHAT WE WILL USE TO PLOT 
	x_sigma_h2_to_use = (x_sigma_h2 - x0)/unit_x_1 
	y_sigma_h2_to_use = 10**(np.abs(y_sigma_h2 - y0)/unit_y_1 - 3.) 
	
	x_sigma_h1_to_use = (x_sigma_h1 - x0)/unit_x_1
	y_sigma_h1_to_use = 10**(np.abs(y_sigma_h1 - y0)/unit_y_1 - 3.) 
	
	
	x_fh2_to_use 	  = (x_fh2 - x01)/unit_x_2
	
	y_fh2_to_use 	  = 10**(np.abs(y_fh2 - y01)/unit_y_2 - 3.)	

	x_gmc_to_use  = (x_gmc1 - x02)/unit_x_3
	y_gmc_to_use  = 10**(np.abs(y_gmc1 - y02)/unit_y_3)	
				
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

        #ax.plot(radius_center_bins, sigma_h2_1, 'o', markersize = 10, label = r'This Study')
        #ax.plot(radius_center_bins, sigma_h2_2, 'o', label = r'$\Sigma_{H2}$ This Study, 2.')
	ax.plot(radius_center_bins, sigma_h2_6, 'o', markersize = 10, label = r'This Study')
	#ax.plot(radius_center_bins, sigma_h2_10, 'o', label = r'$\Sigma_{H2}$ This Study, 10.')
	#ax.plot(radius_center_bins, sigma_gas, 'o', label=r'$\Sigma_{gas}$')	
	#ax.plot(radius_center_bins, sigma_h2_gas_average, 'ro')
        #ax.plot(radius_center_bins, sigma_h2_no_gas, 'go')
        #ax.plot(x_sigma_h2_to_use, y_sigma_h2_to_use, '*', markersize= 12, label = r'Milville-Deschenes et al. 2017')
	ax.plot(x_gmc_to_use, y_gmc_to_use, '*', markersize = 12, label = r'Milville-Deschenes et al. 2017')

	#ax.plot(radius_center_bins, sigma_h1, 'o', label = r'$\Sigma_{HI}$ This Study')
	#ax.plot(x_sigma_h1_to_use, y_sigma_h1_to_use+y_sigma_h2_to_use, '*', label = r'$\Sigma{gas}$, Milville-Deschenes et al. 2017') 
	ax.set_xlim(0., 20.)
        ax.set_ylim(1.e-3, 1.e2)
        ax.set_yscale('log')
        ax.set_xlabel(r'$R_{\rm g}$[Kpc]')
        ax.set_ylabel(r'$\Sigma_{\rm H_{2}}\ [\rm M_{\odot} pc^{-2} \rm]$')
	ax.legend(frameon=False, loc='best')
        plt.savefig(out_dir+"six_Sigma_H2_vs_R_plot_"+str(snapshot)+"_"+str(redshift)+"_.pdf", bbox_inches="tight")
	ax.cla()

        #ax.plot(radius_center_bins, f_h2_annulus_1, 'o', markersize=10, label='This Study')
        #ax.plot(radius_center_bins, f_h2_annulus_2, 'o', label='2.')
	ax.plot(radius_center_bins, f_h2_annulus_6, 'o', markersize = 10, label='This Study')
	ax.plot(x_fh2_to_use, y_fh2_to_use, '*', markersize=12, label = 'Milville-Deschenes et al. 2017')
	#ax.plot(radius_center_bins, f_h2_mass_average, 'o', label= 'This Study.')
	#ax.plot(radius_center_bins, f_h2_annulus_10, 'o', label='10.')
	
	ax.set_xlim(0., 20.)
        ax.set_ylim(1.e-3, 1.e0)
        ax.set_yscale('log')
        ax.set_xlabel(r'$R_{\rm g}$[Kpc]')
        ax.set_ylabel(r'$\rm f_{\rm H_{2}}$')
        ax.legend(frameon=False, loc='best')
	plt.savefig(out_dir+"six_f_H2_vs_R_plot_"+str(snapshot)+"_"+str(redshift)+"_.pdf", bbox_inches="tight")
        ax.cla()


	plt.close(fig)


'''















