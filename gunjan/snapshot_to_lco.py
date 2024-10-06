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


'''plt.rcParams.update({'axes.labelsize': 20, 'font.size': 20,
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
'''
#####################################################

snapshot_i = 440
snapshot_f = 440
nsteps = 1 #SPECIFY THE LOOP STEP FOR SNAPSHOTS

#file_dir = '/mnt/scratch-lustre/lakhlani/m12b_res56000/output/' 

#file_dir = '/mnt/scratch-lustre/lakhlani/m13_mr_Dec16_2013/'

file_dir = '/mnt/scratch-lustre/lakhlani/m12qq_hr_Dec16_2013/'

#out_dir = '/home/lakhlani/Figures/m12b_res56000/'

out_dir = '/home/lakhlani/Figures/m12qq_hr_Dec16_2013/'

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

        xc, yc, zc, Mhalo = funcs.find_mostmassive(file_dir, snapshot)

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


        Total_gas_mass_fraction = np.sum(gas_mass)/(np.sum(gas_mass) + np.sum(stellar_mass))    # Gunjan did not use it - dtolgay

        # Calculating the theta and phi using the integrated angular momentums of gas particles - dtolgay 
        theta, phi = funcs.angular_momentum(R_index, gas_mass, x_gas, y_gas, z_gas, vx_gas, vy_gas, vz_gas)

        # QQ1 Why do we need to transform the coordinate frame?
        # New coordinates after the rotation of gas particles. In here gas particles are considered as a solid, and the angle is calculated
        # using the above formula. After calculating the angle the new x, y and z coordinates are calculated using geometry.  - dtolgay 
        x_gas, y_gas, z_gas = funcs.rotation(theta, phi, x_gas, y_gas, z_gas)

        vx_gas, vy_gas, vz_gas = funcs.rotation(theta, phi, vx_gas, vy_gas, vz_gas)

        theta_new, phi_new = funcs.angular_momentum(R_index, gas_mass, x_gas, y_gas, z_gas, vx_gas, vy_gas, vz_gas) # It is never used - dtolgay

        x_gas, y_gas, z_gas = x_gas[index_gas], y_gas[index_gas], z_gas[index_gas]
        gas_mass, smoothing_gas, gas_density = gas_mass[index_gas], smoothing_gas[index_gas], gas_density[index_gas]
        R_gas = R_gas[index_gas]
        total_metallicity = total_metallicity[index_gas]

##########################################################################

	exception_indices = np.zeros(0)
        shielding_length = np.zeros(0)
        shielding_length_phil = np.zeros(0)
        shielding_length_index = np.size(gas_density)   # Size of the gas_density vector - dtolgay
        density_local = np.zeros(0)
        nearest_neighbours = 64
        pos = np.zeros((shielding_length_index,3))
        pos[:,0] = x_gas
        pos[:,1] = y_gas
        pos[:,2] = z_gas
        posTree  = cKDTree(pos)  # from scipy.spatial import * - dtolgay

	i_start = 0

        for i in range(i_start, shielding_length_index):

                try:
                        # Below is all done to construct gradient and find shielding_length

                        center_particle_index = i
                        center_particle_position = pos[i,:]
                        center_particle_density  = gas_density[i]
                        center_particle_smoothing = smoothing_gas[i]
                        test0   = posTree.query(center_particle_position,k=nearest_neighbours) # scipy.spatial library - This is code determines the
                        # the distances and indices of the closest 64 particles - dtolgay 
                        # test0[1][:] indices of the nearest neighboor

                        xj = pos[ test0[1][:], 0]       # xj is the x position of the closest nearest neighbourhood (NN)
                        yj = pos[ test0[1][:], 1]       # yj is the y position of the closest nearest neighbourhood (NN)
                        zj = pos[ test0[1][:], 2]       # zj is the z position of the closest nearest neighbourhood (NN)
                        densj    = gas_density[ test0[1][:] ]           # gas density of closest NN  
                        smoothj  = smoothing_gas[ test0[1][:] ]         # smoothj is the smooting_gas of the closest NNs

                        dx0 = xj - center_particle_position[0]  # distance in x direction between closest NN and center particle
                        dy0 = yj - center_particle_position[1]  # distance in y direction between closest NN and center particle
                        dz0 = zj - center_particle_position[2]  # distance in z direction between closest NN and center particle

                        dr0 = test0[0][:] # The magnitude of the distance between closest NN and center_particle
	
			ddens0 = densj - center_particle_density  # The difference between the density of closest NN and center particle

                        grad0  = funcs.construct_gradient(ddens0, dx0, dy0, dz0, dr0, smoothj) # Density gradient is calculated

                        grad_scalar = np.sqrt(grad0[0]**2 + grad0[1]**2 + grad0[2]**2)  # Magnitude of density gradient

                        shielding_length_temp = (center_particle_density/grad_scalar) + smoothing_gas[i]

                        shielding_length = np.append(shielding_length, shielding_length_temp) # Shielding length is stored 
                        # for each center particle
                        # A comparison of methods for determining the molecular content of model galaxies Gnedin 2011. Since H2 destruction rate
# depends on the amount of shielding, one must estimate a column density for each cell. Local density scale height h = rho/|grad_rho|, and then
# takes the column desnity to be = rho*h.


			#massi = np.sum(gas_mass[test0[1][:]]*funcs.quintic_spline(dr0, smoothj))
			
			densityi = 0.
			for k in range(0, nearest_neighbours):
				densityi += gas_mass[test0[1][k]]*funcs.quintic_spline(dr0[k], smoothj[k])
                        # In here gas_mass is the mass of the particles that satisfy R < R_index. Again in these particles the 64 closest nearest
                        # neighboorhood particles are selected, and the resulting density for these 64 particles are calculated
	
			density_local = np.append(density_local, densityi) 	# density i corresponds to the density of the 64 closest NNs particles
                        # to center_particle. Then it is appended to the density_local vector to store the local density for all center particles.
                        
			#print dr0[1], smootj[1]

			#print funcs.quintic_spline(dr0, smoothj)
			
			#massj = np.sum(gas_mass[test0[1]]*funcs.quintic_spline(dr0, smoothj))
			
			#print massj 
                        #mass_local = np.append(mass_local, massj)
	

		except:
                        exception_indices = np.append(exception_indices, i)


		pass
        # Deleting the rows corresponding to exception indices
        gas_density = np.delete(gas_density, exception_indices)
        gas_mass    = np.delete(gas_mass, exception_indices)
        total_metallicity = np.delete(total_metallicity, exception_indices)
        
	smoothing_gas = np.delete(smoothing_gas, exception_indices)
        R_gas = np.delete(R_gas, exception_indices)

#####################################################################

	n_bins = 2*R_index + 1.

        # Creating an array that starts from 0, and ends at R_index, that has nbins elements
        radius_bins = np.linspace(0., R_index, n_bins)

        radius_center_bins = np.linspace(0, R_index, n_bins-1)

        annulus_area = np.pi*(radius_bins[1:]**2 - radius_bins[:-1]**2)


        digitized_gas = np.digitize(R_gas, radius_bins)

        gas_mass_digitized = np.asarray([gas_mass[digitized_gas == i].sum() for i in range(1, len(radius_bins))]) # With the above code mass 
        # of halos are categorized by looking their displacement (R_gas). With this code, the masses that are in the same category is 
        # summed 

        sigma_gas = np.asarray([gas_mass[digitized_gas == i].sum()/annulus_area[i-1] for i in range(1, len(radius_bins))])*1.e10/1.e6
        # sigma_gas is the local GAS column density or dust cross section per  H nucleus. Gunjan Thesis paragraph between 4.3 and 4.4. Is 
        # 1.e10/1.e6 a smooting length?

 ####################################################################################################
        Z_prime = total_metallicity/solar_metallicity
        # Metallicity normalized to Milky way metallicity

	local_volume = gas_mass/density_local  # Finding the local volume of the gas particles in the same category
	local_length = local_volume**(1./3.)   # Volume element was a square, therefore length is vol^1/3
	local_column_density = 4./3.*density_local*local_length # QQ Isn't the local column density is calculated with the parameter
        # sigma_gas? What is the difference between sigma_gas and local_column_density? 
        
	tau_c = 2.*local_column_density*1.e3*Z_prime*M_sun*1.e10/(kpc**2.)
        # Multipliying with 2 is due to clumping factor. What is clumping factor? 
        
        chi               = 3.1*(1.+3.1*Z_prime**0.365)/4.1

        s                 = np.log(1.+0.6*chi+0.01*chi**2)/(0.6*tau_c)


        h2_fraction       = 1. - (3.*s)/(4.*(1.+ 0.25*s))


        h2_fraction[h2_fraction < 0. ]  = 0.


        mass_h2           =  gas_mass*h2_fraction


        total_h2_mass = np.sum(mass_h2)


	gas_mass_total = np.sum(gas_mass)  #gas mass in R_index kpc from the center (in 10^10 M_sun)

        h2_mass_fraction = total_h2_mass/ gas_mass_total

        SFR_total = np.sum(SFR) #SFR in R_index kpc from the center (in units of M_sun/year )  

        h2_mass_digitized = np.asarray([mass_h2[digitized_gas == i].sum() for i in range(1, len(radius_bins))])

        sigma_h2 = np.asarray([mass_h2[digitized_gas == i].sum()/annulus_area[i-1] for i in range(1, len(radius_bins))])*1.e10/1.e6

        sigma_h2_particles = mass_h2/(np.pi*smoothing_gas**2.)*1.e4

        sigma_h2_gas_average = np.asarray([np.sum(gas_mass[digitized_gas == i]*sigma_h2_particles[digitized_gas == i])/np.sum(gas_mass[digitized_gas == i]) for i in range(1, len(radius_bins))])

        sigma_h2_no_gas = np.asarray([np.sum(sigma_h2_particles[digitized_gas == i])/sigma_h2_particles[digitized_gas == i].size for i in range(1, len(radius_bins))])


        f_h2_annulus = sigma_h2/sigma_gas

        #CONVERT REDSHIFT TO REQUIRED PRECISION 
        redshift = '%1.3f' %redshift



	fig = plt.figure(figsize=(10,5))
        ax = plt.subplot(111)
        for axis in ['top','bottom','left','right']:
                ax.spines[axis].set_linewidth(2)

        ax.tick_params('both', length=10, width=2, which='major', color='k')
        ax.tick_params('both', length=6, width=1.5, which='minor', color='k')

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

	ax.plot(radius_center_bins, sigma_h2, 'o') 
        ax.plot(radius_center_bins, sigma_h2_gas_average, 'ro')
	ax.plot(radius_center_bins, sigma_h2_no_gas, 'go')
	ax.set_xlim(0., 20.)
        ax.set_ylim(1.e-3, 1.e2)
        ax.set_yscale('log')
        ax.set_xlabel(r'$R_{\rm g}$[Kpc]')
        ax.set_ylabel(r'$\Sigma_{\rm H_{2}}\ [\rm M_{\odot} pc^{-2} \rm]$')
        plt.savefig(out_dir+"Sigma_H2_vs_R_plot_"+str(snapshot)+"_"+str(redshift)+"_.pdf", bbox_inches="tight")
        ax.cla()        

        ax.plot(radius_center_bins, f_h2_annulus, 'o')
        ax.set_xlim(0., 20.)
        ax.set_ylim(1.e-3, 1.e0)
        ax.set_yscale('log')
        ax.set_xlabel(r'$R_{\rm g}$[Kpc]')
        ax.set_ylabel(r'$\rm f_{\rm H_{2}}$')
        plt.savefig(out_dir+"f_H2_vs_R_plot_"+str(snapshot)+"_"+str(redshift)+"_.pdf", bbox_inches="tight")
        ax.cla()




