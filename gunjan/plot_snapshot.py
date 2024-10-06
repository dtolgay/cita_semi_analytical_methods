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

snapshot_i = 100 #135 #430
snapshot_f = 200 #440
nsteps     = 5 #SPECIFY THE LOOP STEP FOR SNAPSHOTS

#file_dir   = '/mnt/scratch-lustre/lakhlani/m12qq_hr_Dec16_2013/'
#out_dir    = 'Images/m12qq_hr_Dec16_2013'
#file_dir   = '/mnt/scratch-lustre/lakhlani/m12b_res56000/output/'
#out_dir    = 'Images/m12b/m12b_res56000'

file_dir = '/mnt/scratch-lustre/lakhlani/m12b_res56000/output/' 
#file_dir = '/mnt/scratch-lustre/lakhlani/m13_mr_Dec16_2013/'

#out_dir = '/mnt/scratch-lustre/lakhlani/dick_slovenia/m12qq_hr_Dec16_2013/'
out_dir = '/mnt/scratch-lustre/lakhlani/dick_slovenia/m12b_res56000/'
#out_dir = '/mnt/scratch-lustre/lakhlani/dick_slovenia/m13_mr_Dec16_2013/'

plt.rcParams.update({'axes.labelsize': 20, 'font.size': 20,
                    'legend.fontsize': 16, 'xtick.labelsize': 20, 
                    'ytick.labelsize': 20,
                    'axes.linewidth': 6,
                    'xtick.labelsize': 20})
    
font = {'family' : 'normal',
        'weight' : 'normal', 
        'size'   : 20}
matplotlib.rc('font', **font)

plt.rc('text', usetex=True)
plt.rc('font',**{'family':'serif','serif':['Palatino']})

xcent = np.zeros(0)
ycent = np.zeros(0)
zcent = np.zeros(0)
snapcent = np.zeros(0)

#Added polynomial fit to centre of halo. Should give smoother movie
'''polyfits = np.loadtxt("polyfit_xyz_start135_m12b.txt")
xpoly    = polyfits[:,0]
pxpoly   = np.poly1d(xpoly) 
ypoly    = polyfits[:,1]
pypoly   = np.poly1d(ypoly) 
zpoly    = polyfits[:,2]
pzpoly   = np.poly1d(zpoly) 
'''
for snapshot in range(snapshot_i, snapshot_f+1, nsteps):

	#xc =  pxpoly(snapshot)
	#yc =  pypoly(snapshot)
	#zc =  pzpoly(snapshot)
	xc, yc, zc, Mhalo = funcs.find_mostmassive(file_dir, snapshot, 0)
#	xcent = np.append(xcent,xc)
#	ycent = np.append(ycent,yc)
#	zcent = np.append(zcent,zc)
#	snapcent = np.append(snapcent,snapshot)
	print "xc, yc, zc, shapshot = ", xc, yc, zc, snapshot

#np.savetxt("halocenter_m12b.txt", np.c_[snapcent,xcent,ycent,zcent])

        print "\n\nrunning on snapshot ", snapshot
	snapshot    = '%03d' %snapshot
        P           = readsnap.readsnap(file_dir, snapshot, 0, cosmological=1, loud=1 )
        P_star      = readsnap.readsnap(file_dir, snapshot, 4, cosmological=1)
        P_dm        = readsnap.readsnap(file_dir, snapshot, 1, cosmological=1)

        header_info  = readsnap.readsnap(file_dir, snapshot, 0 ,cosmological=1, header_only=1)
        scale_factor = header_info['time']
        hubble_run   = header_info['hubble']
        redshift     = header_info['redshift']
	print "scale_factor, redshift = ", scale_factor, redshift 

        ## p stands for the position. Check the readsnap.py
        x_gas,  y_gas, z_gas    = P['p'][:,0], P['p'][:,1], P['p'][:,2]
        x_star, y_star, z_star  = P_star['p'][:,0], P_star['p'][:,1], P_star['p'][:,2]
        x_dm,   y_dm, z_dm      = P_dm['p'][:,0], P_dm['p'][:,1], P_dm['p'][:,2]

        ## m stands for the mass. Check the readsnap.py
        gas_mass           = P['m']
        stellar_mass       = P_star['m']
        dm_mass            = P_dm['m']
        gas_density        = P['rho']
        smoothing_gas      = P['h']
        internal_energy    = P['u']
        electron_abundance = P['ne']
        SFR                = P['sfr']

        total_metallicity    = P['z'][:,0]
        helium_mass_fraction = P['z'][:,1]
        vx_gas, vy_gas, vz_gas = P['v'][:,0], P['v'][:,1], P['v'][:,2]

        # funcs.change_coordinates(xc,yc,zc,x,y,z) => xnew = xc - x ... 
        x_dm, y_dm, z_dm       = funcs.change_coordinates(xc/hubble_run*scale_factor, yc/hubble_run*scale_factor, zc/hubble_run*scale_factor, x_dm, y_dm, z_dm)
        x_gas, y_gas, z_gas    = funcs.change_coordinates(xc/hubble_run*scale_factor, yc/hubble_run*scale_factor, zc/hubble_run*scale_factor, x_gas, y_gas, z_gas)
        x_star, y_star, z_star = funcs.change_coordinates(xc/hubble_run*scale_factor, yc/hubble_run*scale_factor, zc/hubble_run*scale_factor, x_star, y_star, z_star)


	#CUT PARTICLES

	R_index = 30. #The max distance from the galactic center to which we keep the particles, done to speed up the calculations.

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
#	if int(snapshot) == snapshot_i:
#		theta, phi              = funcs.angular_momentum(R_index, gas_mass, x_gas, y_gas, z_gas, vx_gas, vy_gas, vz_gas)
	theta, phi              = funcs.angular_momentum(R_index, gas_mass, x_gas, y_gas, z_gas, vx_gas, vy_gas, vz_gas)
	print theta, phi

#m12qq
#	theta = 1.81839448053 #hardwired to snapshot 440
#	phi   = 1.15777652324 

#m12b
#	theta = 2.98947201738 
#	phi   = -0.28195511555

	x_gas, y_gas, z_gas     = funcs.rotation(theta, phi, x_gas, y_gas, z_gas)
        vx_gas, vy_gas, vz_gas  = funcs.rotation(theta, phi, vx_gas, vy_gas, vz_gas)

        x_dm, y_dm, z_dm        = funcs.rotation(theta, phi, x_dm, y_dm, z_dm)
        x_star, y_star, z_star  = funcs.rotation(theta, phi, x_star, y_star, z_star)

        theta_new, phi_new      = funcs.angular_momentum(R_index, gas_mass, x_gas, y_gas, z_gas, vx_gas, vy_gas, vz_gas)

	Z_prime 		= total_metallicity/solar_metallicity 

	h2_mass 		= gas_mass*funcs.return_h2_fraction(gas_density, smoothing_gas, Z_prime, 6.)  
##########################################################################

	dm_binned, xedges, yedges, binnumber = binned_statistic_2d(x_dm,y_dm,dm_mass*0.+1,statistic='sum',bins=1024,range=[[-R_index,R_index],[-R_index,R_index]])
	gas_binned, xedges, yedges, binnumber = binned_statistic_2d(x_gas,y_gas,gas_mass,statistic='sum',bins=1024,range=[[-R_index,R_index],[-R_index,R_index]])

	h2_mass_binned, xedges, yedges, binnumber = binned_statistic_2d(x_gas, y_gas, h2_mass, statistic='sum', bins=1024, range=[[-R_index,R_index],[-R_index,R_index]])


	np.savetxt(out_dir+"_60kpc_gas_binned_"+str(snapshot)+".txt", h2_mass_binned)

	h2_mass_binned = gaussian_filter(h2_mass_binned,sigma=1.5)
#	if int(snapshot) == snapshot_i:
#		vmin_gas = np.log10(np.min(h2_mass_binned[h2_mass_binned>np.min(h2_mass_binned)])*1e4)
#		vmax_gas = np.log10(np.max(h2_mass_binned))

	line = np.linspace(-28, -24, 1000) 	

	vmin_gas = -6.
	vmax_gas = -3.75
	redshift = '%1.1f' %redshift

	fig = plt.figure(figsize=(8,8))
	ax  = plt.subplot(111)
	for axis in ['top','bottom','left','right']:
		ax.spines[axis].set_linewidth(2)	
	ax.tick_params('both', length=6, width=2, which='major', color='k')
	ax.tick_params('both', length=3, width=1, which='minor', color='k')

	ax.imshow(np.log10(h2_mass_binned+1e-16) ,vmin=vmin_gas, vmax=vmax_gas, interpolation='spline16',origin='lower',extent=[-R_index,R_index,-R_index,R_index], cmap=plt.get_cmap('inferno'))

	ax.plot(line, line*0+25, 'w', linewidth=3)

	#ax.set_xlabel(r'x [kpc]')
	#ax.set_ylabel(r'y [kpc]')
	
	ax.text(18,25, r"z = "+redshift,color="w", fontsize=30)
	ax.text(-28, 22, r'5 Kpc', color='w', fontsize=20)
	
	plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
			hspace = 0, wspace = 0)
	plt.margins(0,0)
	plt.gca().xaxis.set_major_locator(plt.NullLocator())
	plt.gca().yaxis.set_major_locator(plt.NullLocator())
	plt.savefig(out_dir+"_h2_imshow_"+str(snapshot)+".png", bbox_inches="tight",pad_inches=0)



