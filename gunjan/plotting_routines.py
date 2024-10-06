import healpy as hp
import numpy as np
import sys
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib import rcParams
import scipy.interpolate
from scipy.optimize import curve_fit
import funcs


M_sun = 2.e33
year = 3.154e7 #year in seconds 
kpc = 3.08e21
k_b = 1.38e-16
gamma = 5./3.
m_proton   = 1.672e-24 # in gms 
omega_m    = 0.272
omega_b    = 0.0455
omega_c    = omega_m - omega_b
sigma_dust = 5.e-22    # cm^(2) g^(-1)]] 
h          = 0.7       # Hubble_parameter
solar_metallicity = 0.02 #solar mass fraction
L_sun = 3.839e33 # L_sun in cgs erg/s 
speed_of_light = 3.e10 #speed of light in cm/s 
epsilon = 8.e-4




plt.rcParams.update({'axes.labelsize': 20, 'font.size': 20,
                    'legend.fontsize': 16, 'xtick.labelsize': 20,
                    'ytick.labelsize': 20,
                    'axes.linewidth': 2,
                    'xtick.labelsize': 18})

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 20}
matplotlib.rc('font', **font)


#change font
plt.rc('text', usetex=True)
plt.rc('font',**{'family':'serif','serif':['Palatino']})


#Load the txt file 
filedir_m12qq_1 = ['/home/lakhlani/Figures/m12qq_hr_Dec16_2013/_newest_L_co_arrays_20-440_.txt']
filedir_m12qq_2 = ['/home/lakhlani/Figures/m12qq_hr_Dec16_2013/second_most_massive_newest_L_co_arrays_20-440_.txt']
filedir_m12qq_3 = ['/home/lakhlani/Figures/m12qq_hr_Dec16_2013/third_most_massive_newest_L_co_arrays_20-440_.txt']

##########
filedir_m12b_1 = ['/home/lakhlani/Figures/m12b_res56000/_newest_L_co_arrays_20-600_.txt']
filedir_m12b_2 = ['/home/lakhlani/Figures/m12b_res56000/second_most_massive_newest_L_co_arrays_20-600_.txt']
filedir_m12b_3 = ['/home/lakhlani/Figures/m12b_res56000/third_most_massive_newest_L_co_arrays_20-600_.txt']
##########
filedir_m13_mr_1 = ['/home/lakhlani/Figures/m13_mr_Dec16_2013/_newest_L_co_arrays_20-440_.txt']
filedir_m13_mr_2 = ['/home/lakhlani/Figures/m13_mr_Dec16_2013/second_most_massive_newest_L_co_arrays_20-440_.txt']
filedir_m13_mr_3 = ['/home/lakhlani/Figures/m13_mr_Dec16_2013/third_most_massive_newest_L_co_arrays_20-440_.txt']

#out_dir = ['/home/lakhlani/Figures/m12qq_hr_Dec16_2013/']
#out_dir = ['/home/lakhlani/Figures/m12b_res56000/']
out_dir = ['/home/lakhlani/Figures/m13_mr_Dec16_2013/']
#DEFINE ON/OFF BUTTONS FOR PLOTS FOR VARIOUS RUNS
PLOT_M12QQ_HR = 1  #1=ON, 0 = OFF
PLOT_M12B = 1 
PLOT_M13_MR = 1
###################################

if PLOT_M12QQ_HR==1: 
	out_dir = ['/home/lakhlani/Figures/m12qq_hr_Dec16_2013/']
	for filein in filedir_m12qq_1: 
        	data        = np.loadtxt(filein)
        	snapnum     = data[:,0]
        	halo_mass   = data[:,2]
        	z           = data[:,1] 
       	 	Lco_desika  = data[:,8]
        	Lco_alpha   = data[:,9]
       		SFR         = data[:,4]
        	gas_mass    = data[:,6]


	for filein in filedir_m12qq_2: 
		data_2  	= np.loadtxt(filein) 
		snapnum_2 	= data_2[:,0]
		halo_mass_2 	= data_2[:,2]
		z_2  		= data_2[:,1]
		SFR_2		= data_2[:,7]
		L_co_desika_2   = data_2[:,13]
		L_co_alpha_2    = data_2[:,17]


	for filein in filedir_3_m12qq_3: 
		data_3 = np.loadtxt(filein) 
		snapnum_3 = data_3[:,0] 
		halo_mass_3     = data_3[:,2]
        	z_3             = data_3[:,1]
        	SFR_3           = data_3[:,7]
        	L_co_desika_3   = data_3[:,13]
        	L_co_alpha_3    = data_3[:,17]

	age_of_universe = funcs.age_of_universe(z, h, omega_m)
	
	'''#LOAD SFR TABLE
	dat_zp1, dat_logm, dat_logsfr, _ = np.loadtxt("/home/lakhlani/Desktop/sfr_release.dat", unpack=True) # Columns are: z+1,logmass,logsfr,logstellarmass  		                                                                       
	dat_logzp1 = np.log10(dat_zp1)
	dat_sfr    = 10.**dat_logsfr

	# Reshape arrays                                                                                                    
	dat_logzp1  = np.unique(dat_logzp1)  # log(z+1), 1D                                                                 
	dat_logm    = np.unique(dat_logm)  # log(Mhalo), 1D                                                                 
	dat_sfr     = np.reshape(dat_sfr, (dat_logm.size, dat_logzp1.size))

	# Get interpolated SFR value(s)                                                                                     
	rbv         = scipy.interpolate.RectBivariateSpline(dat_logm, dat_logzp1, dat_sfr, kx=1, ky=1)


#sfr = rbv.ev(np.log10(halo_mass_12), np.log10(z_12+1))
#sfr_13 = rbv.ev(np.log10(halo_mass), np.log10(z+1))

sfr_13 = rbv.ev(np.log10(halo_mass), np.log10(z+1))
sfr_2  = rbv.ev(np.log10(halo_mass_2), np.log10(z+1))

delta_mf=1.0
alpha=1.37
beta=-1.74
## SFR to LCO ###
                                                                                                  
#ir      = sfr * 1e10 / delta_mf
#lphainv = 1./alpha
#cop     = lir**alphainv * 10**(-beta * alphainv)
#co_t      =  4.9e-5 * lcop

###################
lir_13      = sfr_13 * 1e10 / delta_mf
alphainv = 1./alpha
lcop_13     = lir_13**alphainv * 10**(-beta * alphainv)
Lco_t_13      =  4.9e-5 * lcop_13

lir_2      = sfr_2 * 1.e10 / delta_mf
alphainv = 1./alpha
lcop_2     = lir_2**alphainv * 10**(-beta * alphainv)
Lco_t_2      =  4.9e-5 * lcop_2'''
################################################################

SFR_cgs = SFR*M_sun/year 
SFR_cgs[SFR_cgs == 0.] = 1. 

parameter = Lco_desika*L_sun/(epsilon*SFR_cgs*speed_of_light**2)


'''
fig = plt.figure(figsize=(10,5))
ax = plt.subplot(111)
ax2= ax.twiny()
ax3=ax.twinx()

for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(2)
	ax2.spines[axis].set_linewidth(2)
	ax3.spines[axis].set_linewidth(2)	

#ax.tick_params('both', length=10, width=2, which='major', color='k')
#ax.tick_params('both', length=6, width=1.5, which='minor', color='k')
#ax2.tick_params('both', length=8, width=2, which='major', color='k')
#ax2.tick_params('both', length=6, width=1.5, which='minor', color='k')
#ax3.tick_params('both', length=8, width=2, which='major', color='k')
#ax3.tick_params('both', length=6, width=1.5, which='minor', color='k')

#index_to_plot = np.where((z < 4.0) & (z > 2.0 ))
#index_to_plot = np.where(z < 2.0)
#index_to_plot = np.where((snapnum >=100) & (snapnum <=200))
#index_to_plot = np.where(z < 1000.)
'''
'''
#ax.plot(z[index_to_plot], parameter[index_to_plot]**(-1), 'o', markersize=8)
#ax.plot(z[index_to_plot], Lco_desika[index_to_plot], 'o', markersize=6 , alpha=7, label = r'\rm Desika et al., L$_{CO}$ [L$_{\odot}$]')
#ax.plot(z[index_to_plot], Lco_alpha[index_to_plot],  'o', markersize =6, alpha=7, label = r'($\alpha_{CO}$ =3.17), L$_{CO}$ [L$_{\odot}$]')
#ax.plot(z[index_to_plot], Lco_t_13[index_to_plot] , '-', linewidth=3.0, label = r'Li et al.')
ax.plot(z[index_to_plot], Lco_desika[index_to_plot]+L_co_desika_2[index_to_plot], 'o', markersize=6., alpha=7) 
ax.plot(z[index_to_plot], Lco_alpha[index_to_plot]+L_co_alpha_2[index_to_plot], 'o', markersize=6., alpha=7)
ax.plot(z[index_to_plot], Lco_t_13[index_to_plot]+Lco_t_2[index_to_plot] , '-', linewidth=3.0, label = r'Li et al.')

ax2.plot(age_of_universe[index_to_plot], np.ones(np.size(z[index_to_plot])), 'o', alpha = 0.0)
ax2.invert_xaxis()
ax3.plot(z[index_to_plot][SFR[index_to_plot] > 0.], SFR[index_to_plot][SFR[index_to_plot] > 0.], '*', markersize=6 ,alpha = 5, label = r'Star Formation Rate (10$^{6}M$_\odot$/yr)')
ax3.plot(z[index_to_plot][SFR[index_to_plot] > 0.], SFR[index_to_plot][SFR[index_to_plot] > 0.], '--', linewidth=3, label = r'Star Formation Rate (10$^{6}M$_\odot$/yr)')
ax3.plot(z[index_to_plot][SFR[index_to_plot] > 0.], SFR[index_to_plot][SFR[index_to_plot] > 0.], '*', markersize=6 ,alpha = 5)
ax3.plot(z[index_to_plot][SFR[index_to_plot] > 0.], SFR[index_to_plot][SFR[index_to_plot] > 0.], '--', linewidth=3)
ax3.set_yscale('log') 
#ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel(r'$z$[Redshift]')
#ax.set_ylabel(r'$\epsilon \dot{M}_{\ast} c^{2}/L_{CO}$')
ax.set_ylabel(r'L$_{CO}$ [L$_{\odot}$]')
ax2.set_xlabel('Time (Gyr)')
ax3.set_ylabel('Star Formation Rate (M$_\odot$/yr)')
#ax.set_xlim(1.e-2, 10)
#ax.set_ylim(1.e4, 2.e7)
#ax.legend(frameon = False, loc = 'best')
#ax.legend(frameon = False,bbox_to_anchor=(1.05, 1), loc = 2)
#plt.savefig(out_dir[0]+"trajectories_.pdf", bbox_inches="tight")
plt.savefig(out_dir[0]+"one_plus_two_small_trajectories_.png", bbox_inches="tight")

#plt.savefig(out_dir[0]+"epsilon_plot_.png", bbox_inches="tight")

'''
'''
plt.subplot(211)
plt.plot(z, halo_mass, linewidth=3, label=r'First') 
plt.plot(z, halo_mass_2, linewidth=3, label=r'Second') 
plt.plot(z, halo_mass_3, linewidth=3, label=r'Third') 
plt.yscale('log') 
plt.xlabel('z')
plt.ylabel(r'M$_{Halo}$ [M$_\odot$]') 
plt.legend(frameon= False, loc='best') 
plt.subplot(212) 
plt.plot(z, SFR, 'b', linewidth=3) 
plt.plot(z, SFR_2,'g', linewidth=3) 
plt.plot(z, SFR_3,'r',  linewidth=3)
plt.legend(frameon = False, loc = 'best') 
plt.yscale('log') 
plt.xlabel('z')
plt.ylabel(r'SFR')
plt.savefig('/home/lakhlani/Desktop/m12b_mhalo_SFR_vs_z_plots.pdf', bbox_inches='tight') 
'''

index_to_plot = np.where((z < 4.0) & (z > 2.0 ))

#index_to_plot = np.where(z < 1000.) 

Lco_log = np.log10(Lco_desika) 

A = np.vstack([z[index_to_plot], np.ones(len(z[index_to_plot]))]).T

slope, intercept  = np.linalg.lstsq(A, Lco_log[index_to_plot])[0]

Ltilda = Lco_log[index_to_plot] - slope*z[index_to_plot]-intercept

plt.plot(z[index_to_plot], Lco_log[index_to_plot], 'o')
plt.plot(z[index_to_plot], slope*z[index_to_plot]+intercept, 'o')
plt.xlabel('z [Redshift]')
plt.ylabel(r'log$_{10}$(L$_{CO}$)')
plt.savefig(out_dir[0]+"Lco_best_fit_line_.pdf", bbox_inches="tight")
plt.clf()
plt.hist(Ltilda)
plt.xlabel(r'L$_{CO}$-mean')
plt.ylabel('Number of count')
#plt.yscale('log')
#plt.savefig(out_dir[0]+"Lco_histograms.pdf", bbox_inches="tight")
















