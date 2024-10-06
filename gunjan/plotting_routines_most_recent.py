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
                    'legend.fontsize': 16, 'xtick.labelsize': 13,
                    'ytick.labelsize': 13,
                    'axes.linewidth': 2})
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
##########


#out_dir = ['/home/lakhlani/Figures/m12qq_hr_Dec16_2013/']
#out_dir = ['/home/lakhlani/Figures/m12b_res56000/']
out_dir = ['/home/lakhlani/Figures/m13_mr_Dec16_2013/']
#DEFINE ON/OFF BUTTONS FOR PLOTS FOR VARIOUS RUNS
PLOT_M12QQ_HR = 0  #1=ON, 0 = OFF
PLOT_M12B = 0
PLOT_M13_MR = 0
PLOT_JHU = 1
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
                data_2          = np.loadtxt(filein)
                snapnum_2       = data_2[:,0]
                halo_mass_2     = data_2[:,2]
                z_2             = data_2[:,1]
                SFR_2           = data_2[:,7]
                L_co_desika_2   = data_2[:,13]
                L_co_alpha_2    = data_2[:,17]


        for filein in filedir_m12qq_3:
                data_3 = np.loadtxt(filein)
                snapnum_3 = data_3[:,0]
                halo_mass_3     = data_3[:,2]
                z_3             = data_3[:,1]
                SFR_3           = data_3[:,7]
                L_co_desika_3   = data_3[:,13]
                L_co_alpha_3    = data_3[:,17]

        age_of_universe = funcs.age_of_universe(z, h, omega_m)
	sfr_li_m12_qq_1, lco_li_m12qq_1 = funcs.return_li_sfr_lco(halo_mass, z)

	fig = plt.figure(figsize=(10,5))
	ax = plt.subplot(111)
	ax2= ax.twiny()
	for axis in ['top','bottom','left','right']:
        	ax.spines[axis].set_linewidth(2)
        	ax2.spines[axis].set_linewidth(2)
        	#ax3.spines[axis].set_linewidth(2) 
 
	ax.tick_params('both', length=10, width=2, which='major', color='k')
	ax.tick_params('both', length=6, width=1.5, which='minor', color='k')
	ax2.tick_params('both', length=8, width=2, which='major', color='k')
	ax2.tick_params('both', length=6, width=1.5, which='minor', color='k')
	#ax3.tick_params('both', length=8, width=2, which='major', color='k')
	#ax3.tick_params('both', length=6, width=1.5, which='minor', color='k')

	index_to_plot = np.where((z < 4.0) & (z > 2.0 ))
	#index_to_plot = np.where(z < 2.0)
	#index_to_plot = np.where((snapnum >=100) & (snapnum <=200))
	#index_to_plot = np.where(z < 1000.)

	#ax.plot(z[index_to_plot], parameter[index_to_plot]**(-1), 'o', markersize=8)
	#ax.plot(z[index_to_plot], Lco_desika[index_to_plot], 'o', markersize=6 , alpha=7, label = r'\rm Desika et al.]')
	#ax.plot(z[index_to_plot], Lco_alpha[index_to_plot],  'o', markersize =6, alpha=7, label = r'($\alpha_{CO}$ = 3.17)]')
	#ax.plot(z[index_to_plot], lco_li_m12qq_1[index_to_plot] , '-', linewidth=3.0, label = r'Li et al.')
	ax2.plot(age_of_universe[index_to_plot], np.ones(np.size(z[index_to_plot])), 'o', alpha = 0.0)
	ax2.invert_xaxis()
	ax.plot(z[index_to_plot][SFR[index_to_plot] > 0.], SFR[index_to_plot][SFR[index_to_plot] > 0.], 'o', markersize=6 ,alpha = 5, label = r'FIRE SFR')
        ax.plot(z[index_to_plot], sfr_li_m12_qq_1[index_to_plot], '-', linewidth=3, label = r'Behroozi et al.')
	#ax3.set_yscale('log') 
	#ax.set_xscale('log')
	ax.set_yscale('log')
	ax.set_xlabel(r'$z$[Redshift]')
	#ax.set_ylabel(r'$\epsilon \dot{M}_{\ast} c^{2}/L_{CO}$')
	#ax.set_ylabel(r'L$_{CO}$ [L$_{\odot}$]')
	ax2.set_xlabel('Time (Gyr)')
	ax.set_ylabel('Star Formation Rate (M$_\odot$/yr)')
	#ax.set_xlim(1.e-2, 10)
	#ax.set_ylim(1.e4, 2.e7)
	ax.legend(frameon = False, loc = 'best')
	#ax.legend(frameon = False,bbox_to_anchor=(1.05, 1), loc = 2)
	#plt.savefig(out_dir[0]+"dick_slovenia_lco_m12qq_plots.pdf", bbox_inches="tight")
	plt.savefig(out_dir[0]+'dick_slovenia_m12qq_sfr_plots.pdf', bbox_inches='tight') 







if PLOT_M12B==1:
        out_dir = ['/home/lakhlani/Figures/m12b_res56000/']
        for filein in filedir_m12b_1:
                data        = np.loadtxt(filein)
                snapnum     = data[:,0]
                halo_mass   = data[:,2]
                z           = data[:,1]
                Lco_desika  = data[:,8]
                Lco_alpha   = data[:,9]
                SFR         = data[:,4]
                gas_mass    = data[:,6]


        for filein in filedir_m12b_2:
                data_2          = np.loadtxt(filein)
                snapnum_2       = data_2[:,0]
                halo_mass_2     = data_2[:,2]
                z_2             = data_2[:,1]
                SFR_2           = data_2[:,7]
                L_co_desika_2   = data_2[:,13]
                L_co_alpha_2    = data_2[:,17]


        for filein in filedir_m12b_3:
                data_3 = np.loadtxt(filein)
                snapnum_3 = data_3[:,0]
                halo_mass_3     = data_3[:,2]
                z_3             = data_3[:,1]
                SFR_3           = data_3[:,7]
                L_co_desika_3   = data_3[:,13]
                L_co_alpha_3    = data_3[:,17]

        age_of_universe = funcs.age_of_universe(z, h, omega_m)
        sfr_li_m12b_1, lco_li_m12b_1 = funcs.return_li_sfr_lco(halo_mass, z)

        fig = plt.figure(figsize=(10,5))
        ax = plt.subplot(111)
        ax2= ax.twiny()
        for axis in ['top','bottom','left','right']:
                ax.spines[axis].set_linewidth(2)
                ax2.spines[axis].set_linewidth(2)
                #ax3.spines[axis].set_linewidth(2) 

        ax.tick_params('both', length=10, width=2, which='major', color='k')
        ax.tick_params('both', length=6, width=1.5, which='minor', color='k')
        ax2.tick_params('both', length=8, width=2, which='major', color='k')
        ax2.tick_params('both', length=6, width=1.5, which='minor', color='k')
        #ax3.tick_params('both', length=8, width=2, which='major', color='k')
        #ax3.tick_params('both', length=6, width=1.5, which='minor', color='k')

        index_to_plot = np.where((z < 4.0) & (z > 2.0 ))
        #index_to_plot = np.where(z < 2.0)
        #index_to_plot = np.where((snapnum >=100) & (snapnum <=200))
        #index_to_plot = np.where(z < 1000.)

        #ax.plot(z[index_to_plot], parameter[index_to_plot]**(-1), 'o', markersize=8)
        #ax.plot(z[index_to_plot], Lco_desika[index_to_plot], 'o', markersize=6 , alpha=7, label = r'\rm Desika et al.]')
        #ax.plot(z[index_to_plot], Lco_alpha[index_to_plot],  'o', markersize =6, alpha=7, label = r'($\alpha_{CO}$ = 3.17)]')
        #ax.plot(z[index_to_plot], lco_li_m12b_1[index_to_plot] , '-', linewidth=3.0, label = r'Li et al.')
        ax2.plot(age_of_universe[index_to_plot], np.ones(np.size(z[index_to_plot])), 'o', alpha = 0.0)
        ax2.invert_xaxis()
        #ax3.set_yscale('log') 
        #ax.set_xscale('log')
        ax.plot(z[index_to_plot][SFR[index_to_plot] > 0.], SFR[index_to_plot][SFR[index_to_plot] > 0.], 'o', markersize=6 ,alpha = 5, label = r'FIRE SFR')
        ax.plot(z[index_to_plot], sfr_li_m12b_1[index_to_plot], '-', linewidth=3, label = r'Behroozi et al.')
	ax.set_yscale('log')
        ax.set_xlabel(r'$z$[Redshift]')
        #ax.set_ylabel(r'$\epsilon \dot{M}_{\ast} c^{2}/L_{CO}$')
        #ax.set_ylabel(r'L$_{CO}$ [L$_{\odot}$]')
        ax2.set_xlabel('Time (Gyr)')
        ax.set_ylabel('Star Formation Rate (M$_\odot$/yr)')
        #ax.set_xlim(1.e-2, 10)
        #ax.set_ylim(1.e4, 2.e7)
        ax.legend(frameon = False, loc = 'best')
        #ax.legend(frameon = False,bbox_to_anchor=(1.05, 1), loc = 2)
        #plt.savefig(out_dir[0]+"dick_slovenia_lco_m12b_plots.pdf", bbox_inches="tight")
	plt.savefig(out_dir[0]+"dick_slovenia_m12b_sfr_plots.pdf", bbox_inces='tight') 





if PLOT_M13_MR==1:
        out_dir = ['/home/lakhlani/Figures/m13_mr_Dec16_2013/']
        for filein in filedir_m13_mr_1:
                data        = np.loadtxt(filein)
                snapnum     = data[:,0]
                halo_mass   = data[:,2]
                z           = data[:,1]
                Lco_desika  = data[:,8]
                Lco_alpha   = data[:,9]
                SFR         = data[:,4]
                gas_mass    = data[:,6]


        for filein in filedir_m13_mr_2:
                data_2          = np.loadtxt(filein)
                snapnum_2       = data_2[:,0]
                halo_mass_2     = data_2[:,2]
                z_2             = data_2[:,1]
                SFR_2           = data_2[:,7]
                L_co_desika_2   = data_2[:,13]
                L_co_alpha_2    = data_2[:,17]


        for filein in filedir_m13_mr_3:
                data_3 = np.loadtxt(filein)
                snapnum_3 = data_3[:,0]
                halo_mass_3     = data_3[:,2]
                z_3             = data_3[:,1]
                SFR_3           = data_3[:,7]
                L_co_desika_3   = data_3[:,13]
                L_co_alpha_3    = data_3[:,17]

        age_of_universe = funcs.age_of_universe(z, h, omega_m)
        sfr_li_m13_mr_1, lco_li_m13_mr_1 = funcs.return_li_sfr_lco(halo_mass, z)

        fig = plt.figure(figsize=(10,5))
        ax = plt.subplot(111)
        ax2= ax.twiny()
        for axis in ['top','bottom','left','right']:
                ax.spines[axis].set_linewidth(2)
                ax2.spines[axis].set_linewidth(2)
                #ax3.spines[axis].set_linewidth(2) 

        ax.tick_params('both', length=10, width=2, which='major', color='k')
        ax.tick_params('both', length=6, width=1.5, which='minor', color='k')
        ax2.tick_params('both', length=8, width=2, which='major', color='k')
        ax2.tick_params('both', length=6, width=1.5, which='minor', color='k')
        #ax3.tick_params('both', length=8, width=2, which='major', color='k')
        #ax3.tick_params('both', length=6, width=1.5, which='minor', color='k')

        index_to_plot = np.where((z < 4.0) & (z > 2.0 ))
        #index_to_plot = np.where(z < 2.0)
        #index_to_plot = np.where((snapnum >=100) & (snapnum <=200))
        #index_to_plot = np.where(z < 1000.)

        #ax.plot(z[index_to_plot], parameter[index_to_plot]**(-1), 'o', markersize=8)
        #ax.plot(z[index_to_plot], Lco_desika[index_to_plot], 'o', markersize=6 , alpha=7, label = r'\rm Desika et al.]')
        #ax.plot(z[index_to_plot], Lco_alpha[index_to_plot],  'o', markersize =6, alpha=7, label = r'($\alpha_{CO}$ = 3.17)]')
        #ax.plot(z[index_to_plot], lco_li_m13_mr_1[index_to_plot] , '-', linewidth=3.0, label = r'Li et al.')
        ax2.plot(age_of_universe[index_to_plot], np.ones(np.size(z[index_to_plot])), 'o', alpha = 0.0)
        ax2.invert_xaxis()
        ax.plot(z[index_to_plot][SFR[index_to_plot] > 0.], SFR[index_to_plot][SFR[index_to_plot] > 0.], 'o', markersize=6 ,alpha = 5, label = r'FIRE SFR')
        ax.plot(z[index_to_plot], sfr_li_m13_mr_1[index_to_plot], '-', linewidth=3, label = r'Behroozi et al.')
        #ax3.plot(z[index_to_plot][SFR[index_to_plot] > 0.], SFR[index_to_plot][SFR[index_to_plot] > 0.], '*', markersize=6 ,alpha = 5)
        #ax3.plot(z[index_to_plot][SFR[index_to_plot] > 0.], SFR[index_to_plot][SFR[index_to_plot] > 0.], '--', linewidth=3)
        #ax3.set_yscale('log') 
        #ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(r'$z$[Redshift]')
        #ax.set_ylabel(r'$\epsilon \dot{M}_{\ast} c^{2}/L_{CO}$')
        #ax.set_ylabel(r'L$_{CO}$ [L$_{\odot}$]')
        ax2.set_xlabel('Time (Gyr)')
        ax.set_ylabel('Star Formation Rate (M$_\odot$/yr)')
        #ax.set_xlim(1.e-2, 10)
        #ax.set_ylim(1.e4, 2.e7)
        ax.legend(frameon = False, loc = 'best')
        #ax.legend(frameon = False,bbox_to_anchor=(1.05, 1), loc = 2)
        #plt.savefig(out_dir[0]+"dick_slovenia_lco_m13_mr_plots.pdf", bbox_inches="tight")
	plt.savefig(out_dir[0]+"dick_slovenia_m13_sfr_plots.pdf", bbox_inches="tight")


if PLOT_JHU==1: 
	for filein in filedir_m12qq_1:
                data_m12qq_1        = np.loadtxt(filein)
                snapnum_m12qq_1     = data_m12qq_1[:,0]
                halo_mass_m12qq_1   = data_m12qq_1[:,2]
                z_m12qq_1           = data_m12qq_1[:,1]
                Lco_desika_m12qq_1  = data_m12qq_1[:,8]
                Lco_alpha_m12qq_1   = data_m12qq_1[:,9]
                SFR_m12qq_1         = data_m12qq_1[:,4]
                gas_mass_1    	    = data_m12qq_1[:,6]

	for filein in filedir_m12b_1:
                data_m12b_1        = np.loadtxt(filein)
                snapnum_m12b_1     = data_m12b_1[:,0]
                halo_mass_m12b_1   = data_m12b_1[:,2]
                z_m12b_1           = data_m12b_1[:,1]
                Lco_desika_m12b_1  = data_m12b_1[:,8]
                Lco_alpha_m12b_1   = data_m12b_1[:,9]
                SFR_m12b_1         = data_m12b_1[:,4]
                gas_mass_m12b_1    = data_m12b_1[:,6]

	
	for filein in filedir_m13_mr_1:
                data_m13_mr_1        = np.loadtxt(filein)
                snapnum_m13_mr_1     = data_m13_mr_1[:,0]
                halo_mass_m13_mr_1   = data_m13_mr_1[:,2]
                z_m13_mr_1           = data_m13_mr_1[:,1]
                Lco_desika_m13_mr_1  = data_m13_mr_1[:,8]
                Lco_alpha_m13_mr_1   = data_m13_mr_1[:,9]
                SFR_m13_mr_1         = data_m13_mr_1[:,4]
                gas_mass_m13_mr_1    = data_m13_mr_1[:,6]

	age_of_universe = funcs.age_of_universe(z_m12qq_1, h, omega_m)
        sfr_li_m12_qq_1, lco_li_m12qq_1 = funcs.return_li_sfr_lco(halo_mass_m12qq_1, z_m12qq_1)
	sfr_li_m12b_1, lco_li_m12b_1 = funcs.return_li_sfr_lco(halo_mass_m12b_1, z_m12b_1)
	sfr_li_m13_mr_1, lco_li_m13_mr_1 = funcs.return_li_sfr_lco(halo_mass_m13_mr_1, z_m13_mr_1)

	index_to_plot_m12qq = np.where((z_m12qq_1 <= 4.0) & (z_m12qq_1 >= 2.0 ))
	index_to_plot_m12b  = np.where((z_m12b_1 <= 4.0 ) & (z_m12b_1  >= 2.0))
	index_to_plot_m13_mr = index_to_plot_m12qq

	f, (ax1, ax2, ax3) = plt.subplots(3, sharex=True) 
	ax4 = ax1.twiny()

	for axis in ['top','bottom','left','right']:
                ax1.spines[axis].set_linewidth(2)
                ax2.spines[axis].set_linewidth(2)
                ax3.spines[axis].set_linewidth(2) 
		ax4.spines[axis].set_linewidth(2)
        ax1.tick_params('both', length=8, width=2, which='major', color='k')
        ax1.tick_params('both', length=6, width=1.5, which='minor', color='k')
        ax2.tick_params('both', length=8, width=2, which='major', color='k')
        ax2.tick_params('both', length=6, width=1.5, which='minor', color='k')
	ax3.tick_params('both', length=8, width=2, which='major', color='k')
        ax3.tick_params('both', length=6, width=1.5, which='minor', color='k')
	ax4.tick_params('both', length=8, width=2, which='major', color='k')
        ax4.tick_params('both', length=6, width=1.5, which='minor', color='k')

	ax1.plot(z_m12qq_1[index_to_plot_m12qq], np.log10(lco_li_m12qq_1[index_to_plot_m12qq]), linewidth=2, label='Li et al.')
	ax1.plot(z_m12qq_1[index_to_plot_m12qq], np.log10(Lco_desika_m12qq_1[index_to_plot_m12qq]), 'o', label='This Work') 
	ax1.set_xlabel('m12qq, FIRE-1', fontsize=10)	
	ax1.set_ylim(2, 6)
	
	ax2.plot(z_m12b_1[index_to_plot_m12b], np.log10(lco_li_m12b_1[index_to_plot_m12b]), linewidth=2)
        ax2.plot(z_m12b_1[index_to_plot_m12b], np.log10(Lco_desika_m12b_1[index_to_plot_m12b]), 'o')
	ax2.set_xlabel('m12b, FIRE-2', fontsize=10)
	ax2.set_ylim(2, 6)	

	ax3.plot(z_m13_mr_1[index_to_plot_m13_mr], np.log10(lco_li_m13_mr_1[index_to_plot_m13_mr]), linewidth=2)
        ax3.plot(z_m13_mr_1[index_to_plot_m13_mr], np.log10(Lco_desika_m13_mr_1[index_to_plot_m13_mr]), 'o')
	ax3.set_xlabel('m13, FIRE-1', fontsize=10)
	ax3.set_ylim(3.5, 7)	

	ax4.plot(age_of_universe[index_to_plot_m12qq], np.ones(np.size(z_m12qq_1[index_to_plot_m12qq])), 'o', alpha = 0.0)
	ax4.set_xlabel('Time (Gyr)')

	ax1.legend(frameon = False,bbox_to_anchor=(1., 1), loc = 2) 
	f.text(0.2, 0.02, 'z [Redshift]', ha='center') 
	f.text(0.04, 0.5, r'\rm log$_{10}$(\rm $\frac{L_{CO}}{L_{\odot}}$)', va='center', rotation='vertical') 
	
	plt.savefig('/home/lakhlani/Figures/JHU_summary_paper.png', bbox_inches='tight')








	



