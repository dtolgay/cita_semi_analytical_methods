# Imported modules
import sys
sys.path.append("/mnt/raid-cita/dtolgay/FIRE/post_processing_fire_outputs")
from tools import constants

import pandas as pd 
import numpy as np 

import sys # To run the batch script

import os # To check if file exists.

def main(
    galaxy_name,
    galaxy_type,
    redshift,
    verbose = False,    
    ):


    print(f"-------------------------------------------------------------- {galaxy_name} --------------------------------------------------------------")

    # Read cloudy_gas_txt 
    run_dir = "voronoi_1e6"
    cloudy_gas_particles_file_directory = f"/mnt/raid-cita/dtolgay/FIRE/post_processing_fire_outputs/skirt/runs_hden_radius/{galaxy_type}/z{redshift}/{galaxy_name}/{run_dir}" 


    # Define the column names based on your description
    gas_column_names = [
        "x", 
        "y", 
        "z", 
        "smoothing_length", 
        "mass", 
        "metallicity", # Zsolar but will be converted to mass fraction in the future
        "temperature", 
        "vx", 
        "vy", 
        "vz", 
        "hden", 
        "radius", 
        "sfr", 
        "turbulence", 
        "density", 
        "mu_theoretical", 
        "average_sobolev_smoothingLength",
        "index", 
        "isrf"
    ]


    gas_particles_df = pd.read_csv(
        f"{cloudy_gas_particles_file_directory}/cloudy_gas_particles.txt",
        delim_whitespace=True, 
        comment='#', 
        names=gas_column_names
    )


    # Converting density back to 1e10 Msolar / kpc^3 from gr/cm^3
    # gas_particles_df["density"] *= (1 / (1e10 * constants.M_sun2gr)) * (constants.kpc2cm)**3  # 1e10 Msolar / kpc^3 
    # gas_particles_df["density"] /= (1e10 * constants.M_sun2gr / (constants.kpc2cm)**3)  # 1e10 Msolar / kpc^3


    print(f"{cloudy_gas_particles_file_directory}/cloudy_gas_particles.txt read and dataframe is created!")


    ############################################################################################################################################

    # h2 mass fraction is calculated by following Krumholz, and Gnedin (2011) 
    # "A Comparison of Methods for Determining the Molecular Content of Model Galaxies by Krumholz, and Gnedin (2011)"
    
    runs = {
        "cf_1": {
            "df": pd.DataFrame(),
            "clumping_factor": 1,
            "write_file_path": f"{cloudy_gas_particles_file_directory}/semi_analytical_averageSobolevH_cf_1.txt"
        },
        "cf_2": {
            "df": pd.DataFrame(),
            "clumping_factor": 2,
            "write_file_path": f"{cloudy_gas_particles_file_directory}/semi_analytical_averageSobolevH_cf_2.txt"
        },  
        "cf_10": {
            "df": pd.DataFrame(),
            "clumping_factor": 10,
            "write_file_path": f"{cloudy_gas_particles_file_directory}/semi_analytical_averageSobolevH_cf_10.txt"
        },
        "cf_100": {
            "df": pd.DataFrame(),
            "clumping_factor": 100,
            "write_file_path": f"{cloudy_gas_particles_file_directory}/semi_analytical_averageSobolevH_cf_100.txt"
        },
        "cf_500": {
            "df": pd.DataFrame(),
            "clumping_factor": 500,
            "write_file_path": f"{cloudy_gas_particles_file_directory}/semi_analytical_averageSobolevH_cf_500.txt"
        },    
        "cf_functionOfTurbulence" : {
            "df": pd.DataFrame(),
            "clumping_factor": clumping_factor_from_turbulence_velocity(gas_particles_df['turbulence']),
            "write_file_path": f"{cloudy_gas_particles_file_directory}/semi_analytical_averageSobolevH_cf_functionOfTurbulence_.txt"
        },                                      
    }    
    
    columns2write = [
        "index",
        "fh2",
        "Mh2",
        "tau_c",
        "alfa_co",
        "L_co"
    ]
    
    header2write = """
    Column 1: index [1],
    Column 2: fh2 [1],
    Column 3: Mh2 [Msolar],
    Column 4: dust_optical_depth [1],
    Column 5: alfa_co [Msolar / (K-km s^-1 pc^2)],
    Column 6: L_co [K-km s^-1 pc^2]
    """
    
    
    for key in list(runs.keys()):
        # Check if file exists. If does don't run the code 
        if not does_file_exist(runs[key]["write_file_path"]):        
            if (verbose): print(f"I am doing for the {key}")
            runs[key]["df"] = calculate_properties_of_gas_particles(
                gas_df = gas_particles_df.copy(),
                clumping_factor = runs[key]["clumping_factor"]
            )
            if (verbose): print("\n\n")
            
            # Write dataframe into a file 
            np.savetxt(
                fname=runs[key]["write_file_path"],
                X = runs[key]["df"][columns2write],
                fmt='%.18e', 
                header=header2write
            )
            
            if (verbose): print(f"Written to {runs[key]['write_file_path']}")
            
        else:
            if (verbose): print(f"{runs[key]['write_file_path']} exists. I am not computing anything for this run.")
            pass
    

    return runs


################ Functions

################ Functions

def h2_mass_fraction_calculator(
    local_density_scale_height, 
    density, 
    metallicity, 
    clumping_factor
):

    print("I am in the function h2_mass_fraction_calculator") 


    """This function is used to calculate the H2 mass fraction by using the formula 1 in the paper:
    A Comparison of Methods for Determining the Molecular Content of Model Galaxies by Krumholz, and Gnedin (2011)

    Arguments:
    ----------
    local_density_scale_height: array_like
        In this equation smooting length of the gas is assumed to be accurate estimation of the local density scale height. 
        Therefore smooting length is used instead of local density scale height
        [pc]

    density: array_like
        Density of the gas particles
        [gr / cm^3]

    metallicity: array_like
        metallicity of the gas particles 
        [Zsolar]

    clumping_factor: double or int or array-like
        It is a parameter to boost the h2 mass fraction and therefore h2 column density and CO luminosity
        [unitless]

    Returns:
    ----------
    h2_mass_fraction: array_like
        h2_mass_fraction = h2_gas_mass / total_gas_mass
        [unitless]

    column_density: array_like
        It is the column density considering all elements in the gas particle
        [gr/cm^2]

    dust_optical_depth: array_like  
        tau_c in the reference paper. It was being output in order to control the code
        [unitless]

    References: 
    -----------
    A Comparison of Methods for Determining the Molecular Content of Model Galaxies by Krumholz, and Gnedin (2011)

    """ 


    # The units of local_density_scale_height is kpc 
    local_density_scale_height = local_density_scale_height * constants.pc2cm  # [cm]

    # Calculation of column density 
    column_density = density * local_density_scale_height   # [gr / cm^2]
    # Column density is the summation sign in the paper
    # Gunjan assumed that local density scale height is same with the smoooting length of the gas particles. I will continue on this assumption
    # but I don't know how it works

    # Calculation of dust cross section (sigma_d in txhe paper)
    dust_cross_section_per_H_nucleus_normalized_to_1eminus21 = metallicity
    dust_cross_section = dust_cross_section_per_H_nucleus_normalized_to_1eminus21 * 1e-21   # [cm^2]


    # Calculation of dust optical depth (tau_c in the paper)
    # mu_h is the mean mass per H nucleus
    mu_h = 2.3e-24  # [gr] 
    # clumping factor is used to increase the H2 formation to account for density inhomogeneities that are unresolved on the computational grid
    # since the H2 formation rate varies as the square of density, these inhomogeneities increase the overall rate
    dust_optical_depth = clumping_factor * column_density * dust_cross_section / mu_h     # [dimensionless]   

    # Calculation for scaled radiation field (chi in the paper) Eq 4 
    # This scaled radiation field will not likely to hold cell-by-cell every time step, but it should hold on average
    # clumping factor is used to boost the formation rate of the H2 molecules on dust grains (the R term)   
    scaled_radiation_field = 3.1 * (1 + 3.1 * metallicity**0.365) / 4.1  # [dimensionless]

    # Calculation for s in the paper (Eq 2)
    s = np.log(1 + 0.6*scaled_radiation_field + 0.01 * scaled_radiation_field**2 ) / ( 0.6 * dust_optical_depth )
    
    # If s > 2 then fh2 < 0 which is unphysical. So set s greter than 2 to 2. 
    s[s>2] = 2

    # Calculation for the H2 mass fraction (f_H2 in the paper Eq 1)
    h2_mass_fraction = 1 - (3/4) * (s / (1 + 0.25*s))   # [dimensionless]
    
    return h2_mass_fraction, column_density, dust_optical_depth, scaled_radiation_field, s


def X_co_calculator(
    h2_column_density, 
    metallicity, 
):
    print("I am in the function X_co_calculator")

    """This function is being used in order to calculate the X_co for each annulus

    Arguments:
    ----------
    h2_column_density: array-like
        h2_column_density for each particle 
        [gr/cm^2]

    metallicity: array-like
        [Zsolar]

    Returns:
    ----------
    X_co: vector 
        CO conversion factor for each particle
        [cm^-2 /K-km s^-1]
        
    X_co_solar_metallicity: vector
        X_co when metallicity is set to solar metallicity
        [cm^-2 /K-km s^-1]

    References: 
    -----------
    A general model for the CO-H2 conversion factor in galaxies with applications to the star formation law. (Narayanan et al. 2012)

    """ 


    h2_column_density = h2_column_density * constants.gr2M_sun / (constants.cm2pc)**2   
    # h2_column_density [M_sun/pc^2]

    # A general model for the CO-H2 conversion factor in galaxies with applications to the star formation law. (Narayanan et al. 2012)
    # Eq 6 

    X_co = 1.3e21/(metallicity * h2_column_density**0.5) #[cm^-2 /K-km s^-1]

    ####    

    X_co_solar_metallicity = 1.3e21/(constants.solar_metallicity * h2_column_density**0.5) #[cm^-2 /K-km s^-1]

    # Set inf values to NaN. This happens when s >= 2, which it sets fh2 = 0 and h2 column density to zero consequtively and Xco to infinity.
    # Replace inf/-inf with NaN
    X_co[np.isinf(X_co)] = np.nan #
    X_co_solar_metallicity[np.isinf(X_co_solar_metallicity)] = np.nan
    
    
    return X_co, X_co_solar_metallicity


def clumping_factor_from_turbulence_velocity(turbulence_velocity):
    
    """
    Calculating the clumping factor of the every gas particle by using their turbulent velocities. 

    Arguments:
    ----------
    turbulence_velocity: array-like
        turbulence velocity of each gas particle. 
        [km/sec]

    Returns:
    ----------
    clumping_factor: array-like
        clumping factor of each gas particle.
        [1]

    References: 
    -----------
    Private talk with Norman Murray. 

    """
    
    MW_turbulence = 5 # km/s 
    MW_clumping = 1 # 
    
    clumping_factor = MW_clumping * (turbulence_velocity / MW_turbulence)**2 
    
    # Clumping factor can be 1 smallest. 
    clumping_factor[clumping_factor < 1] = 1
    
    return clumping_factor


def calculate_properties_of_gas_particles(gas_df, clumping_factor):

    h2_mass_fraction, gas_column_density, dust_optical_depth, scaled_radiation_field, s = h2_mass_fraction_calculator(
        local_density_scale_height = np.array(gas_df["average_sobolev_smoothingLength"]), # pc
        density = np.array(gas_df["density"]), # gr / cm^3
        metallicity= np.array(gas_df["metallicity"]),  # Zsolar
        clumping_factor = clumping_factor
    )
    
    gas_df["fh2"] = h2_mass_fraction
    gas_df["Mh2"] = gas_df["mass"] * gas_df["fh2"]
    gas_df["tau_c"] = dust_optical_depth

    # Findind the column density of gas particles. 
    h2_column_density = h2_mass_fraction * gas_column_density  # [gr/cm^2]  

    # Estimating the Xco of gas particles. 
    gas_df["Xco"], Xco_solar_metallicity = X_co_calculator(
        h2_column_density = h2_column_density,  # [gr / cm^2]
        metallicity = np.array(gas_df["metallicity"]), # Zsolar
    )    

    # Calculating Lco
    gas_df["alfa_co"] = gas_df["Xco"] / 6.3e19 # [M_solar/pc^-2 (K-km s^-1)^-1]

    gas_df["L_co"] = gas_df["Mh2"] / gas_df["alfa_co"] 
    gas_df["L_co"].fillna(0, inplace=True) # Set NaN to zero.

    return gas_df


def does_file_exist(file_path):
    
    # Check if write_file_name exits. If exists do not run the code.  
    if os.path.isfile(file_path):
        print("File exits. Returning nothing!")
        return True
    else:
        print(f"{file_path} does not exists. Continuing...")
        return False
        



if __name__ == "__main__":

    # galaxy_name = "m12i_res7100_md"
    # galaxy_type = "zoom_in"
    # redshift = "0.0"    

    galaxy_name = sys.argv[1] 
    galaxy_type = sys.argv[2]
    redshift = sys.argv[3]
    
    main(galaxy_name, galaxy_type, redshift)