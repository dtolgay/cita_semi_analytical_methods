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
    ):


    print(f"-------------------------------------------------------------- {galaxy_name} --------------------------------------------------------------")

    # Read cloudy_gas_txt 
    run_dir = "voronoi_1e6"
    cloudy_gas_particles_file_directory = f"/mnt/raid-cita/dtolgay/FIRE/post_processing_fire_outputs/skirt/runs_hden_radius/{galaxy_type}/z{redshift}/{galaxy_name}/{run_dir}" 


    # Check if write_file_name exits. If exists do not run the code.  
    write_file_name = f"{cloudy_gas_particles_file_directory}/semi_analytical_average_sobolev_smoothingLength.txt"
    if os.path.isfile(f"{write_file_name}"):
        print("File exits. Returning nothing!")
        return 0
    else:
        print(f"{write_file_name} does not exists. Continuing...")
    

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

    h2_mass_fraction, gas_column_density, dust_optical_depth, scaled_radiation_field, s, dust_optical_depth = h2_mass_fraction_calculator(
        local_density_scale_height = np.array(gas_particles_df["average_sobolev_smoothingLength"]), # pc
        density = np.array(gas_particles_df["density"]), # gr / cm^3
        metallicity= np.array(gas_particles_df["metallicity"]),  # Zsolar
        clumping_factor = 1
    )

    # Find molecular gas mass
    gas_particles_df["h2_mass"] = h2_mass_fraction * gas_particles_df["mass"] # Msolar    


    # Findind the column density of gas particles. 
    h2_column_density = h2_mass_fraction * gas_column_density  # [gr/cm^2]  

    # Estimating the Xco of gas particles. 
    gas_particles_df["Xco"], Xco_solar_metallicity = X_co_calculator(
        h2_column_density = h2_column_density,  # [gr / cm^2]
        metallicity = np.array(gas_particles_df["metallicity"]), # Zsolar
    )    

    # Calculating Lco
    alfa_co = gas_particles_df["Xco"] / 6.3e19 # [M_solar/pc^-2 (K-km s^-1)^-1]

    gas_particles_df["L_co"] = gas_particles_df["h2_mass"] / alfa_co   

    
    # Write to an output file 

    # Chancing the unit of density once more to make it same with the cloudy gas particle file 
    gas_particles_df["density"] *= (1e10 * constants.M_sun2gr / (constants.kpc2cm)**3)  # gr / cm^3


    write_df = gas_particles_df[[
        "x",
        "y",
        "z",
        'smoothing_length', 
        'mass', 
        'metallicity', 
        'temperature',
        'vx', 
        'vy', 
        'vz', 
        'hden', 
        'radius', 
        'sfr', 
        'turbulence', 
        'density',
        'mu_theoretical', 
        'average_sobolev_smoothingLength', 
        'index', 
        'isrf',
        'h2_mass',
        'Xco',
        'L_co',
    ]]

    # Write to a file 
    header = f"""Gas particles for {galaxy_name} galaxy
    Column 0: x-coordinate (pc)
    Column 1: y-coordinate (pc)
    Column 2: z-coordinate (pc)
    Column 3: smoothing length (pc)
    Column 4: mass (Msolar)
    Column 5: metallicity (Zsolar)
    Column 6: temperature (K)
    Column 7: vx (km/s)
    Column 8: vy (km/s)
    Column 9: vz (km/s)
    Column 10: hydrogen density (cm^-3)
    Column 11: radius (pc)
    Column 12: sfr (Msolar/yr)
    Column 13: turbulence (km/s)
    Column 14: density (gr/cm^-3)
    Column 15: mu_theoretical (1)
    Column 16: average_sobolev_smoothingLength (pc)    
    Column 17: index (1)
    Column 18: isrf [G0]
    Column 19: h2 mass (Msolar)
    Column 20: Xco (cm^-2 (K km s^-1)^-1)
    Column 21: Lco (K km s^-1 pc^2)
    """

    np.savetxt(fname=write_file_name, X = write_df, fmt='%.8e', header=header)
    print(f"{write_file_name} is written!") 


    return 0


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

    clumping_factor: double or int
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
    dust_optical_depth = column_density * dust_cross_section / mu_h     # [dimensionless]   

    # Calculation for scaled radiation field (chi in the paper) Eq 4 
    # This scaled radiation field will not likely to hold cell-by-cell every time step, but it should hold on average
    # clumping factor is used to boost the formation rate of the H2 molecules on dust grains (the R term)   
    scaled_radiation_field = 3.1 * (1 + 3.1 * metallicity**0.365) / (4.1 * clumping_factor)  # [dimensionless]

    # Calculation for s in the paper (Eq 2)
    s = np.log(1 + 0.6*scaled_radiation_field + 0.01 * np.power(scaled_radiation_field,2)) / ( 0.6 * dust_optical_depth )

    # Calculation for the H2 mass fraction (f_H2 in the paper Eq 1)
    h2_mass_fraction = 1 - (3/4) * (s / (1 + 0.25*s))   # [dimensionless]
    h2_mass_fraction[h2_mass_fraction < 0] = 0      # If the result is negative set it to zero
    
    # Set inf values to NaN
    # Replace inf/-inf with NaN
    h2_mass_fraction[np.isinf(h2_mass_fraction)] = np.nan
    column_density[np.isinf(column_density)] = np.nan    
    dust_optical_depth[np.isinf(dust_optical_depth)] = np.nan    
    scaled_radiation_field[np.isinf(scaled_radiation_field)] = np.nan    
    dust_optical_depth[np.isinf(dust_optical_depth)] = np.nan    
    
    return h2_mass_fraction, column_density, dust_optical_depth, scaled_radiation_field, s, dust_optical_depth


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

    # Set inf values to NaN
    # Replace inf/-inf with NaN
    X_co[np.isinf(X_co)] = np.nan
    X_co_solar_metallicity[np.isinf(X_co_solar_metallicity)] = np.nan
    
    
    return X_co, X_co_solar_metallicity


if __name__ == "__main__":

    # galaxy_name = "m12i_res7100_md"
    # galaxy_type = "zoom_in"
    # redshift = "0.0"    

    galaxy_name = sys.argv[1] 
    galaxy_type = sys.argv[2]
    redshift = sys.argv[3]
    
    main(galaxy_name, galaxy_type, redshift)