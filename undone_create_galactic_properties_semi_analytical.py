import sys
sys.path.append("/mnt/raid-cita/dtolgay/FIRE/post_processing_fire_outputs")

from tools import readsnap, functions, constants # type: ignore
from tools.functions_create_galactic_properties import sfr_calculator, halo_mass_calculator, mass_average # type: ignore
from tools import functions_readfiles as readfiles # type: ignore

import numpy as np 
import pandas as pd 
from time import time



# Functions

def main(redshift:str, directory_name:str, base_dir:str, verbose:bool = False):

    galaxies_list = []

    ####################################### For firebox 
    for i in range(1000):
        try:
            galaxies_list.append(get_galactic_properties_for_a_galaxy(
                galaxy_name=f"gal{i}", 
                galaxy_type="firebox", 
                redshift=redshift, 
                directory_name=directory_name,
                base_dir=base_dir,
                verbose=verbose
            ))
            
        except Exception as e:
            print(f"Exception occured: \n{e}")     


    # ####################################### For zoom_in
    galaxy_names = [
        "m12b_res7100_md", 
        "m12c_res7100_md",
        "m12f_res7100_md",
        "m12i_res7100_md",
        "m12m_res7100_md",
        "m12q_res7100_md",
        "m12r_res7100_md",
        "m12w_res7100_md",
        "m11d_r7100",
        "m11e_r7100",
        "m11h_r7100",
        "m11i_r7100",
        "m11q_r7100",            
    ]

    for galaxy_name in galaxy_names:
        try:
            galaxies_list.append(get_galactic_properties_for_a_galaxy(
                galaxy_name=galaxy_name, 
                galaxy_type="zoom_in", 
                redshift=redshift, 
                directory_name=directory_name,
                base_dir=base_dir
            ))
            
        except Exception as e:
            print(f"Exception occured: \n{e}")   


    ####################################### For particle_split
    galaxy_names = [
        "m12i_res7100_md",
    ]

    for galaxy_name in galaxy_names:
        try:
            galaxies_list.append(get_galactic_properties_for_a_galaxy(
                galaxy_name=galaxy_name, 
                galaxy_type="particle_split", 
                redshift=redshift,
                directory_name=directory_name,
                base_dir=base_dir
            ))
            
        except Exception as e:
            print(f"Exception occured: \n{e}")   


    ####################################### Turn list to a pandas dataframe
    galaxies = pd.DataFrame(galaxies_list)
    

    ###################################### Write to a file
    header = f"""
    # For cf_velocity_dependent the MW velocity is 8 km/sec MW clumping is 1.
    # name 
    # galaxy_type
    # redshift 
    # sfr_instantaneous [Msolar/year]
    # sfr_5Myr [Msolar/year]
    # sfr_10Myr [Msolar/year]
    # sfr_100Myr [Msolar/year]
    # gas_mass [Msolar]
    # star_mass [Msolar]
    # gas_average_metallicity [Zsolar]
    # star_average_metallicity [Zsolar]
    # alpha_co [Msolar / (K km s^-1 pc^2)]
    # halo_mass [Msolar]
    # number_of_NaN_indices [1]
    # Mh2 [Msolar]
    # tau_c [1]
    # L_co_10 [K km s^-1 pc^2] 
    """

    save_dir = "/mnt/raid-cita/dtolgay/FIRE/post_processing_fire_outputs/skirt/python_files/semi_analytical_methods/data"
    file_path = f"{save_dir}/galactic_properties_L_line_average_sobolev_smoothingLength_z{int(float(redshift))}_semiAnalytical.csv"

    # Append the DataFrame to the file
    galaxies.to_csv(file_path, mode='w', sep=',', header=header, index=False, float_format='%.5e')

    print(f"File saved to {file_path}")    

    print(galaxies)

    return 0


def get_galactic_properties_for_a_galaxy(galaxy_name:str, galaxy_type:str, redshift:str, directory_name:str, base_dir:str, verbose:bool = False):
    
    print(f"--------------  {galaxy_name}  --------------")
    
    start = time()
    
    fdir = f"{base_dir}/{galaxy_type}/z{redshift}/{galaxy_name}/{directory_name}"
    
    # Read gas particles
    if verbose: print("I am reading cloudy gas particles.")
    gas = readfiles.read_cloudy_gas_particles(
        galaxy_name, 
        galaxy_type, 
        redshift, 
        directory_name,
        base_fdir = f"/mnt/raid-cita/dtolgay/FIRE/post_processing_fire_outputs/skirt/runs_hden_radius"
    )   

    # Read star particles 
    star_column_names = [
        "x",  # pc
        "y",  
        "z",  
        "vx", # km/s
        "vy",
        "vz",
        "metallicity", # (1)
        "mass", # Msolar
        "age",  # Myr
    ]

    if verbose: print("I am reading comphrensive star particles.")
    star = np.loadtxt(
        fname=f"{fdir}/comprehensive_star.txt",
        skiprows=1
    )

    star = pd.DataFrame(star, columns=star_column_names)
    
    
    # Reading different clumping factor runs 
    semi_analytical_runs = {
        "cf_1": {
            "df": pd.DataFrame(),
            "title": "cf_1",
            "read_file_path": f"{fdir}/semi_analytical_averageSobolevH_cf_1.txt"
        },
        # "cf_2": {
        #     "df": pd.DataFrame(),
        #     "title": "cf_2",
        #     "read_file_path": f"{fdir}/semi_analytical_averageSobolevH_cf_2.txt"
        # },  
        # "cf_10": {
        #     "df": pd.DataFrame(),
        #     "title": "cf_10",
        #     "read_file_path": f"{fdir}/semi_analytical_averageSobolevH_cf_10.txt"
        # },
        # "cf_100": {
        #     "df": pd.DataFrame(),
        #     "title": "cf_100",
        #     "read_file_path": f"{fdir}/semi_analytical_averageSobolevH_cf_100.txt"
        # },
        # "cf_500": {
        #     "df": pd.DataFrame(),
        #     "title": "cf_500",
        #     "read_file_path": f"{fdir}/semi_analytical_averageSobolevH_cf_500.txt"
        # },    
        # "cf_functionOfTurbulence" : {
        #     "df": pd.DataFrame(),
        #     "title": "cf_velocity_dependent",
        #     "read_file_path": f"{fdir}/semi_analytical_averageSobolevH_cf_functionOfTurbulence_.txt"
        # },                                      
    }     
    
    if verbose: print("I am reading different clumping factors for semi analytical runs.")
    semi_analytical_runs = read_file_for_clumping_factors(runs = semi_analytical_runs)
    
    
    ############ Calculate galactic properties 
    if verbose: print("I am calculating several galactic properties.")
    
    ## sfr
    sfr_instantaneous = np.sum(gas["sfr"]) 
    sfr_5Myr = sfr_calculator(star_df = star, within_how_many_Myr = 5)
    sfr_10Myr = sfr_calculator(star_df = star, within_how_many_Myr = 10)
    sfr_100Myr = sfr_calculator(star_df = star, within_how_many_Myr = 100)
    
    ## gas mass
    total_gas_mass = np.sum(gas["mass"])
    total_star_mass = np.sum(star["mass"])

    # Find the indices of gas and star particles within 8 kpc from the center
    R_inner = 8e3 #kpc 
    indices_gas_inner_galaxy = np.where(np.sqrt(gas['x']**2 + gas['y']**2 + gas['z']**2) < R_inner)[0] 
    indices_star_inner_galaxy = np.where(np.sqrt(star['x']**2 + star['y']**2 + star['z']**2) < R_inner)[0] 
    gas_inner_galaxy = gas.iloc[indices_gas_inner_galaxy]
    star_inner_galaxy = star.iloc[indices_star_inner_galaxy]

    ## metallicity
    average_gas_metallicity = sum(gas["metallicity"] * gas["mass"]) / sum(gas["mass"])
    average_star_metallicity = sum(star["metallicity"] * star["mass"]) / sum(star["mass"]) / constants.solar_metallicity
    
    gas_metallicity_inner_galaxy = sum(gas_inner_galaxy['metallicity'] * gas_inner_galaxy['mass']) / sum(gas_inner_galaxy['mass'])
    star_metallicity_inner_galaxy = sum(star_inner_galaxy['metallicity'] * star_inner_galaxy['mass']) / sum(star_inner_galaxy['mass'])

    
    
    ## line_luminosities
    
    lines = [
        'L_co'
    ]
    
    total_line_luminosities = calculate_galactic_properties_for_different_clumping_factors(
        runs = semi_analytical_runs, 
        column_names=lines, 
        operation="sum"
    )
    
    
    ## alpha_co
#     alpha_co = np.sum(semi_analytical["h2_mass"]) / total_line_luminosities["L_co_10"] # if h2 mass is NaN probably it is zero already.
    averaged_galactic_properties = calculate_galactic_properties_for_different_clumping_factors(
        runs = semi_analytical_runs,
        column_names=['fh2', 'alfa_co', 'tau_c'],
        operation="nan_avg",
    )
    
    ### Mh2 
    Mh2 = calculate_galactic_properties_for_different_clumping_factors(
        runs = semi_analytical_runs,
        column_names = ['Mh2'],
        operation = "sum"
    )
    
    ## halo mass
    Mhalo = halo_mass(
        galaxy_name = galaxy_name, 
        galaxy_type = galaxy_type, 
        redshift = redshift
    )

    # Create a dictionary 
    other_properties = {
        "name": f"{galaxy_name}",
        "galaxy_type": f"{galaxy_type}",
        "redshift": f"{redshift}",
        "sfr": sfr_instantaneous, # Msolar / year
        "sfr_5Myr": sfr_5Myr, # Msolar / year
        "sfr_10Myr": sfr_10Myr, # Msolar / year
        "sfr_100Myr": sfr_100Myr, # Msolar / year
        "gas_mass": total_gas_mass, # Msolar
        "star_mass": total_star_mass, # Msolar
        "gas_average_metallicity": average_gas_metallicity, # Zsolar
        "star_average_metallicity": average_star_metallicity, # Zsolar
        "gas_metallicity_inner_galaxy": gas_metallicity_inner_galaxy, # Zsolar
        "star_metallicity_inner_galaxy": star_metallicity_inner_galaxy, # Zsolar
        # "alpha_co": alpha_co, # Msolar / (K km s^-1 pc^2)
        "halo_mass": Mhalo, # Msolar,
#         "number_of_NaN_indices": len(nan_indices),
    }    

    # Merge dictionaries
    galactic_properties = {
        **other_properties, 
        **total_line_luminosities, 
        **averaged_galactic_properties,
        **Mh2,
    }

    stop = time()
    
    print(f"For {galaxy_name}, it took {round((stop-start)/60, 3)} minutes")

    return galactic_properties


def read_file_for_clumping_factors(runs):
        
    columns_names = [
        "index",
        "fh2",
        "Mh2",
        "tau_c",
        "alfa_co",
        "L_co"
    ]
    
    runs_updated = runs.copy()


    for key in list(runs.keys()):
        runs_updated[key]["df"] = pd.DataFrame(
            np.loadtxt(f"{runs_updated[key]['read_file_path']}"),
            columns = columns_names
        )
    
    return runs_updated


def calculate_galactic_properties_for_different_clumping_factors(runs:pd.DataFrame, column_names:str, operation:str):
    
    prop_dict = {}
    
    for key in list(runs.keys()):
        df = runs[key]["df"]
        for column in column_names:
            if operation == "sum":
                prop_dict[f"{column}_{runs[key]['title']}"] = sum(df[column])
            if operation == "nan_avg":
                non_nan_count = df[column].notna().sum()
                prop_dict[f"{column}_{runs[key]['title']}"] = np.sum(df[column]) / non_nan_count
    
    return prop_dict


if __name__ == "__main__":

    base_dir = "/mnt/raid-cita/dtolgay/FIRE/post_processing_fire_outputs/skirt/runs_hden_radius"
    redshift = sys.argv[1]
    directory_name=sys.argv[2]


    main(redshift=redshift, directory_name=directory_name, base_dir=base_dir, verbose=True)