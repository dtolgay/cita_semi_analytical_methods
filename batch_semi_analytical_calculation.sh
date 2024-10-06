#!/bin/bash -l
#PBS -l nodes=1:ppn=128
#PBS -l walltime=23:00:00
#PBS -N all_galaxies_semi_analytical_calculation
#PBS -o all_galaxies_semi_analytical_calculation.out
#PBS -e all_galaxies_semi_analytical_calculation.err
#PBS -q starq
#PBS -r n
#PBS -j oe

# hpq -> 16
# starq -> 128

module purge 
module load python 

cd $post_processing_fire_outputs
cd skirt/python_files/semi_analytical_methods


number_of_background_galaxies=128

# Function to wait for all background processes to finish
wait_for_jobs() {
    for job in $(jobs -p)
    do
        wait $job
    done
}



####### zoom_in
# Counter for every 10 galaxies
counter=0

# List of galaxy names
galaxy_names=(
    "m12b_res7100_md" 
    "m12c_res7100_md"
    "m12f_res7100_md"
    "m12i_res7100_md"
    "m12m_res7100_md"
    "m12q_res7100_md"
    "m12r_res7100_md"
    "m12w_res7100_md"
    "m11d_r7100"
    "m11e_r7100"
    "m11h_r7100"
    "m11i_r7100"
    "m11q_r7100"
)


for galaxy in "${galaxy_names[@]}"; do

    python semi_analytical_calculation.py $galaxy zoom_in 3.0 &

    # Increment counter
    ((counter++))

    # Every 10th galaxy, wait for all background jobs to finish
    if [ $counter -ge $number_of_background_galaxies ]; then
        wait_for_jobs
        counter=0
    fi
done

# Wait for the last set of jobs to finish
wait_for_jobs



####### firebox
counter=0

for i in {0..999}
do
    python semi_analytical_calculation.py gal$i firebox 3.0 &

    # Increment counter
    ((counter++))

    # Every 10th galaxy, wait for all background jobs to finish
    if [ $counter -ge $number_of_background_galaxies ]; then
        wait_for_jobs
        counter=0
    fi
done

# Wait for the last set of jobs to finish
wait_for_jobs



####### particle_split
counter=0

List of galaxy names
galaxy_names=(
    "m12i_r880_md" 
)


for galaxy in "${galaxy_names[@]}"; do

    python semi_analytical_calculation.py $galaxy particle_split 3.0 &

    # Increment counter
    ((counter++))

    # Every 10th galaxy, wait for all background jobs to finish
    if [ $counter -ge $number_of_background_galaxies ]; then
        wait_for_jobs
        counter=0
    fi
done

# Wait for the last set of jobs to finish
wait_for_jobs
