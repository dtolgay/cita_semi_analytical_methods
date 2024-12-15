#!/bin/bash -l
#PBS -l nodes=1:ppn=128
#PBS -l walltime=23:50:00
#PBS -N z3_galactic_semi_analytical
#PBS -o z3_galactic_semi_analytical.out
#PBS -e z3_galactic_semi_analytical.err
#PBS -q starq
#PBS -r n
#PBS -j oe


module purge 
ml python 

cd /mnt/raid-cita/dtolgay/FIRE/post_processing_fire_outputs/skirt/python_files/semi_analytical_methods

redshift=3.0

python create_galactic_properties_semi_analytical.py $redshift