#!/bin/bash
#SBATCH --time=0-20:55:00
#SBATCH --partition=public-bigmem
#SBATCH --output=full_model_test_moses_project_EMR12_EMR12plus_noNUC_LoL.%j.out
#SBATCH --mem=145000
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=ALL


#Activate environment
module load GCCcore/9.3.0
module load Python/3.8.2
. ~/py38_arven/bin/activate

#Send costoptimal run to cluster
echo "Running GRIMSEL for HP and cooling in HPC" $(hostname)
~/py38_arven/bin/python call_model_moses_project.py
