#!/bin/bash
#SBATCH --job-name=h_ftjob
#SBATCH --account=kempner_dam_lab
#SBATCH --partition=kempner
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=4
#SBATCH --time=0-05:00:00
#SBATCH --mem=128G
#SBATCH --output=my_job_output.out
#SBATCH --error=my_job_error.err
#SBATCH --mail-type=END
#SBATCH --mail-user=hdiaz@g.harvard.edu
cd /n/home00/hdiaz/StarterProj


# Load modules
module load python/3.10.9-fasrc01 

# Activate conda environment (optional)
mamba activate myenv

# Run the job
python Finetune.py
