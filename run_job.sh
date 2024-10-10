#!/bin/bash
#SBATCH --job-name=my_job
#SBATCH --account=kempner_dam_lab
#SBATCH --partition=kempner
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --time=0-01:00
#SBATCH --mem=32G
#SBATCH --output=my_job_output.out
#SBATCH --error=my_job_error.err
#SBATCH --mail-type=END
#SBATCH --mail-user=hdiaz@g.harvard.edu

# Load modules
module load python/3.10.9-fasrc01 

# Activate conda environment (optional)

# Run the job
python generates.py