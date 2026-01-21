#!/bin/bash
#SBATCH --account=gts-drahnev6
#SBATCH -N1 --ntasks-per-node=6
#SBATCH --gres=gpu:RTX_6000:1
#SBATCH --mem-per-gpu=150GB
#SBATCH -t 1-00:00:00
#SBATCH -qinferno
#SBATCH -o Report-%A_%a.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=hfung3@gatech.edu

cd $SLURM_SUBMIT_DIR
module load pytorch/2.1.0
export WANDB_API_KEY=REMOVED

# Run the script with the instance corresponding to the task ID
python test_model.py

