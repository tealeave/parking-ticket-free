#!/bin/bash

#SBATCH --job-name=pytorch_parking_job
#SBATCH -p free-gpu          ## free partition
#SBATCH --nodes=1            ## use  node, don't ask for multiple
##SBATCH --ntasks=8          
#SBATCH --mem=64G
#SBATCH --gres=gpu:V100:1
#SBATCH --error=%x.%A.err    ## Slurm error  file, %x - job name, %A job id
#SBATCH --out=%x.%A.out      ## Slurm output file, %x - job name, %A job id
#SBATCH --mail-type=fail,end
#SBATCH --mail-user=ddlin@hs.uci.edu

# load modules will actially throw an error
# module purge
# module load python/3.10.2
# module load cuda/11.7.1
# module load pytorch/1.11.1

# will need to activate my own python env for deep learning before submitting the job
# source /data/homezvol2/ddlin/mambaforge-pypy3/envs/d2l/lib/python3.9/venv/scripts/common/activate

python la_parking_torch.py
