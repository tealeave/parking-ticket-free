#!/bin/bash

#SBATCH --job-name=parsing_data_job
#SBATCH -p free              ## free partition
#SBATCH --nodes=1            ## use  node, don't ask for multiple
#SBATCH --ntasks=32          
#SBATCH --mem=96G
#SBATCH --error=%x.%A.err    ## Slurm error  file, %x - job name, %A job id
#SBATCH --out=%x.%A.out      ## Slurm output file, %x - job name, %A job id
#SBATCH --mail-type=fail,end
#SBATCH --mail-user=ddlin@hs.uci.edu

# source /data/homezvol2/ddlin/mambaforge-pypy3/envs/d2l/lib/python3.9/venv/scripts/common/activate

python parsing_data.py
