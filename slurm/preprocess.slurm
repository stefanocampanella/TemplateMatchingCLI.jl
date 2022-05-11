#! /usr/bin/env bash
#SBATCH --job-name=LAB-PREPROCESS
#SBATCH --account=IscrC_SPACE
#SBATCH --array=0-130
#SBATCH --partition=m100_usr_prod
#SBATCH --time=5:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=13
#SBATCH --output=logs/%A_%a.log
#SBATCH	--requeue 
#SBATCH --verbose

DATA_ROOT=$CINECA_SCRATCH/Acoustic-Emissions
RAW_DIR=$DATA_ROOT/Gabbro6
SOURCE_DIR=$DATA_ROOT/data

export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PATH=$PATH:~/.julia/bin

mapfile -t DATETIMES < <(find "$RAW_DIR" -type f -name "*.bin" -printf '%f\n' | sed -rn 's/(^[0-9\-]+_[0-9\-]+)_gab6_ch[0-9]+&[0-9]+.bin/\1/p' | sort -nu)
DATETIME="${DATETIMES[SLURM_ARRAY_TASK_ID]}"
templatematchingcli preprocess "$RAW_DIR" "$DATETIME" gab6 "$SOURCE_DIR" --exclude=1,2,3

exit
