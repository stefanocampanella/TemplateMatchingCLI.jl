#! /usr/bin/env bash
#SBATCH --job-name=LAB-PREPROCESS
#SBATCH --account=IscrC_SPACE
#SBATCH --partition=m100_usr_prod
#SBATCH --time=5:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --output=/m100_scratch/userexternal/scampane/Acoustic-Emissions/logs/%A_%a.log
#SBATCH --verbose

module load julia

export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PATH=$PATH:~/.julia/bin

DATA_ROOT=$CINECA_SCRATCH/Acoustic-Emissions
RAW_DIR=$DATA_ROOT/Gabbro6/$1
SOURCE_DIR=$DATA_ROOT/data

mapfile -t DATETIMES < <(find "$RAW_DIR" -type f -name "*.bin" -printf '%f\n' | sed -rn 's/(^[0-9\-]+_[0-9\-]+)_gab6_ch[0-9]+&[0-9]+.bin/\1/p' | sort -u)
DATETIME="${DATETIMES[SLURM_ARRAY_TASK_ID]}"
templatematchingcli preprocess "$RAW_DIR" "$DATETIME" gab6 2021-01-12_20-24-18 16 "$SOURCE_DIR/${DATETIME}.jld2" --inputfreq=$2

exit
