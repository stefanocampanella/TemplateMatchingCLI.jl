#! /usr/bin/env bash
#SBATCH --job-name=LAB-MAKETEMPLATES
#SBATCH --account=IscrC_SPACE
#SBATCH --array=0-130
#SBATCH --partition=m100_usr_prod
#SBATCH --time=10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/%A_%a.log
#SBATCH	--requeue 
#SBATCH --verbose

DATA_ROOT=$CINECA_SCRATCH/Acoustic-Emissions
SOURCE_DIR=$DATA_ROOT/data
SENSORSXYZ=$DATA_ROOT/passive-xyz.csv
CATALOGUE=$DATA_ROOT/catalogue.csv
TEMPLATES_DIR=$DATA_ROOT/templates

export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PATH=$PATH:~/.julia/bin

mapfile -t DATAFILES < <(find "$SOURCE_DIR" -type f -name "*.jld2" | sort -n)
DATAFILE="${DATAFILES[SLURM_ARRAY_TASK_ID]}"
templatematchingcli maketemplates "$DATAFILE" "$SENSORSXYZ" "$CATALOGUE" "$TEMPLATES_DIR"

exit