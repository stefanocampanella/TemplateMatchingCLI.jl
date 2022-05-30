#! /usr/bin/env bash
#SBATCH --job-name=LAB-MATCHTEMPLATES
#SBATCH --account=IscrC_SPACE
#SBATCH --partition=m100_usr_prod
#SBATCH --time=5:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --output=logs/%A_%a.log
#SBATCH	--requeue 
#SBATCH --verbose

DATA_ROOT=$CINECA_SCRATCH/Acoustic-Emissions
SOURCE_DIR=$DATA_ROOT/data
OUTPUT_DIR=$DATA_ROOT/outputs
TEMPLATES_FILE=$DATA_ROOT/gab6.jld2
SENSORS_FILE=$DATA_ROOT/passive-xyz.csv
OUTPUT_DIR=$DATA_ROOT/outputs

export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PATH=$PATH:~/.julia/bin

mapfile -t DATA_SOURCE_FILES < <(find "$SOURCE_DIR" -type f -name "*_gab6.jld2" | sort -u)
DATA_SOURCE_FILE=${DATA_SOURCE_FILES[SLURM_ARRAY_TASK_ID]}
echo "Processing $DATA_SOURCE_FILE, templates batch $TEMPLATES_BATCH"

mkdir -p "$OUTPUT_DIR"
time templatematchingcli matchtemplates "$DATA_SOURCE_FILE" "$TEMPLATES_FILE" "$SENSORS_FILE" "$OUTPUT_DIR"

exit
