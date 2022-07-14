#! /usr/bin/env bash
#SBATCH --job-name=LAB-MATCHTEMPLATES
#SBATCH --account=IscrC_SPACE
#SBATCH --partition=m100_usr_prod
#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --output=/m100_scratch/userexternal/scampane/Acoustic-Emissions/logs/%A_%a.log
#SBATCH --constraint=gpureport
#SBATCH	--requeue 
#SBATCH --verbose

module load spack && module load julia

export JULIA_CUDA_USE_BINARYBUILDER=false
export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PATH=$PATH:~/.julia/bin

DATA_ROOT=$CINECA_SCRATCH/Acoustic-Emissions
DATA_DIR=$DATA_ROOT/data
OUTPUT_DIR=$DATA_ROOT/outputs
TEMPLATES_FILE=$DATA_ROOT/gab6.jld2
SENSORS_FILE=$DATA_ROOT/passive-xyz.csv
OUTPUT_DIR=$DATA_ROOT/outputs

mapfile -t DATA_FILES < <(find "$DATA_DIR" -type f -name "*_gab6.jld2" | sort -u)
DATA=${DATA_FILES[SLURM_ARRAY_TASK_ID]}

mkdir -p "$OUTPUT_DIR"
DATA_NAME=$(basename "${DATA%_gab6.jld2}")
templatematchingcli matchtemplates "$DATA" "$TEMPLATES_FILE" "$SENSORS_FILE" "$OUTPUT_DIR/$DATA_NAME.jld2"

exit
