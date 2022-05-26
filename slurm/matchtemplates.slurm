#! /usr/bin/env bash
#SBATCH --job-name=LAB-MATCHTEMPLATES
#SBATCH --account=IscrC_SPACE
#SBATCH --array=0-31
#SBATCH --partition=m100_usr_prod
#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --output=logs/%A_%a.log
#SBATCH	--requeue 
#SBATCH --verbose

DATA_ROOT=$CINECA_SCRATCH/Acoustic-Emissions
SOURCE_DIR=$DATA_ROOT/data
OUTPUT_DIR=$DATA_ROOT/outputs
TEMPLATES_FILE=$DATA_ROOT/gab6.jld2
NUM_TEMPLATES_BATCHES=100
SENSORS_FILE=$DATA_ROOT/passive-xyz.csv
OUTPUT_DIR=$DATA_ROOT/outputs


export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PATH=$PATH:~/.julia/bin


mapfile -t DATA_SOURCE_FILES < <(find "$SOURCE_DIR" -type f -name "*_gab6.jld2" | sort -u)
TEMPLATES_BATCH_INDEX=$((SLURM_ARRAY_TASK_ID % NUM_TEMPLATES_BATCHES))
TEMPLATES_BATCH="$TEMPLATES_BATCH_INDEX/$NUM_TEMPLATES_BATCHES"
DATA_SOURCE_FILE=${DATA_SOURCE_FILES[$1]}
echo "Processing $DATA_SOURCE_FILE, templates batch $TEMPLATES_BATCH"

mkdir -p "$OUTPUT_DIR"
STARTTIME=$(date +%s)
templatematchingcli matchtemplates --batches="$TEMPLATES_BATCH" "$DATA_SOURCE_FILE" "$TEMPLATES_FILE" "$SENSORS_FILE" "$OUTPUD_DIR"
ENDTIME=$(date +%s)
echo "Elapsed time: $((ENDTIME - STARTTIME)) seconds"

exit
