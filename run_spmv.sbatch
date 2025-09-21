#!/usr/bin/env bash
#SBATCH --job-name=spmv_bench
#SBATCH --output=logs/spmv_%j.out
#SBATCH --error=logs/spmv_%j.err
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --ntasks=1

# Single-threaded BLAS/Julia to avoid noisy contention
export JULIA_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1

# send start notification
NTFY_TOPIC="gabrielpereira-fenway-8bwR6y2O5vL6L2qgPPuC"
ERROR_FILE="logs/spmv_${SLURM_JOB_ID}.err"
curl -d "SLURM job '${SLURM_JOB_NAME}' started." "ntfy.sh/${NTFY_TOPIC}"

echo "[$(date)] Starting SpMV benchmark job on $(hostname)"

# kill other Julia processes before starting
pkill -f julia

# Instantiate the environment and run the local SpMV benchmarks
julia --project=. bench_spmv_real_complex.jl

echo "[$(date)] SpMV benchmark job finished"

# send notification including end of error log to ntfy.sh
curl -d "SLURM job '${SLURM_JOB_NAME}' finished." "ntfy.sh/${NTFY_TOPIC}"
if [ -s "$ERROR_FILE" ]; then
    LOG_TAIL=$(tail -n 50 "$ERROR_FILE")
    curl -H "Title: Error Log for ${SLURM_JOB_NAME}" \
    -d "$LOG_TAIL" \
    "ntfy.sh/${NTFY_TOPIC}"
fi