#!/usr/bin/env bash
#SBATCH --job-name=spmv_bench
#SBATCH --output=logs/spmv_%j.out
#SBATCH --error=logs/spmv_%j.err
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --ntasks=1

# Single-threaded BLAS/Julia to avoid noisy contention
export JULIA_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1

echo "[$(date)] Starting SpMV benchmark job on $(hostname)"

# kill other Julia processes before starting
pkill -f julia

# Instantiate the environment and run the local SpMV benchmarks
julia --project=. bench_spmv_real_complex.jl

echo "[$(date)] SpMV benchmark job finished"