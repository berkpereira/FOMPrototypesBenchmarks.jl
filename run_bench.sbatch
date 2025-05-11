#!/bin/bash
#SBATCH --job-name=fombench
#SBATCH --output=logs/bench_%j.out
#SBATCH --error=logs/bench_%j.err

#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32       # grab however many logical CPUs
#SBATCH --exclusive              # no other jobs on this node
#SBATCH --time=00:30:00

cd $HOME/repos/FOMPrototypesBenchmarks.jl

# ensure each Julia worker is single-threaded
export JULIA_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

# fire up the one driver that will spawn its own workers
julia --project=. bench_driver.jl