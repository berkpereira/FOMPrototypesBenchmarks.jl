#!/bin/bash
#SBATCH --job-name=fombench
#SBATCH --output=logs/bench_%j.out
#SBATCH --error=logs/bench_%j.err

#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=27          # grab logical CPUs

# SBATCH --hint=nomultithread       # no hyperthread scheduling...
# SBATCH --cpu-bind=cores         # bind to cores, don't move worker threads

#SBATCH --cpu-freq=performance

#SBATCH --mail-type=ALL
#SBATCH --mail-user=gabriel.pereira@eng.ox.ac.uk

#SBATCH --exclusive              # no other jobs on this node
#SBATCH --time=24:00:00

cd $HOME/repos/FOMPrototypesBenchmarks.jl
# kill my Julia processes before starting
pkill -u "$USER" -f julia

echo "[$(date)] Starting solver benchmark job on $(hostname)"

# ensure each Julia worker is single-threaded (?)
# and/or linear algebra libraries?
export JULIA_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1

# fire up the (one) driver that will spawn its own workers
julia --project=. bench_driver.jl

echo "[$(date)] solver benchmark job finished"