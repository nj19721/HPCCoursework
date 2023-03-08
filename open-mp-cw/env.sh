# Add any `module load` or `export` commands that your code needs to
# compile and run to this file.
module load icc/2017.1.132-GCC-5.4.0-2.26

export OMP_PLACES=cores
export OMP_PROC_BIND=true
export OMP_NUM_THREADS=28
