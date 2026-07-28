#!/bin/sh
#SBATCH -J fig2_statistics -o job-%j-%x.out -e job-%j-%x.out
#SBATCH -p CPU-192C768GB -n 1001 --qos=qos_cpu_192c768gb
export OMP_NUM_THREADS=1 # This is very important, otherwise the routine will run extremely slow.
##. /etc/profile.d/modules.sh

echo Time is `date`, Directory is $PWD
echo This job runs on the following nodes: $SLURM_JOB_NODELIST, allocated $SLURM_JOB_CPUS_PER_NODE cpu cores, $SLURM_NTASKS tasks in total.

mpiexec -n $SLURM_NTASKS python -m mpi4py.futures fig2_statistics.py --scan_epsilon