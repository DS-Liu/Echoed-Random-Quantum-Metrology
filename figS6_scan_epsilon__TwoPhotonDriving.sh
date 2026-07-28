#!/bin/bash
#SBATCH -J figS6_scan_epsilon__TwoPhotonDriving
#SBATCH -o job-%j-%x.out -e job-%j-%x.out
#SBATCH -p CPU-192C768GB --qos=qos_cpu_192c768gb -n 1001
export OMP_NUM_THREADS=1 # This is very important, otherwise the routine will run extremely slow.
##. /etc/profile.d/modules.sh

echo "Time is $(date), Directory is $PWD"
echo "This job runs on the following nodes: $SLURM_JOB_NODELIST"
echo "Allocated $SLURM_JOB_CPUS_PER_NODE cpu cores, $SLURM_NTASKS tasks in total."

mpiexec -n $SLURM_NTASKS python -m mpi4py.futures figS6_scan_epsilon__TwoPhotonDriving.py --scan_epsilon