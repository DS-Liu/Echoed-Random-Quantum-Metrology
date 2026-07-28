#!/bin/bash
#SBATCH -J photon_loss_SinglePhotonDriving
#SBATCH -o job-%j-%x.out -e job-%j-%x.out
#SBATCH -p CPU-192C768GB --qos=qos_cpu_192c768gb -n 1001
export OMP_NUM_THREADS=1 # This is very important, otherwise the routine will run extremely slow.
##. /etc/profile.d/modules.sh

echo "Time is $(date), Directory is $PWD"
echo "This job runs on the following nodes: $SLURM_JOB_NODELIST"
echo "Allocated $SLURM_JOB_CPUS_PER_NODE cpu cores, $SLURM_NTASKS tasks in total."

for i in 0 1 2 3 4
do
    echo "Running kappa$i for SinglePhotonDriving, '$\epsilon/\chi=100$, $\chi t=1.5$'."
    mpiexec -n $SLURM_NTASKS python -m mpi4py.futures photon_loss.py --get_rho0 --dim=170 --epsilon_index=10 --N=15 --driving_type='SinglePhotonDriving' --kappa_index=$i
done

for i in 1 2 3 4
do
    echo "Running kappa$i for SinglePhotonDriving, '$\epsilon/\chi=100$, $\chi t=1.5$'."
    mpiexec -n $SLURM_NTASKS python -m mpi4py.futures photon_loss.py --get_CFI_thetab --epsilon_index=10 --N=15 --driving_type='SinglePhotonDriving' --kappa_index=$i
done

for i in 0 1 2 3 4
do
    echo "Running kappa$i for SinglePhotonDriving, '$\epsilon/\chi=40$, $\chi t=1.5$'."
    mpiexec -n $SLURM_NTASKS python -m mpi4py.futures photon_loss.py --get_rho0 --dim=170 --epsilon_index=6 --N=15 --driving_type='SinglePhotonDriving' --kappa_index=$i
done

for i in 1 2 3 4
do
    echo "Running kappa$i for SinglePhotonDriving, '$\epsilon/\chi=40$, $\chi t=1.5$'."
    mpiexec -n $SLURM_NTASKS python -m mpi4py.futures photon_loss.py --get_CFI_thetab --epsilon_index=6 --N=15 --driving_type='SinglePhotonDriving' --kappa_index=$i
done