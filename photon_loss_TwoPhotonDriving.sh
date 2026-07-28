#!/bin/bash
#SBATCH -J photon_loss_TwoPhotonDriving
#SBATCH -o job-%j-%x.out -e job-%j-%x.out
#SBATCH -p CPU-192C768GB --qos=qos_cpu_192c768gb -n 1001
export OMP_NUM_THREADS=1 # This is very important, otherwise the routine will run extremely slow.
##. /etc/profile.d/modules.sh

echo "Time is $(date), Directory is $PWD"
echo "This job runs on the following nodes: $SLURM_JOB_NODELIST"
echo "Allocated $SLURM_JOB_CPUS_PER_NODE cpu cores, $SLURM_NTASKS tasks in total."

for i in 0 1 2 3 4
do
    echo "Running kappa$i for TwoPhotonDriving, '$\epsilon/\chi=20$, $\chi t=0.8$'."
    mpiexec -n $SLURM_NTASKS python -m mpi4py.futures photon_loss.py --get_rho0 --dim=200 --epsilon_index=13 --N=8 --driving_type='TwoPhotonDriving' --kappa_index=$i
done

for i in 1 2 3 4
do
    echo "Running kappa$i for TwoPhotonDriving, '$\epsilon/\chi=20$, $\chi t=0.8$'."
    mpiexec -n $SLURM_NTASKS python -m mpi4py.futures photon_loss.py --get_CFI_thetab --epsilon_index=13 --N=8 --driving_type='TwoPhotonDriving' --kappa_index=$i
done

for i in 0 1 2 3 4
do
    echo "Running kappa$i for TwoPhotonDriving, '$\epsilon/\chi=10$, $\chi t=0.8$'."
    mpiexec -n $SLURM_NTASKS python -m mpi4py.futures photon_loss.py --get_rho0 --dim=200 --epsilon_index=10 --N=8 --driving_type='TwoPhotonDriving' --kappa_index=$i
done

for i in 1 2 3 4
do
    echo "Running kappa$i for TwoPhotonDriving, '$\epsilon/\chi=10$, $\chi t=0.8$'."
    mpiexec -n $SLURM_NTASKS python -m mpi4py.futures photon_loss.py --get_CFI_thetab --epsilon_index=10 --N=8 --driving_type='TwoPhotonDriving' --kappa_index=$i
done