#!/bin/sh

# Batch this job with `sbatch launch.sh``

#SBATCH --job-name=DYCOMS_CFL_2
#SBATCH --output=out_%j.txt
#SBATCH --partition=allgpu
#SBATCH --gres=gpu:titanv:1

#SBATCH --ntasks=1
#SBATCH --time=08:00:00
#SBATCH --mem=12GB


echo ------------------------------------------------------
if [ "${SLURM_NNODES}" -eq "1" ]
then
  echo 'CPUS(xNODES): '${SLURM_JOB_CPUS_PER_NODE}'(x1)'
else
  echo 'CPUS(xNODES): '${SLURM_JOB_CPUS_PER_NODE}
fi
echo 'Job is running on nodes:'
echo $SLURM_JOB_NODELIST
echo ------------------------------------------------------
echo SLURM: submission node:        $SLURM_SUBMIT_HOST
echo SLURM: partition:              $SLURM_JOB_PARTITION
echo SLURM: submission directory:   $SLURM_SUBMIT_DIR
echo SLURM: job identifier:         $SLURM_JOBID
echo SLURM: job name:               $SLURM_JOB_NAME
echo SLURM: current home directory: $HOME
echo SLURM: PATH:                   $PATH
echo ------------------------------------------------------

source /etc/profile
module load compile/gcc/7.2.0 openmpi/3.0.0 lib/cuda/10.1.243

export PATH=/home/jekozdon/opt/julia/1.3.1/bin:$PATH
mpirun julia --project experiments/Atmos_LES/dycoms_CFL_2.jl --output-dir=output_${SLURM_JOBID}
