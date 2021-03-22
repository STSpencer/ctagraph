#!/bin/bash

#SBATCH --account IRIS-IP007-GPU


#SBATCH --partition pascal


#SBATCH -t 20:00:00


#SBATCH --nodes=1


#SBATCH --ntasks=1


#SBATCH --gres=gpu:1


#SBATCH --ntasks-per-node=1


#SBATCH --cpus-per-task=1


#SBATCH -o /home/ir-jaco1/output_test1025.txt

runname='test1025'


module purge


module load rhel7/default-gpu


module unload cuda/8.0


module load python/3.6 cuda/10.1 cudnn/7.6_cuda-10.1 graphviz/2.40.1

source /home/ir-jaco1/t2/bin/activate
python /home/ir-jaco1/k4.py ${runname}


