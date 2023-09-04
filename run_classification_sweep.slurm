#! /bin/sh

#SBATCH --job-name=mae-sweep-training # job name
#SBATCH --output=/home/yandex/MLFH2023/giladd/hiera/mae-sweep-training.out # redirect stdout
#SBATCH --error=/home/yandex/MLFH2023/giladd/hiera/mae-sweep-training.err # redirect stderr
#SBATCH --partition=studentkillable # (see resources section)
#SBATCH --signal=USR1@120 # how to end job when time’s up
#SBATCH --nodes=1 # number of machines
#SBATCH --ntasks=1 # number of processes
#SBATCH --mem=50000 # CPU memory (MB)
#SBATCH --cpus-per-task=4 # CPU cores per process
#SBATCH --gpus=1 # GPUs in total
#SBATCH --account=gpu-students

python run_classification_sweep.py