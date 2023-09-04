#! /bin/sh

#SBATCH --job-name=mae-sweep-trainig # job name
#SBATCH --output=/home/yandex/MLFH2023/giladd/hiera/mae-sweep-trainig.out # redirect stdout
#SBATCH --error=/home/yandex/MLFH2023/giladd/hiera/mae-sweep-trainig.err # redirect stderr
#SBATCH --partition=studentkillable # (see resources section)
#SBATCH --signal=USR1@120 # how to end job when time’s up
#SBATCH --nodes=1 # number of machines
#SBATCH --ntasks=1 # number of processes
#SBATCH --mem=50000 # CPU memory (MB)
#SBATCH --cpus-per-task=4 # CPU cores per process
#SBATCH --gpus=1 # GPUs in total
#SBATCH --account=gpu-students

python run_mae_sweep.py