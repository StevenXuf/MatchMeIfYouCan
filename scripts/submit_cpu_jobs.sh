#!/bin/bash -l
#SBATCH --time=48:00:00
#SBATCH --account=p200630
#SBATCH --partition=cpu
#SBATCH --qos=default
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1

module load Python/3.11.3-GCCcore-12.3.0
cd /home/users/u101139
source ./.bashrc
source /project/home/p200630/my_env/bin/activate

cd /home/users/u101139/D4H/scripts/

python3 laka_scraper.py
