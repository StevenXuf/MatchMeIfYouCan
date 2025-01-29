#!/bin/sh -l
#SBATCH --time=48:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu
#SBATCH --qos=default
#SBATCH --cpus-per-task=32
#SBATCH --account=p200630

module load Python/3.11.3-GCCcore-12.3.0
cd /home/users/u101139
source ./.bashrc
source /project/home/p200630/my_env/bin/activate

cd /home/users/u101139/D4H/scripts/

model='blip'
task='txt2img'

#python3 metrics_computation_poster.py $model -t $task
#python3 poster_manipulation.py
#python3 -u metrics_computation_poster.py > ./img2txt_clip_top10_test.log
python3 laka_scraper.py
