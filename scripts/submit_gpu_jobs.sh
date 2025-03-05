#!/bin/sh -l
#SBATCH --time=48:00:00
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=2
#SBATCH --partition=gpu
#SBATCH --qos=default
#SBATCH --cpus-per-task=32
#SBATCH --account=p200630

module load env/release/2023.1
module load Python/3.11.3-GCCcore-12.3.0
cd /home/users/u101139
source ./.bashrc
source /project/home/p200630/my_env/bin/activate

cd /home/users/u101139/D4H/scripts/

model='blip'
task='txt2img'

MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
MASTER_PORT=$RANDOM

#python3 metrics_computation_poster.py $model -t $task
#python3 poster_manipulation.py
#python3 -u metrics_computation_poster.py > ./img2txt_clip_top10_test.log
#python3 laka_scraper.py
#python3 deduplicates.py
python3 text_manipulation.py
#python3 clean_text_dataset.py
#VISIBLE_CUDA_DEVICES=0 python3 cross_modal_retrieval.py clip -t img2txt > ./logs/clip_img2txt.log
#VISIBLE_CUDA_DEVICES=1 python3 cross_modal_retrieval.py clip -t txt2img > ./logs/clip_txt2img.log
#VISIBLE_CUDA_DEVICES=2 python3 cross_modal_retrieval.py blip -t img2txt > ./logs/blip_img2txt.log
#VISIBLE_CUDA_DEVICES=3 python3 cross_modal_retrieval.py blip -t txt2img > ./logs/blip_txt2img.log

#VISIBLE_CUDA_DEVICES=0 python3 plot_examples.py blip
#VISIBLE_CUDA_DEVICES=0 python3 plot_examples.py clip 

#VISIBLE_CUDA_DEVICES=0 python3 compare_with_openai.py 
