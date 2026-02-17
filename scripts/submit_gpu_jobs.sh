#!/bin/sh -l
#SBATCH --time=48:00:00
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu
#SBATCH --qos=default
#SBATCH --cpus-per-task=48
#SBATCH --account=p200630

module load env/release/2023.1
module load Python/3.11.3-GCCcore-12.3.0
cd /home/users/u101139
source ./.bashrc
source /project/home/p200630/my_env/bin/activate

cd /home/users/u101139/NuclearAlignment/scripts/

system_role='translator'
task='txt2img'
model_name='mblip'

MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
MASTER_PORT=$RANDOM

# VISIBLE_CUDA_DEVICES=0 python3 cross_modal_retrieval.py --task $task --system_role editor --model_name mblip --meta true &> ./logs/${task}_meta_mblip.log &
# VISIBLE_CUDA_DEVICES=1 python3 cross_modal_retrieval.py --task $task --system_role editor --model_name mblip &> ./logs/${task}_editor_mblip.log &
# VISIBLE_CUDA_DEVICES=2 python3 cross_modal_retrieval.py --task $task --system_role translator --model_name mblip &> ./logs/${task}_translator_mblip.log &
# VISIBLE_CUDA_DEVICES=3 python3 cross_modal_retrieval.py --task $task --system_role summarizer --model_name mblip &> ./logs/${task}_summarizer_mblip.log &
VISIBLE_CUDA_DEVICES=0 python3 plot_examples.py clip --seed 42 &
VISIBLE_CUDA_DEVICES=1 python3 plot_examples.py blip --seed 42 &
VISIBLE_CUDA_DEVICES=2 python3 plot_examples.py clip --seed 0 &
VISIBLE_CUDA_DEVICES=3 python3 plot_examples.py blip --seed 0 &

wait
echo "All tasks completed."