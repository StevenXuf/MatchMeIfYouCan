#!/bin/bash

models=("clip" "blip" "mclip" "mblip")
tasks=("img2txt" "txt2img")

for model in "${models[@]}"; do
    for task in "${tasks[@]}"; do
        echo "Running model=$model, task=$task"
        python cross_modal_retrieval.py "$model" -t "$task" &> "results_${model}_${task}.log"
    done
done
