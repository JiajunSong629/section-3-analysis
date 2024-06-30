#!/bin/bash

export PYTHONWARNINGS="ignore::FutureWarning"
MODEL_NAMES=("gpt2" "gpt2-xl" "llama2-7b" "falcon-7b" "mistral-7b" "olmo-7b")

for model in "${MODEL_NAMES[@]}";
do
    echo "SHUFFLE..ING $model"
    python shuffle.py --model_name=$model --n_exp=10 --method=diagonal
    python shuffle.py --model_name=$model --n_exp=10 --method=subspace
done