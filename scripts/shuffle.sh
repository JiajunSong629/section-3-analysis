#!/bin/bash

export PYTHONWARNINGS="ignore::FutureWarning"
MODEL_NAMES=("gpt2" "gpt2-xl" "llama2-7b" "gemma-7b" "falcon-7b" "mistral-7b" "olmo-7b")


do
    echo "SHUFFLE..ING $model"
    python shuffle.py --model_name=$model --K=10 --n_exp=10
    python shuffle.py --model_name=$model --K=20 --n_exp=10
done