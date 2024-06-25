#!/bin/bash

export PYTHONWARNINGS="ignore::FutureWarning"
MODEL_NAMES=("llama2-7b" "mistral-7b")

for model in "${MODEL_NAMES[@]}";
do
    echo "PROJECT..ING $model"
    python project.py --model_name=$model --component="QK" --proj_out=True
    python project.py --model_name=$model --component="QK" --proj_out=False
    python project.py --model_name=$model --component="OV" --proj_out=True
    python project.py --model_name=$model --component="OV" --proj_out=False

done