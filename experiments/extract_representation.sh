# !/bin/bash
export PYTHONPATH=$(pwd)


# Please check the NOTE in `experiment_utils/extract_representation.py` first
for model in "clip-vit-base-patch32" "clip-vit-base-patch16" "clip-vit-large-patch14"
do
    mkdir -p "outputs/experiments/representations/${model}"
    
    python experiment_utils/extract_representation.py --model-name ${model}
done
