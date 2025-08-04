# !/bin/bash
export PYTHONPATH=$(pwd)


# For RegMean and RegMeanPlusPlus
for model in "clip-vit-base-patch32" "clip-vit-base-patch16" "clip-vit-large-patch14"
do
    for method in "regmean" "regmean_plusplus"
    do
        save_dir="outputs/eight_task_merge_checkpoints/${method}"
        mkdir -p ${save_dir}

        python fusion_bench/scripts/cli.py \
            method=${method}/clip_regmean \
            modelpool=CLIPVisionModelPool/${model}_TA8 \
            merged_model_save_path=${save_dir}/${model}
    done
done
