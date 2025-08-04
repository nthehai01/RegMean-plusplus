# !/bin/bash
export PYTHONPATH=$(pwd)


for reduce_non_diagonal_ratio in 1.00 0.95 0.90 0.85 0.80 0.75 0.70 0.65 0.60 0.55 0.50 0.45 0.40 0.35 0.30 0.25 0.20 0.15 0.10
do
    for model in "clip-vit-base-patch32" "clip-vit-base-patch16" "clip-vit-large-patch14"
    do
        for method in "regmean" "regmean_plusplus"
        do
            report_save_dir="outputs/experiments/hyperparam/${model}/${method}"
            mkdir -p ${report_save_dir}

            python fusion_bench/scripts/cli.py \
                method=${method}/clip_regmean \
                    method.reduce_non_diagonal_ratio=${reduce_non_diagonal_ratio} \
                modelpool=CLIPVisionModelPool/${model}_TA8 \
                taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8_val \
                    taskpool.base_model=openai/${model} \
                report_save_path=${report_save_dir}/${reduce_non_diagonal_ratio}.json
        done
    done
done
