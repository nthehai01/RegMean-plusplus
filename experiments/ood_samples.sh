# !/bin/bash
export PYTHONPATH=$(pwd)


for model in "clip-vit-base-patch32" "clip-vit-base-patch16" "clip-vit-large-patch14"
do
    report_save_dir="outputs/experiments/ood_samples/${model}"
    mkdir -p ${report_save_dir}

    # For RegMean and RegMeanPlusPlus
    for method in "regmean" "regmean_plusplus"
    do
        python fusion_bench/scripts/cli.py \
            method=${method}/clip_regmean \
            modelpool=CLIPVisionModelPool/${model}_TA8_control_task \
            taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8 \
                taskpool.base_model=openai/${model} \
            report_save_path=${report_save_dir}/${method}.json
    done


    # For Fisher
    for method in "fisher_merging"
    do
        python fusion_bench/scripts/cli.py \
            method=fisher_merging/clip_fisher_merging \
            modelpool=CLIPVisionModelPool/${model}_TA8_control_task \
            taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8 \
                taskpool.base_model=openai/${model} \
            report_save_path=${report_save_dir}/${method}.json
    done
done
