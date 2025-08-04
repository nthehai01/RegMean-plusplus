# !/bin/bash
export PYTHONPATH=$(pwd)


for num_samples in 1 2 4 8 16 32 64 128 512 1024
do
    for model in "clip-vit-base-patch32" "clip-vit-base-patch16" "clip-vit-large-patch14"
    do
        report_save_dir="outputs/experiments/number_of_samples/${model}/${num_samples}"
        mkdir -p ${report_save_dir}

        
        # For RegMean and RegMeanPlusPlus
        for method in "regmean" "regmean_plusplus"
        do
            python fusion_bench/scripts/cli.py \
                method=${method}/clip_regmean \
                    method.num_regmean_examples=${num_samples} \
                modelpool=CLIPVisionModelPool/${model}_TA8 \
                taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8 \
                    taskpool.base_model=openai/${model} \
                report_save_path=${report_save_dir}/${method}.json
        done


        # For Fisher
        for method in "fisher_merging"
        do
            python fusion_bench/scripts/cli.py \
                method=fisher_merging/clip_fisher_merging \
                    method.num_fisher_examples=${num_samples} \
                modelpool=CLIPVisionModelPool/${model}_TA8 \
                taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8 \
                    taskpool.base_model=openai/${model} \
                report_save_path=${report_save_dir}/${method}.json
        done
    done
done



for num_samples in "all"
do
    for model in "clip-vit-base-patch32" "clip-vit-base-patch16" "clip-vit-large-patch14"
    do
        report_save_dir="outputs/experiments/number_of_samples/${model}/${num_samples}"
        mkdir -p ${report_save_dir}

        
        # For RegMean and RegMeanPlusPlus
        for method in "regmean" "regmean_plusplus"
        do
            python fusion_bench/scripts/cli.py \
                method=${method}/clip_regmean \
                    method.num_regmean_examples=20000 \
                modelpool=CLIPVisionModelPool/${model}_TA8 \
                taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8 \
                    taskpool.base_model=openai/${model} \
                report_save_path=${report_save_dir}/${method}.json
        done


        # For Fisher
        for method in "fisher_merging"
        do
            python fusion_bench/scripts/cli.py \
                method=fisher_merging/clip_fisher_merging \
                    method.num_fisher_examples=20000 \
                modelpool=CLIPVisionModelPool/${model}_TA8 \
                taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8 \
                    taskpool.base_model=openai/${model} \
                report_save_path=${report_save_dir}/${method}.json
        done
    done
done
