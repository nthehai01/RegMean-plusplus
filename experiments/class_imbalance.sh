# !/bin/bash
export PYTHONPATH=$(pwd)


for class_id in {0..4}
do
    export EXP_CLASS_ID=${class_id}
    
    for model in "clip-vit-base-patch32" "clip-vit-base-patch16" "clip-vit-large-patch14"
    do
        report_save_dir="outputs/experiments/class_imbalance/${model}/class_${class_id}"
        mkdir -p ${report_save_dir}

        # RegMean and RegMeanPlusPlus
        for method in "regmean" "regmean_plusplus"
        do
            python fusion_bench/scripts/cli.py \
                method=${method}/clip_regmean \
                modelpool=CLIPVisionModelPool/${model}_TA8 \
                taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8 \
                    taskpool.base_model=openai/${model} \
                report_save_path=${report_save_dir}/${method}.json
        done


        # Fisher Merging
        for method in "fisher_merging"
        do
            python fusion_bench/scripts/cli.py \
                method=fisher_merging/clip_fisher_merging \
                modelpool=CLIPVisionModelPool/${model}_TA8 \
                taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8 \
                    taskpool.base_model=openai/${model} \
                report_save_path=${report_save_dir}/${method}.json
        done
    done
    
    unset EXP_CLASS_ID
done

