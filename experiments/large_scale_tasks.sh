# !/bin/bash
export PYTHONPATH=$(pwd)


# Hyperparameters for ISO-C
declare -A iso_c_scaling_factor
iso_c_scaling_factor["clip-vit-base-patch32"]=1.30
iso_c_scaling_factor["clip-vit-base-patch16"]=1.40
iso_c_scaling_factor["clip-vit-large-patch14"]=1.50

# Hyperparameters for ISO-CTS
declare -A iso_cts_scaling_factor
iso_cts_scaling_factor["clip-vit-base-patch32"]=1.50
iso_cts_scaling_factor["clip-vit-base-patch16"]=1.60
iso_cts_scaling_factor["clip-vit-large-patch14"]=1.90


for num_task in "TALL2" "TALL4" "TALL12" "TALL16" "TALL20"
do
    for model in "clip-vit-base-patch32" "clip-vit-base-patch16" "clip-vit-large-patch14"
    do
        report_save_dir="outputs/experiments/large_scale_tasks/${model}/${num_task}"
        mkdir -p ${report_save_dir}
        
        
        # For RegMean and RegMeanPlusPlus
        for method in "regmean" "regmean_plusplus"
        do
            python fusion_bench/scripts/cli.py \
                method=${method}/clip_regmean \
                modelpool=CLIPVisionModelPool/${model}_${num_task} \
                taskpool=CLIPVisionModelTaskPool/clip-vit-classification_${num_task} \
                    taskpool.base_model=openai/${model} \
                report_save_path=${report_save_dir}/${method}.json
        done

        
        # For Model Soups, Task Arithmetic, and TIES-Merging
        for method in "simple_average" "task_arithmetic" "ties_merging"
        do
            python fusion_bench/scripts/cli.py \
                method=${method} \
                modelpool=CLIPVisionModelPool/${model}_${num_task} \
                taskpool=CLIPVisionModelTaskPool/clip-vit-classification_${num_task} \
                    taskpool.base_model=openai/${model} \
                report_save_path=${report_save_dir}/${method}.json
        done

        
        # TSV-M
        for method in "tsv_m"
        do
            python fusion_bench/scripts/cli.py \
                method=task_singular_vector/TaskSingularVectorMerging \
                modelpool=CLIPVisionModelPool/${model}_${num_task} \
                taskpool=CLIPVisionModelTaskPool/clip-vit-classification_${num_task} \
                    taskpool.base_model=openai/${model} \
                report_save_path=${report_save_dir}/${method}.json
        done


        # For DOGE TA
        for method in "doge_ta"
        do
            python fusion_bench/scripts/cli.py \
                method=doge_ta/doge_ta \
                modelpool=CLIPVisionModelPool/${model}_${num_task} \
                taskpool=CLIPVisionModelTaskPool/clip-vit-classification_${num_task} \
                    taskpool.base_model=openai/${model} \
                report_save_path=${report_save_dir}/${method}.json
        done


        # For ISO-C
        for method in "iso_c"
        do
            python fusion_bench/scripts/cli.py \
                method=isotropic_merging/${method} \
                    method.scaling_factor=${iso_c_scaling_factor[$model]} \
                modelpool=CLIPVisionModelPool/${model}_${num_task} \
                taskpool=CLIPVisionModelTaskPool/clip-vit-classification_${num_task} \
                    taskpool.base_model=openai/${model} \
                report_save_path=${report_save_dir}/${method}.json
        done


        # For ISO-CTS
        for method in "iso_cts"
        do
            python fusion_bench/scripts/cli.py \
                method=isotropic_merging/${method} \
                    method.scaling_factor=${iso_cts_scaling_factor[$model]} \
                modelpool=CLIPVisionModelPool/${model}_${num_task} \
                taskpool=CLIPVisionModelTaskPool/clip-vit-classification_${num_task} \
                    taskpool.base_model=openai/${model} \
                report_save_path=${report_save_dir}/${method}.json
        done


        # For Fisher
        for method in "fisher_merging"
        do
            python fusion_bench/scripts/cli.py \
                method=fisher_merging/clip_fisher_merging \
                modelpool=CLIPVisionModelPool/${model}_${num_task} \
                taskpool=CLIPVisionModelTaskPool/clip-vit-classification_${num_task} \
                    taskpool.base_model=openai/${model} \
                report_save_path=${report_save_dir}/${method}.json
        done


        # For AdaMerging
        for method in "adamerging"
        do
            python fusion_bench/scripts/cli.py \
                method=${method} \
                    method.name=clip_layer_wise_adamerging \
                modelpool=CLIPVisionModelPool/${model}_${num_task} \
                taskpool=CLIPVisionModelTaskPool/clip-vit-classification_${num_task} \
                    taskpool.base_model=openai/${model} \
                report_save_path=${report_save_dir}/${method}.json
        done


        # For DOGE AM
        for method in "doge_am"
        do
            python fusion_bench/scripts/cli.py \
                method=adamerging \
                    method.name=clip_layer_wise_adamerging_doge_ta \
                modelpool=CLIPVisionModelPool/${model}_${num_task} \
                taskpool=CLIPVisionModelTaskPool/clip-vit-classification_${num_task} \
                    taskpool.base_model=openai/${model} \
                report_save_path=${report_save_dir}/${method}.json
        done
    done
done
