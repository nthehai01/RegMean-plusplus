# !/bin/bash
export PYTHONPATH=$(pwd)



# For RegMean and RegMeanPlusPlus
for module in "mlp"
do
    for model in "clip-vit-base-patch32" "clip-vit-base-patch16" "clip-vit-large-patch14"
    do
        for method in "regmean" "regmean_plusplus"
        do
            report_save_dir="outputs/experiments/component_specific/${model}/${module}"
            mkdir -p ${report_save_dir}

            python fusion_bench/scripts/cli.py \
                method=${method}/clip_regmean \
                    method.exclude_param_names_regex=[".*self_attn.*"] \
                modelpool=CLIPVisionModelPool/${model}_TA8 \
                taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8 \
                    taskpool.base_model=openai/${model} \
                report_save_path=${report_save_dir}/${method}.json
        done
    done
done

for module in "self_attn"
do
    for model in "clip-vit-base-patch32" "clip-vit-base-patch16" "clip-vit-large-patch14"
    do
        for method in "regmean" "regmean_plusplus"
        do
            report_save_dir="outputs/experiments/component_specific/${model}/${module}"
            mkdir -p ${report_save_dir}

            python fusion_bench/scripts/cli.py \
                method=${method}/clip_regmean \
                    method.exclude_param_names_regex=[".*mlp.*"] \
                modelpool=CLIPVisionModelPool/${model}_TA8 \
                taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8 \
                    taskpool.base_model=openai/${model} \
                report_save_path=${report_save_dir}/${method}.json
        done
    done
done


# For other methods
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


for module in "self_attn" "mlp"
do
    export MODULE_TO_MERGE=${module}
    
    for model in "clip-vit-base-patch32" "clip-vit-base-patch16" "clip-vit-large-patch14"
    do
        report_save_dir="outputs/experiments/component_specific/${model}/${module}"
        mkdir -p ${report_save_dir}

    
        # For Task Arithmetic, and TIES-Merging
        for method in "task_arithmetic" "ties_merging"
        do
            python fusion_bench/scripts/cli.py \
                method=${method} \
                modelpool=CLIPVisionModelPool/${model}_TA8 \
                taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8 \
                    taskpool.base_model=openai/${model} \
                report_save_path=${report_save_dir}/${method}.json
        done


        # For TSV-M
        for method in "tsv_m"
        do
            python fusion_bench/scripts/cli.py \
                method=task_singular_vector/TaskSingularVectorMerging \
                modelpool=CLIPVisionModelPool/${model}_TA8 \
                taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8 \
                    taskpool.base_model=openai/${model} \
                report_save_path=${report_save_dir}/${method}.json
        done


        # For DOGE TA
        for method in "doge_ta"
        do
            python fusion_bench/scripts/cli.py \
                method=doge_ta/doge_ta \
                modelpool=CLIPVisionModelPool/${model}_TA8 \
                taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8 \
                    taskpool.base_model=openai/${model} \
                report_save_path=${report_save_dir}/${method}.json
        done


        # For ISO-C
        for method in "iso_c"
        do
            python fusion_bench/scripts/cli.py \
                method=isotropic_merging/${method} \
                    method.scaling_factor=${iso_c_scaling_factor[$model]} \
                modelpool=CLIPVisionModelPool/${model}_TA8 \
                taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8 \
                    taskpool.base_model=openai/${model} \
                report_save_path=${report_save_dir}/${method}.json
        done
        

        # For ISO-CTS
        for method in "iso_cts"
        do
            python fusion_bench/scripts/cli.py \
                method=isotropic_merging/${method} \
                    method.scaling_factor=${iso_cts_scaling_factor[$model]} \
                modelpool=CLIPVisionModelPool/${model}_TA8 \
                taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8 \
                    taskpool.base_model=openai/${model} \
                report_save_path=${report_save_dir}/${method}.json
        done
    done
    
    unset MODULE_TO_MERGE
done
