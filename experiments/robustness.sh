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


for model in "clip-vit-base-patch32" "clip-vit-base-patch16" "clip-vit-large-patch14"
do
    # For RegMean and RegMeanPlusPlus
    for method in "regmean" "regmean_plusplus"
    do
        report_save_dir="outputs/experiments/robustness/${model}/${method}"
        mkdir -p ${report_save_dir}

        python fusion_bench/scripts/cli.py \
            method=${method}/clip_regmean \
            modelpool=CLIPVisionModelPool/${model}_robustness_clean \
            taskpool=${model}_robustness_clean \
            report_save_path=${report_save_dir}/clean.json

        for corruption in contrast gaussian_noise impulse_noise jpeg_compression motion_blur pixelate spatter
        do
            python fusion_bench/scripts/cli.py --config-name ${model}_robustness_corrupted \
                corruption=$corruption \
                method=${method}/clip_regmean \
                report_save_path=${report_save_dir}/${corruption}.json
        done
    done


    # For Model Soups, Task Arithmetic, and TIES-Merging
    for method in "simple_average" "task_arithmetic" "ties_merging"
    do
        report_save_dir="outputs/experiments/robustness/${model}/${method}"
        mkdir -p ${report_save_dir}

        python fusion_bench/scripts/cli.py \
            method=${method} \
            modelpool=CLIPVisionModelPool/${model}_robustness_clean \
            taskpool=${model}_robustness_clean \
            report_save_path=${report_save_dir}/clean.json

        for corruption in contrast gaussian_noise impulse_noise jpeg_compression motion_blur pixelate spatter
        do
            python fusion_bench/scripts/cli.py --config-name ${model}_robustness_corrupted \
                corruption=$corruption \
                method=${method} \
                report_save_path=${report_save_dir}/${corruption}.json
        done
    done


    # TSV-M
    for method in "tsv_m"
    do
        report_save_dir="outputs/experiments/robustness/${model}/${method}"
        mkdir -p ${report_save_dir}

        python fusion_bench/scripts/cli.py \
            method=task_singular_vector/TaskSingularVectorMerging \
            modelpool=CLIPVisionModelPool/${model}_robustness_clean \
            taskpool=${model}_robustness_clean \
            report_save_path=${report_save_dir}/clean.json

        for corruption in contrast gaussian_noise impulse_noise jpeg_compression motion_blur pixelate spatter
        do
            python fusion_bench/scripts/cli.py --config-name ${model}_robustness_corrupted \
                corruption=$corruption \
                method=task_singular_vector/TaskSingularVectorMerging \
                report_save_path=${report_save_dir}/${corruption}.json
        done
    done


    # For DOGE TA
    for method in "doge_ta"
    do
        report_save_dir="outputs/experiments/robustness/${model}/${method}"
        mkdir -p ${report_save_dir}

        python fusion_bench/scripts/cli.py \
            method=doge_ta/doge_ta \
            modelpool=CLIPVisionModelPool/${model}_robustness_clean \
            taskpool=${model}_robustness_clean \
            report_save_path=${report_save_dir}/clean.json

        for corruption in contrast gaussian_noise impulse_noise jpeg_compression motion_blur pixelate spatter
        do
            python fusion_bench/scripts/cli.py --config-name ${model}_robustness_corrupted \
                corruption=$corruption \
                method=doge_ta/doge_ta \
                report_save_path=${report_save_dir}/${corruption}.json
        done
    done


    # For ISO-C
    for method in "iso_c"
    do
        report_save_dir="outputs/experiments/robustness/${model}/${method}"
        mkdir -p ${report_save_dir}

        python fusion_bench/scripts/cli.py \
            method=isotropic_merging/${method} \
                method.scaling_factor=${iso_c_scaling_factor[$model]} \
            modelpool=CLIPVisionModelPool/${model}_robustness_clean \
            taskpool=${model}_robustness_clean \
            report_save_path=${report_save_dir}/clean.json

        for corruption in contrast gaussian_noise impulse_noise jpeg_compression motion_blur pixelate spatter
        do
            python fusion_bench/scripts/cli.py --config-name ${model}_robustness_corrupted \
                corruption=$corruption \
                method=isotropic_merging/${method} \
                    method.scaling_factor=${iso_c_scaling_factor[$model]} \
                report_save_path=${report_save_dir}/${corruption}.json
        done
    done


    # For ISO-CTS
    for method in "iso_cts"
    do
        report_save_dir="outputs/experiments/robustness/${model}/${method}"
        mkdir -p ${report_save_dir}

        python fusion_bench/scripts/cli.py \
            method=isotropic_merging/${method} \
                method.scaling_factor=${iso_cts_scaling_factor[$model]} \
            modelpool=CLIPVisionModelPool/${model}_robustness_clean \
            taskpool=${model}_robustness_clean \
            report_save_path=${report_save_dir}/clean.json

        for corruption in contrast gaussian_noise impulse_noise jpeg_compression motion_blur pixelate spatter
        do
            python fusion_bench/scripts/cli.py --config-name ${model}_robustness_corrupted \
                corruption=$corruption \
                method=isotropic_merging/${method} \
                    method.scaling_factor=${iso_cts_scaling_factor[$model]} \
                report_save_path=${report_save_dir}/${corruption}.json
        done
    done


    # For Fisher
    for method in "fisher_merging"
    do
        report_save_dir="outputs/experiments/robustness/${model}/${method}"
        mkdir -p ${report_save_dir}

        python fusion_bench/scripts/cli.py \
            method=fisher_merging/clip_fisher_merging \
            modelpool=CLIPVisionModelPool/${model}_robustness_clean \
            taskpool=${model}_robustness_clean \
            report_save_path=${report_save_dir}/clean.json

        for corruption in contrast gaussian_noise impulse_noise jpeg_compression motion_blur pixelate spatter
        do
            python fusion_bench/scripts/cli.py --config-name ${model}_robustness_corrupted \
                corruption=$corruption \
                method=fisher_merging/clip_fisher_merging \
                report_save_path=${report_save_dir}/${corruption}.json
        done
    done


    # AdaMerging
    for method in "adamerging"
    do
        report_save_dir="outputs/experiments/robustness/${model}/${method}"
        mkdir -p ${report_save_dir}


        python fusion_bench/scripts/cli.py \
            method=${method} \
                method.name=clip_layer_wise_adamerging \
            modelpool=CLIPVisionModelPool/${model}_robustness_clean \
            taskpool=${model}_robustness_clean \
            report_save_path=${report_save_dir}/clean.json

        for corruption in contrast gaussian_noise impulse_noise jpeg_compression motion_blur pixelate spatter
        do
            python fusion_bench/scripts/cli.py --config-name ${model}_robustness_corrupted \
                corruption=$corruption \
                method=${method} \
                    method.name=clip_layer_wise_adamerging \
                report_save_path=${report_save_dir}/${corruption}.json
        done
    done


    # DOGE AM
    for method in "doge_am"
    do
        report_save_dir="outputs/experiments/robustness/${model}/${method}"
        mkdir -p ${report_save_dir}


        python fusion_bench/scripts/cli.py \
            method=adamerging \
                method.name=clip_layer_wise_adamerging_doge_ta \
            modelpool=CLIPVisionModelPool/${model}_robustness_clean \
            taskpool=${model}_robustness_clean \
            report_save_path=${report_save_dir}/clean.json

        for corruption in contrast gaussian_noise impulse_noise jpeg_compression motion_blur pixelate spatter
        do
            python fusion_bench/scripts/cli.py --config-name ${model}_robustness_corrupted \
                corruption=$corruption \
                method=adamerging \
                    method.name=clip_layer_wise_adamerging_doge_ta \
                report_save_path=${report_save_dir}/${corruption}.json
        done
    done
done
