# !/bin/bash
export PYTHONPATH=$(pwd)


CONFIG_DIR="config"
CHECKPOINT_DIR="outputs/checkpoints"
REPORT_DIR="outputs/experiments/sequential_merging"
CALIBRATION_DATA_NAME="imagenet"  # this is just a surrogate name, "imagenet" dataset is not used in this experiment

mkdir -p $CHECKPOINT_DIR


generate_dataset_config() {
    local split=$1
    local -n prev_model=$2
    local -n new_tasks=$3
    local pair_index=$4

    local output_file="$CONFIG_DIR/dataset/image_classification/${split}/sequential_merging${pair_index}.yaml"
    > "$output_file"

    cat >> "$output_file" <<EOF
defaults:
EOF

    if [ -n "$prev_model" ]
    then
        echo "  - $CALIBRATION_DATA_NAME" >> "$output_file"
    fi

    for task in "${new_tasks[@]}"
    do
        echo "  - ${task}" >> "$output_file"
    done
}


generate_model_config() {
    local -n pretrained_model=$1
    local -n prev_model=$2
    local -n new_tasks=$3
    local pair_index=$4

    if [ -n "$prev_model" ]
    then
        local model_card="$CONFIG_DIR/model/clip-vit/${pretrained_model}_previous_merged_model.yaml"
        > "$model_card"

        cat >> "$model_card" <<EOF
$CALIBRATION_DATA_NAME: $prev_model
EOF
    fi

    local output_file="$CONFIG_DIR/model/clip-vit/${pretrained_model}_sequential_merging${pair_index}.yaml"
    > "$output_file"

    cat >> "$output_file" <<EOF
defaults:
  # pre-trained model
  - $pretrained_model
  # candidates
EOF

    if [ -n "$prev_model" ]
    then
        echo "  - ${pretrained_model}_previous_merged_model" >> "$output_file"
    fi

    for task in "${new_tasks[@]}"
    do
        echo "  - ${pretrained_model}_${task}" >> "$output_file"
    done
}


generate_modelpool_config() {
    local -n pretrained_model=$1
    local pair_index=$2

    local output_file="$CONFIG_DIR/modelpool/CLIPVisionModelPool/${pretrained_model}_sequential_merging${pair_index}.yaml"
    > "$output_file"

    cat >> "$output_file" <<EOF
defaults:
  - CLIPVisionModelPool@: _template
  - /model/clip-vit@models: ${pretrained_model}_sequential_merging${pair_index}
  - /dataset/image_classification/train@train_datasets: sequential_merging${pair_index}
processor:
  _target_: transformers.CLIPProcessor.from_pretrained
  pretrained_model_name_or_path: openai/${pretrained_model}
EOF

    if [ -n "$prev_model" ]
    then
        echo "  - $CALIBRATION_DATA_NAME" >> "$output_file"
    fi

    for task in "${new_tasks[@]}"
    do
        echo "  - ${task}" >> "$output_file"
    done
}


seed0=(pcam fer2013 oxford-iiit-pet rendered-sst2 gtsrb fashion_mnist sun397 cifar100 eurosat stanford-cars mnist stl10 dtd oxford_flowers102 cifar10 food101 kmnist emnist_letters svhn resisc45)
seed1=(cifar100 sun397 emnist_letters eurosat resisc45 food101 oxford_flowers102 pcam rendered-sst2 stanford-cars cifar10 gtsrb mnist dtd kmnist fashion_mnist stl10 svhn oxford-iiit-pet fer2013)
seed2=(eurosat rendered-sst2 sun397 fashion_mnist food101 kmnist oxford-iiit-pet dtd pcam fer2013 oxford_flowers102 mnist resisc45 stanford-cars cifar10 stl10 gtsrb emnist_letters svhn cifar100)
seed3=(emnist_letters resisc45 mnist cifar10 fashion_mnist svhn kmnist stl10 gtsrb eurosat sun397 pcam oxford_flowers102 fer2013 oxford-iiit-pet food101 dtd rendered-sst2 stanford-cars cifar100)
seed4=(gtsrb stanford-cars sun397 fashion_mnist cifar10 emnist_letters svhn fer2013 oxford-iiit-pet food101 mnist rendered-sst2 dtd cifar100 oxford_flowers102 pcam kmnist stl10 eurosat resisc45)


task_list_raw=(seed0 seed1 seed2 seed3 seed4)

# For RegMean RegMeanPlusPlus
for method in "regmean" "regmean_plusplus" 
do
    for model in "clip-vit-base-patch32" "clip-vit-base-patch16" "clip-vit-large-patch14"
    do
        previous_merged_model=""
        merged_tasks=()

        for seed in 0 1 2 3 4
        do
            array_name=${task_list_raw[$seed]}
            eval "task_list=(\"\${${array_name}[@]}\")"

            for ((pair_index = 0; pair_index < ${#task_list[@]} - 1; pair_index+=4)); do
                task_pair=("${task_list[$pair_index]}" "${task_list[$pair_index+1]}" "${task_list[$pair_index+2]}" "${task_list[$pair_index+3]}")
                
                # Create configs
                generate_dataset_config "train" previous_merged_model task_pair "$pair_index"
                
                generate_model_config model previous_merged_model task_pair "$pair_index"
                generate_modelpool_config model "$pair_index"

                # Run
                report_save_dir="${REPORT_DIR}/${model}/seed_${seed}/${pair_index}"
                mkdir -p ${report_save_dir}

                merged_model_save_path="${CHECKPOINT_DIR}/${model}_sequential_merging/${pair_index}"

                if [ -n "$merged_tasks" ]
                then
                    joined=$(IFS=:; echo "${merged_tasks[*]}")
                    export MERGED_TASKS="$joined"
                fi

                python fusion_bench/scripts/cli.py \
                    method=${method}/clip_regmean \
                    modelpool=CLIPVisionModelPool/${model}_sequential_merging${pair_index} \
                    merged_model_save_path=${merged_model_save_path} \
                    taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TALL20 \
                        taskpool.base_model=openai/${model} \
                    report_save_path=${report_save_dir}/${method}.json


                unset MERGED_TASKS

                previous_merged_model="${merged_model_save_path}"
                merged_tasks+=("${task_pair[@]}")
            done
        done
    done
done
