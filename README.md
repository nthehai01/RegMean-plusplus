# RegMean++: Enhancing Effectiveness and Generalization of Regression Mean for Model Merging

## Setup
Setup the environment:
```bash
conda create -n merging_env python=3.13.2
conda activate merging_env
pip install -r requirements.txt
```

Download the ImageNet to `calibration_data/` directory:
```bash
python experiment_utils/get_imagenet_data.py
```

*(Optional but recommended)* Set the Hugging Face token for downloading models and datasets:
```bash
export HF_TOKEN=<REPLACE_WITH_YOUR_HUGGINGFACE_TOKEN>
```


## Experiments
The results will be placed in the `outputs/experiments` directory.

- Main Results:
```bash
bash experiments/main_results.sh
```

- Sustainability to Large-Scale Tasks:
```bash
bash experiments/large_scale_tasks.sh
```

- Sequential Merging: 
```bash
bash experiments/sequential_merging.sh
```

- Out-of-Domain Generalization: 
```bash
bash experiments/generalization.sh
```

- Robustness Against Distribution Shifts:
```bash
bash experiments/robustness.sh
```

- Effects of Merging in Different Spaces:
```bash
bash experiments/region_specific.sh
bash experiments/layer_wise.sh
bash experiments/component_specific.sh
```

- Effects of Data Characteristics:
```bash
bash experiments/ood_samples.sh
bash experiments/number_of_samples.sh
bash experiments/class_imbalance.sh
```

- Hyperparameter (Scaling Factor $\alpha$):
```bash
bash experiments/hyperparam.sh
```

- Analysis on Representation Bias:

NOTE: DON'T Forget to OFF the `post_layernorm`'s effect in the vision model in the `HFCLIPClassifier` class (in `fusion_bench/models/hf_clip.py`) 

(1) Uncomment the following code snippet:

    with torch.no_grad():
        self.clip_model.vision_model.post_layernorm.weight.data.fill_(1.0)
        self.clip_model.vision_model.post_layernorm.bias.data.fill_(0.0)

(2) Comment out the embeddings normalization:

    image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)


Run experiments:
```bash
bash experiments/eight_task_merge.sh        # to get the merge models
bash experiments/extract_representation.sh  # to extract representations
python experiment_utils/extract_representation_bias.py  # to get the representation bias
```
