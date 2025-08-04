# RegMean++: Enhancing Effectiveness and Generalization of Regression Mean for Model Merging

## Abstract

*Regression Mean (RegMean), an approach that formulates model merging as a linear regression problem, aims to find the optimal weights for each linear layer in the merge model by minimizing the discrepancy in predictions between the merge and candidate models. RegMean provides a precise closed-form solution for the merging problem; therefore, it offers explainability and computational efficiency. However, RegMean merges each linear layer independently, overlooking how the features and information in the earlier layers propagate through the layers and influence the final prediction in the merge model. In this paper, we introduce RegMean++, a simple yet effective alternative to RegMean, that explicitly incorporates both intra- and cross-layer dependencies between merge models' layers into RegMean's objective. By accounting for these dependencies, RegMean++ better captures the behaviors of the merge model. Extensive experiments demonstrate that RegMean++ consistently outperforms RegMean across diverse settings, including in-domain (ID) and out-of-domain (OOD) generalization, sequential merging, large-scale tasks, and robustness under several types of distribution shifts. Furthermore, RegMean++ achieves competitive or state-of-the-art performance compared to various recent advanced model merging methods.*

## Code For Reproducing the Results in the Paper

### Setup

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


### Experiments

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

**NOTE:** DON'T Forget to OFF the `post_layernorm`'s effect in the vision model in the `HFCLIPClassifier` class (in `fusion_bench/models/hf_clip.py`) by:

(1) Uncomment the following code snippet:

    with torch.no_grad():
        self.clip_model.vision_model.post_layernorm.weight.data.fill_(1.0)
        self.clip_model.vision_model.post_layernorm.bias.data.fill_(0.0)

(2) Comment out the embeddings normalization:

    image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)


**Now, run experiments:**
```bash
bash experiments/eight_task_merge.sh        # to get the merge models
bash experiments/extract_representation.sh  # to extract representations
python experiment_utils/extract_representation_bias.py  # to get the representation bias
```
