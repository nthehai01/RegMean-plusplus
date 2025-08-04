from collections import defaultdict
import pickle

import torch.nn as nn


models = ["clip-vit-base-patch32", 
          "clip-vit-base-patch16", 
          "clip-vit-large-patch14"]

datasets_names = ["tanganke/sun397",
				  "tanganke/stanford_cars",
				  "tanganke/resisc45",
				  "tanganke/eurosat",
				  "ufldl-stanford/svhn",
				  "tanganke/gtsrb",
				  "mnist",
				  "tanganke/dtd"]


loss = nn.L1Loss()
data = defaultdict(list)
for model_name in models: 
    representation_dir = f"outputs/experiments/representations/{model_name}"
    for dataset_name in datasets_names:
        task_name = dataset_name.split('/')[-1]\
                                .replace('_', '-')
        
        with open(f"{representation_dir}/{task_name}/individual.pkl", 'rb') as file:
            individual = pickle.load(file)
        with open(f"{representation_dir}/{task_name}/regmean.pkl", 'rb') as file:
            regmean = pickle.load(file)
        with open(f"{representation_dir}/{task_name}/regmean_plusplus.pkl", 'rb') as file:
            regmean_plusplus = pickle.load(file)

        data[model_name].append({
            "task_name": task_name,
            "regmean":          loss(regmean, individual).item(),
            "regmean_plusplus": loss(regmean_plusplus, individual).item()
        })

print(data)
