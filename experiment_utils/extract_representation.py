"""
NOTE: DON'T Forget to OFF the post_layernorm in the vision model by 

(1) filling the following code snippet for the `HFCLIPClassifier` class:
with torch.no_grad():
	self.clip_model.vision_model.post_layernorm.weight.data.fill_(1.0)
	self.clip_model.vision_model.post_layernorm.bias.data.fill_(0.0)

(2) turn off the `normalize embeddings`:
# image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
"""


import argparse
import os
import pickle
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from datasets import load_dataset
from transformers import CLIPProcessor, CLIPVisionModel, CLIPModel

from fusion_bench.dataset import CLIPDataset
from fusion_bench.models.hf_clip import HFCLIPClassifier
from fusion_bench.tasks.clip_classification import get_classnames_and_templates


def get_representation(candidate_path, dataset_name, task_name, model_name, output_path):
	base_model_path = f'openai/{model_name}'

	vision_model = CLIPVisionModel.from_pretrained(candidate_path, device_map='auto')
	clip_model = CLIPModel.from_pretrained(base_model_path, device_map='auto')
	clip_model.vision_model.load_state_dict(vision_model.vision_model.state_dict())

	clip_processor = CLIPProcessor.from_pretrained(base_model_path, local_files_only=True)
	clip_model = HFCLIPClassifier(clip_model, processor=clip_processor)

	classnames, templates = get_classnames_and_templates(task_name)
	clip_model.set_classification_task(classnames, templates)

	if "svhn" in dataset_name:
		test_dataset = load_dataset(dataset_name, "cropped_digits", split='test')
	else:
		test_dataset = load_dataset(dataset_name, split='test')
	test_dataset = CLIPDataset(test_dataset, clip_processor)
	batch_size = 64 if "clip-vit-large-patch14" in model_name else 128
	test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size, num_workers=8, pin_memory=True, drop_last=False)

	representations = []
	for batch in tqdm(test_loader):
		inputs, _ = batch
		inputs = inputs.to("cuda")
		outputs = clip_model(
			inputs,
			return_image_embeds=True,
			return_dict=True,
			task_name=task_name,
		)
		logits = outputs["image_embeds"].detach().cpu()
		representations.append(logits)

		del outputs

	representations = torch.cat(representations, dim=0)
	with open(output_path, 'wb') as handle:
		pickle.dump(representations, handle, protocol=pickle.HIGHEST_PROTOCOL)

	del vision_model
	del clip_model
	torch.cuda.empty_cache()

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--model-name", type=str, default="clip-vit-base-patch32",)
	args = parser.parse_args()

	output_dir = "outputs/experiments/representations"
	model_name = args.model_name
	datasets_names = ["tanganke/sun397",
					  "tanganke/stanford_cars",
					  "tanganke/resisc45",
					  "tanganke/eurosat",
					  "ufldl-stanford/svhn",
					  "tanganke/gtsrb",
					  "mnist",
					  "tanganke/dtd"]
	for dataset_name in datasets_names:
		task_name = dataset_name.split('/')[-1]\
								.replace('_', '-')
		individual_path = f'tanganke/{model_name}_{task_name}'
		regmean_model_path = f'outputs/eight_task_merge_checkpoints/regmean/{model_name}'
		regmean_plusplus_model_path = f'outputs/eight_task_merge_checkpoints/regmean_plusplus/{model_name}'

		os.makedirs(f"{output_dir}/{model_name}/{task_name}")

		get_representation(individual_path, 
						dataset_name, 
						task_name, 
						model_name,
						f"{output_dir}/{model_name}/{task_name}/individual.pkl")
		get_representation(regmean_model_path, 
						dataset_name, 
						task_name, 
						model_name,
						f"{output_dir}/{model_name}/{task_name}/regmean.pkl")
		get_representation(regmean_plusplus_model_path, 
						dataset_name, 
						task_name, 
						model_name,
						f"{output_dir}/{model_name}/{task_name}/regmean_plusplus.pkl")
