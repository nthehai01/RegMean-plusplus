import os
from datasets import load_dataset, Dataset


os.makedirs("calibration_data", exist_ok=True)


dataset = load_dataset("ILSVRC/imagenet-1k", split="train", streaming=True, trust_remote_code=True)
dataset = Dataset.from_list(list(dataset.take(10000)))

dataset = dataset.shuffle(42)
dataset.save_to_disk("calibration_data/ILSVRC--imagenet-1k.first10ksamples", num_shards=1)
