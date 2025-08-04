from typing import List

import torch

from fusion_bench import BaseAlgorithm, BaseModelPool
from fusion_bench.mixins import LightningFabricMixin
from fusion_bench.utils.state_dict_arithmetic import (
    state_dict_add,
    state_dict_mul,
    state_dict_sub,
)

from .iso_utils import check_parameterNamesMatch, iso_c, iso_cts


class IsotropicMergingInCommonSubspace(BaseAlgorithm, LightningFabricMixin):
    """
    Isotropic Merging in Common Subspace (Iso-C)
    """

    def __init__(
        self,
        scaling_factor: float,
        exclude_keys: List[str] = None,
    ):
        self.scaling_factor = scaling_factor
        self.exclude_keys = exclude_keys
        super().__init__()

    def run(self, modelpool: BaseModelPool):
        # load the pretrained model and the task vectors of all the finetuned models
        with torch.no_grad():
            pretrained_model = modelpool.load_pretrained_model()
            task_vectors = []
            for model_name in modelpool.model_names:
                finetuned_model = modelpool.load_model(model_name)
                task_vectors.append(
                    state_dict_sub(
                        finetuned_model.state_dict(), pretrained_model.state_dict()
                    )
                )
                del finetuned_model  # free memory
            check_parameterNamesMatch(task_vectors)

        # compute the merged task vector
        merged_tv = iso_c(
            task_vectors,
            accelerator=self.fabric.device,
            exclude_keys=self.exclude_keys,
        )

        # merged_parameters = pretrained_parameters + scaling_factor * merged_task_vector
        merged_state_dict = state_dict_add(
            pretrained_model.state_dict(),
            state_dict_mul(merged_tv, self.scaling_factor),
        )


        ##################### ONLY FOR EXPERIMENTS #####################

        import os
        import re
        from fusion_bench.method.simple_average import SimpleAverageAlgorithm

        # 0. Calculate the Model Soups
        state_dict_average = SimpleAverageAlgorithm().run(modelpool).state_dict()

        
        # Region-specific Merging & Layer-wise Merging
        layer_start, layer_end = None, None

        layer_i = os.getenv("LAYER_ID", None)
        if layer_i is not None: layer_start, layer_end = int(layer_i), int(layer_i)

        layer_range = os.getenv("MERGE_LAYER_RANGE", None)
        num_layers = len(modelpool.load_pretrained_model().vision_model.encoder.layers)
        if layer_range is not None:
            if layer_range == "early":
                layer_start = 0 * num_layers/3
                layer_end = 1 * num_layers/3 - 1
            elif layer_range == "middle":
                layer_start = 1 * num_layers/3
                layer_end = 2 * num_layers/3 - 1
            elif layer_range == "late":
                layer_start = 2 * num_layers/3
                layer_end = num_layers - 1
            elif layer_range == "middle+late":
                layer_start = 1 * num_layers/3
                layer_end = num_layers - 1
            layer_start = int(layer_start)
            layer_end = int(layer_end)

        if layer_start is not None and layer_end is not None:
            result_state_dict = {}
            for key, value in merged_state_dict.items():
                match = re.search(r'layers\.(\d+)\.', key)
                if match is not None and (int(match.group(1)) >= layer_start and int(match.group(1)) <= layer_end):
                    result_state_dict[key] = value
                else:
                    result_state_dict[key] = state_dict_average[key]
            
            merged_state_dict = result_state_dict


        # Component-specific Merging
        module_to_merge = os.getenv("MODULE_TO_MERGE", None)
        if module_to_merge is not None:
            result_state_dict = {}
            for key, value in merged_state_dict.items():
                if module_to_merge in key:
                    result_state_dict[key] = value
                    print(key)
                else:
                    result_state_dict[key] = state_dict_average[key]
            
            merged_state_dict = result_state_dict

        

        ##################### ONLY FOR EXPERIMENTS #####################


        pretrained_model.load_state_dict(merged_state_dict)

        return pretrained_model


class IsotropicMergingInCommonAndTaskSubspace(BaseAlgorithm, LightningFabricMixin):
    """
    Isotropic Merging in Common and Task-Specific Subspaces (Iso-CTS)
    """

    def __init__(
        self,
        scaling_factor: float,
        common_space_fraction: float,
        exclude_keys: List[str] = None,
    ):
        self.common_space_fraction = common_space_fraction
        self.scaling_factor = scaling_factor
        self.exclude_keys = exclude_keys
        super().__init__()

    def run(self, modelpool: BaseModelPool):
        # load the pretrained model and the task vectors of all the finetuned models
        with torch.no_grad():
            pretrained_model = modelpool.load_pretrained_model()
            task_vectors = []
            for model_name in modelpool.model_names:
                finetuned_model = modelpool.load_model(model_name)
                task_vectors.append(
                    state_dict_sub(
                        finetuned_model.state_dict(), pretrained_model.state_dict()
                    )
                )
                del finetuned_model  # free memory
            check_parameterNamesMatch(task_vectors)

        # compute the merged task vector
        merged_tv = iso_cts(
            task_vectors,
            common_space_fraction=self.common_space_fraction,
            accelerator=self.fabric.device,
            exclude_keys=self.exclude_keys,
        )

        # merged_parameters = pretrained_parameters + scaling_factor * merged_task_vector
        merged_state_dict = state_dict_add(
            pretrained_model.state_dict(),
            state_dict_mul(merged_tv, self.scaling_factor),
        )


        ##################### ONLY FOR EXPERIMENTS #####################

        import os
        import re
        from fusion_bench.method.simple_average import SimpleAverageAlgorithm

        # 0. Calculate the Model Soups
        state_dict_average = SimpleAverageAlgorithm().run(modelpool).state_dict()

        
        # Region-specific Merging & Layer-wise Merging
        layer_start, layer_end = None, None

        layer_i = os.getenv("LAYER_ID", None)
        if layer_i is not None: layer_start, layer_end = int(layer_i), int(layer_i)

        layer_range = os.getenv("MERGE_LAYER_RANGE", None)
        num_layers = len(modelpool.load_pretrained_model().vision_model.encoder.layers)
        if layer_range is not None:
            if layer_range == "early":
                layer_start = 0 * num_layers/3
                layer_end = 1 * num_layers/3 - 1
            elif layer_range == "middle":
                layer_start = 1 * num_layers/3
                layer_end = 2 * num_layers/3 - 1
            elif layer_range == "late":
                layer_start = 2 * num_layers/3
                layer_end = num_layers - 1
            elif layer_range == "middle+late":
                layer_start = 1 * num_layers/3
                layer_end = num_layers - 1
            layer_start = int(layer_start)
            layer_end = int(layer_end)

        if layer_start is not None and layer_end is not None:
            result_state_dict = {}
            for key, value in merged_state_dict.items():
                match = re.search(r'layers\.(\d+)\.', key)
                if match is not None and (int(match.group(1)) >= layer_start and int(match.group(1)) <= layer_end):
                    result_state_dict[key] = value
                else:
                    result_state_dict[key] = state_dict_average[key]
            
            merged_state_dict = result_state_dict


        # Component-specific Merging
        module_to_merge = os.getenv("MODULE_TO_MERGE", None)
        if module_to_merge is not None:
            result_state_dict = {}
            for key, value in merged_state_dict.items():
                if module_to_merge in key:
                    result_state_dict[key] = value
                    print(key)
                else:
                    result_state_dict[key] = state_dict_average[key]
            
            merged_state_dict = result_state_dict

        

        ##################### ONLY FOR EXPERIMENTS #####################

        
        pretrained_model.load_state_dict(merged_state_dict)

        return pretrained_model


ISO_C_Merge = IsotropicMergingInCommonSubspace  # alias
ISO_CTS_Merge = IsotropicMergingInCommonAndTaskSubspace  # alias
