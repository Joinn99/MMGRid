from collections import defaultdict, OrderedDict
from tqdm import tqdm
import copy
import torch
import torch.nn as nn

from .task_vector import TaskVector
from .utils import get_param_names_to_merge
from .mask_weights_utils import mask_model_weights


class ModelMergingMethod:
    """Base class for model merging methods."""
    
    def __init__(self):
        """Initialize the merging method."""
        pass

    def copy_params_to_model(self, params: dict, model: nn.Module):
        """
        Copy parameters in "params" to the model
        :param params: dict, dictionary of parameters
        :param model: nn.Module, model that needs to copy parameters
        :return:
        """
        for param_name, param_value in model.named_parameters():
            if param_name in params:
                param_value.data.copy_(params[param_name])

    def merge(self, merged_model: nn.Module, models_to_merge: list, exclude_param_names_regex: list, **kwargs):
        """
        Abstract method for merging models
        :param merged_model: nn.Module, the merged model
        :param models_to_merge: list, individual models that need to be merged
        :param exclude_param_names_regex: list, regular expression of names of parameters that need to be excluded
        :param kwargs: additional arguments specific to each merging method
        :return: dict, merged parameters
        """
        raise NotImplementedError("Subclasses must implement merge method")

    def get_merged_model(self, merged_model: nn.Module, models_to_merge: list, exclude_param_names_regex: list, **kwargs):
        """
        Merge the parameters of models_to_merge to merged_model
        :param merged_model: nn.Module, the merged model
        :param models_to_merge: list, individual models that need to be merged
        :param exclude_param_names_regex: list, regular expression of names of parameters that need to be excluded
        :param kwargs: additional arguments specific to each merging method
        :return: nn.Module, the merged model
        """
        merged_params = self.merge(merged_model=merged_model, models_to_merge=models_to_merge, 
                                  exclude_param_names_regex=exclude_param_names_regex, **kwargs)
        self.copy_params_to_model(params=merged_params, model=merged_model)
        return merged_model


class AverageMerging(ModelMergingMethod):
    """
    Average merging method - simple averaging of model parameters.
    This corresponds to basic weight averaging (WA).
    """
    
    def merge(self, merged_model: nn.Module, models_to_merge: list, exclude_param_names_regex: list, **kwargs):
        """
        Average merging method
        :param merged_model: nn.Module, the merged model
        :param models_to_merge: list, individual models that need to be merged
        :param exclude_param_names_regex: list, regular expression of names of parameters that need to be excluded
        :return: dict, averaged parameters
        """
        # dictionary of list, where key is the parameter name,
        # value is a list of the corresponding parameters of all the models that need to be merged
        models_to_merge_param_dict = defaultdict(list)
        
        # iterate each individual model that needs to be merged
        for model_to_merge in models_to_merge:
            param_dict = {param_name: param_value for param_name, param_value in model_to_merge.named_parameters()}
            # exclude parameter whose name matches element in exclude_param_names_regex
            param_names_to_merge = get_param_names_to_merge(input_param_names=list(param_dict.keys()), 
                                                           exclude_param_names_regex=exclude_param_names_regex)
            for param_name in param_names_to_merge:
                models_to_merge_param_dict[param_name].append(param_dict[param_name])

        with torch.no_grad():
            # average merging of individual models' parameters
            averaged_params = {param_name: torch.stack(model_to_merge_param, dim=0).mean(dim=0) 
                              for param_name, model_to_merge_param in models_to_merge_param_dict.items()}

        return averaged_params


from .fuser import TemporalModelFuser

class TaskArithmeticMerging(ModelMergingMethod):
    """
    Task arithmetic merging method - using task vectors with scaling coefficient.
    Calculates parameter deltas from a shared base model and combines them.
    """

    def merge(self, merged_model: nn.Module, models_to_merge: list, exclude_param_names_regex: list, 
              scaling_coefficient: float = 0.05, source_coeff: float = 0.5, **kwargs):
        """
        Task arithmetic method using TemporalModelFuser for extrapolation.
        """
        assert isinstance(scaling_coefficient, float), "wrong type of scaling_coefficient, should be float!"

        fuser = TemporalModelFuser(base_model=merged_model)

        with torch.no_grad():
            merged_params = fuser.extrapolate_linear_velocity(
                checkpoints=[merged_model] + models_to_merge[::-1], scaling_coefficient=scaling_coefficient
            )
        return merged_params


class TiesMerging(ModelMergingMethod):
    """
    TIES merging method (Trim, Elect Sign, and Merge).
    Resolves interference when merging models by filtering small values and aligning signs.
    """
    
    def merge(self, merged_model: nn.Module, models_to_merge: list, exclude_param_names_regex: list, 
              param_value_mask_rate: float = 0.8, scaling_coefficient: float = 1.0, source_coeff: float = 0.5, **kwargs):
        """
        TIES merging implementation.
        """

        def task_vector_param_dict_to_single_vector(task_vector: TaskVector):
            """
            Convert parameter dictionary in task vector to a single vector
            :param task_vector: TaskVector, task vector
            :return: torch.Tensor, flattened parameters
            """
            task_vector_param_dict = copy.deepcopy(task_vector.task_vector_param_dict)
            sorted_task_vector_param_dict = OrderedDict(sorted(task_vector_param_dict.items()))

            # Tensor, shape (num_total_params, )
            return nn.utils.parameters_to_vector([param.flatten() for param in sorted_task_vector_param_dict.values()])

        def single_vector_to_task_vector_param_dict(single_vector: torch.Tensor, task_vector: TaskVector):
            """
            Convert a single vector to parameter dictionary in task vector
            :param single_vector: Tensor, single vector that contain all parameters in task_vector.task_vector_param_dict
            :param task_vector: TaskVector, task vector
            :return: dict, parameter dictionary
            """
            task_vector_param_dict = copy.deepcopy(task_vector.task_vector_param_dict)
            sorted_task_vector_param_dict = OrderedDict(sorted(task_vector_param_dict.items()))

            nn.utils.vector_to_parameters(single_vector, sorted_task_vector_param_dict.values())

            return sorted_task_vector_param_dict

        def mask_smallest_magnitude_param_values(flattened_models_to_merge_param: torch.Tensor, param_value_mask_rate: float = 0.8):
            """
            Mask the smallest-magnitude parameter values (set to zeros) based on parameter value mask rate
            :param flattened_models_to_merge_param: Tensor, shape (num_models_to_merge, num_total_params), flattened parameters of individual models that need to be merged
            :param param_value_mask_rate: float, mask rate of the smallest-magnitude parameter values
            :return: torch.Tensor, masked parameters
            """
            # num_models_to_merge, num_total_params = flattened_models_to_merge_param.shape
            num_mask_params = int(flattened_models_to_merge_param.shape[1] * param_value_mask_rate)

            # Tensor, shape (num_models_to_merge, 1), find the num_mask_params-th smallest magnitude element of all the parameters in each individual model
            kth_values, _ = flattened_models_to_merge_param.abs().kthvalue(k=num_mask_params, dim=1, keepdim=True)
            # Tensor, shape (num_models_to_merge, num_total_params), where True is for parameters that we want to preserve
            mask = flattened_models_to_merge_param.abs() >= kth_values

            return flattened_models_to_merge_param * mask

        def get_param_signs(flattened_models_to_merge_param: torch.Tensor):
            """
            Get the signs for each parameter in flattened_models_to_merge_param, computed over individual models that need to be merged
            :param flattened_models_to_merge_param: Tensor, shape (num_models_to_merge, num_total_params), flattened parameters of individual models that need to be merged
            :return: torch.Tensor, parameter signs
            """
            # Tensor, shape (num_total_params, ), the signs of parameters aggregated across individual models that need to be merged
            param_signs = torch.sign(flattened_models_to_merge_param.sum(dim=0))
            # Tensor, shape (, ), a scalar, replace 0 in param_signs to the major sign in param_signs
            majority_sign = torch.sign(param_signs.sum(dim=0))
            param_signs[param_signs == 0] = majority_sign
            return param_signs


        assert isinstance(scaling_coefficient, float), "wrong type of scaling_coefficient, should be float!"

        models_to_merge_task_vectors = [TaskVector(pretrained_model=merged_model, finetuned_model=model_to_merge, 
                                                  exclude_param_names_regex=exclude_param_names_regex) 
                                       for model_to_merge in models_to_merge]

        # Delete model and release GPU memory
        del models_to_merge
        torch.cuda.empty_cache()

        flattened_models_to_merge_param = [task_vector_param_dict_to_single_vector(task_vector=task_vector) 
                                          for task_vector in models_to_merge_task_vectors]
        # Tensor, shape (num_models_to_merge, num_total_params), flattened parameters of individual models that need to be merged
        flattened_models_to_merge_param = torch.vstack(flattened_models_to_merge_param)

        target_coeff = float(1 - source_coeff)

        coeff = torch.tensor([target_coeff, float(source_coeff)]).unsqueeze(1).to(flattened_models_to_merge_param.device)
        # Rescale each model's vector by its softmax value (broadcast along parameter axis)
        flattened_models_to_merge_param = flattened_models_to_merge_param * coeff
        with torch.no_grad():
            # Tensor, shape (num_models_to_merge, num_total_params), mask the smallest-magnitude parameter values using param_value_mask_rate
            flattened_models_to_merge_param = mask_smallest_magnitude_param_values(
                flattened_models_to_merge_param=flattened_models_to_merge_param, 
                param_value_mask_rate=param_value_mask_rate)

            # Tensor, shape (num_total_params, ), get the signs for each parameter in flattened_models_to_merge_param
            param_signs = get_param_signs(flattened_models_to_merge_param=flattened_models_to_merge_param)

            # Tensor, shape (num_total_params, ), disjoint merge
            # Tensor, shape (num_models_to_merge, num_total_params), where True is for parameters that we want to preserve
            param_to_preserve_mask = ((param_signs.unsqueeze(dim=0) > 0) & (flattened_models_to_merge_param > 0)) | ((param_signs.unsqueeze(dim=0) < 0) & (flattened_models_to_merge_param < 0))
            # Tensor, shape (num_models_to_merge, num_total_params), the preserved parameters
            param_to_preserve = flattened_models_to_merge_param * param_to_preserve_mask
            del param_to_preserve_mask, param_signs, flattened_models_to_merge_param

            torch.cuda.empty_cache()
            
            # Tensor, shape (num_total_params, ), the number of models whose parameters can be preserved
            num_models_param_preserved = torch.all(param_to_preserve != 0, dim=0)
            # Tensor, shape (num_total_params, ), the averaged flattened parameters
            merged_flattened_param = torch.where(num_models_param_preserved, param_to_preserve / 2, param_to_preserve)
            del num_models_param_preserved
            merged_flattened_param = torch.sum(merged_flattened_param, dim=0)

            # merged parameter dictionary
            merged_task_vector_param_dict = single_vector_to_task_vector_param_dict(
                single_vector=merged_flattened_param, task_vector=models_to_merge_task_vectors[0])
            merged_task_vector = TaskVector(task_vector_param_dict=merged_task_vector_param_dict)
            # combine with parameters of the merged model based on scaling coefficient
            merged_params = merged_task_vector.combine_with_pretrained_model(pretrained_model=merged_model, 
                                                                           scaling_coefficient=scaling_coefficient)

        return merged_params


class MaskMerging(ModelMergingMethod):
    """
    Mask merging method - applies weight masking before merging.
    This can be used to disentangle task-aware and context-specific changes
    by masking out parameters that are less relevant to the specific task.
    """
    
    def merge(self, merged_model: nn.Module, models_to_merge: list, exclude_param_names_regex: list, 
              weight_format: str = "delta_weight", weight_mask_rates: list = None, 
              use_weight_rescale: bool = True, mask_strategy: str = "random", 
              mask_apply_method: str = "ties_merging", models_use_deepcopy: bool = False, **kwargs):
        """
        Perform mask merging.
        
        Args:
            merged_model: The target model to merge parameters into.
            models_to_merge: List of models whose parameters will be merged.
            exclude_param_names_regex: Regex list for parameters to exclude.
            weight_format: "delta_weight" (diff from base) or "finetuned_weight".
            weight_mask_rates: List of mask rates for each model.
            use_weight_rescale: Whether to rescale weights after masking.
            mask_strategy: "random" or "magnitude".
            mask_apply_method: The underlying merging method to use after masking.
            models_use_deepcopy: Whether to copy models before masking.
        """
        with torch.no_grad():
            if models_use_deepcopy:
                new_models_to_merge = copy.deepcopy(models_to_merge)
            else:
                new_models_to_merge = models_to_merge
            if not weight_mask_rates:
                weight_mask_rates = [0.1] * len(new_models_to_merge)
            for new_model_to_merge, weight_mask_rate in zip(new_models_to_merge, weight_mask_rates):
                # for each individual model, mask its weight
                masked_param_dict = mask_model_weights(finetuned_model=new_model_to_merge, pretrained_model=merged_model,
                                                       exclude_param_names_regex=exclude_param_names_regex, 
                                                       weight_format=weight_format,
                                                       weight_mask_rate=weight_mask_rate, 
                                                       use_weight_rescale=use_weight_rescale, 
                                                       mask_strategy=mask_strategy)
                self.copy_params_to_model(params=masked_param_dict, model=new_model_to_merge)
                
        # Apply the specified merging method to the masked models
        if mask_apply_method == "average_merging":
            merging_method = AverageMerging()
            merged_params = merging_method.merge(merged_model=merged_model,
                                               models_to_merge=new_models_to_merge, 
                                               exclude_param_names_regex=exclude_param_names_regex)
        elif mask_apply_method == "task_arithmetic":
            merging_method = TaskArithmeticMerging()
            merged_params = merging_method.merge(merged_model=merged_model, 
                                               models_to_merge=new_models_to_merge, 
                                               exclude_param_names_regex=exclude_param_names_regex,
                                               scaling_coefficient=kwargs.get('scaling_coefficient', 1.0),
                                               source_coeff=kwargs.get('source_coeff', 0.5))
        elif mask_apply_method == "ties_merging":
            merging_method = TiesMerging()
            merged_params = merging_method.merge(merged_model=merged_model, 
                                               models_to_merge=new_models_to_merge, 
                                               exclude_param_names_regex=exclude_param_names_regex,
                                               param_value_mask_rate=kwargs.get('param_value_mask_rate', 0.8),
                                               scaling_coefficient=kwargs.get('scaling_coefficient', 1.0),
                                               source_coeff=kwargs.get('source_coeff', 0.5))
        else:
            raise NotImplementedError(f"unsupported for mask_apply_method {mask_apply_method}!")
        return merged_params


# Factory function for backward compatibility
def MergingMethod(merging_method_name: str):
    """
    Factory function to create merging method instances by name.
    
    Supported methods:
    - average_merging: Simple weight averaging.
    - task_arithmetic: Task vector arithmetic with extrapolation.
    - ties_merging: Trim, Elect Sign, and Merge strategy.
    - mask_merging: Masking weights before merging.
    """
    if merging_method_name == "average_merging":
        return AverageMerging()
    elif merging_method_name == "task_arithmetic":
        return TaskArithmeticMerging()
    elif merging_method_name == "ties_merging":
        return TiesMerging()
    elif merging_method_name == "mask_merging":
        return MaskMerging()
    else:
        raise NotImplementedError(f"unsupported for merging_method_name {merging_method_name}!") 
