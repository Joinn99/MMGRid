import argparse
import os
import shutil
import torch
import json
from typing import List

from transformers import AutoModelForCausalLM
from safetensors.torch import save_file

from src.utils.logger import get_logger, log_with_color
from src.merging.merging_methods import MergingMethod
from src.utils.utils import get_merged_name, init_seed

logger = get_logger(__name__)
logger.propagate = False

def hllm_from_pretrained(checkpoint_path: str = None, class_path: str = None, base_model_path: str = None):
    """
    Load HLLM model from pretrained checkpoint.
    
    Args:
        checkpoint_path: Path to the model checkpoint.
        class_path: Path to the HLLM class definition.
        base_model_path: Path to the base model if different from checkpoint.
        
    Returns:
        Loaded HLLM model on CUDA with bfloat16 precision.
    """
    import sys
    import importlib
    assert class_path is not None, "class_path is required for HLLM"
    sys.path.append(os.path.curdir + "/src/genrec/hllm")
    hllm = importlib.import_module("REC.model.HLLM.hllm")
    HLLM = getattr(hllm, "HLLM")

    with open(f"config/hllm.json", "r") as f:
        config = json.load(f)
    if base_model_path and ("pretrain" not in base_model_path) and ("phase1" not in base_model_path) and ("merged" not in base_model_path) and ("phase2" not in base_model_path):
        config["item_pretrain_dir"] = base_model_path
        config["user_pretrain_dir"] = base_model_path
        config["item_llm_init"] = True
        config["user_llm_init"] = True
        config["load_pretrain"] = None
    else:
        if base_model_path:
            config["load_pretrain"] = base_model_path + "/model.safetensors"
        else:
            config["load_pretrain"] = checkpoint_path + "/model.safetensors"
    model = HLLM(config, None)
    return model.to(torch.bfloat16).to("cuda")

def get_model_path(mode:str, split:str, domain:str, checkpoint_dir = f"{os.environ['CKPT_PATH']}"):
    """
    Get the filesystem path for a specific model checkpoint.
    
    Args:
        mode: GR paradigm (title, sem_id, hllm).
        split: Temporal stage (pretrain, phase1, phase2).
        domain: Recommendation domain.
        checkpoint_dir: Base directory for checkpoints.
        
    Returns:
        Path to the model checkpoint.
    """
    if mode in ["title", "sem_id"]:
        epoch = "2" if split == "pretrain" else "1"
        path = f"{checkpoint_dir}/{domain}-{split}-{mode}/epoch_{epoch}"
    else:
        path = f"{checkpoint_dir}/{domain}-{split}"
    return path

def model_loader(mode:str, model_path:str = None, **kwargs):
    """
    Universal model loader for different GR paradigms.
    
    Args:
        mode: GR paradigm (title, sem_id, hllm).
        model_path: Path to the model checkpoint.
        **kwargs: Additional arguments for specific model types.
        
    Returns:
        Loaded model instance.
    """
    if mode in ["title", "sem_id"]:
        if "base_model_path" in kwargs:
            model_path = kwargs.get("base_model_path")
            model = AutoModelForCausalLM.from_pretrained(model_path)
            if mode == "sem_id" and "resize_token_embeddings" in kwargs:
                model.resize_token_embeddings(kwargs.get("resize_token_embeddings", model.vocab_size + 1024))
        else:
            model = AutoModelForCausalLM.from_pretrained(model_path)
        return model
    else:
        return hllm_from_pretrained(model_path, **kwargs)

def get_models(mode:str,merged_model_name: str, models_to_merge_names: List[str], method: str = "average_merging", **kwargs):
    """
    Load multiple models for merging.
    
    Args:
        mode: GR paradigm.
        merged_model_name: Path/Name of the base/target model.
        models_to_merge_names: List of paths/names of models to merge.
        method: Merging method.
        
    Returns:
        tuple: (target_model, list_of_models_to_merge)
    """
    log_with_color(logger, "INFO", f"Loading merged model from: {merged_model_name}", "cyan")

    merged_model = model_loader(mode, merged_model_name, **{"class_path": kwargs.get("class_path", None)})
    models_to_merge = []
    for i, model_name in enumerate(models_to_merge_names):
        log_with_color(logger, "INFO", f"Loading model {i+1}/{len(models_to_merge_names)}: {model_name}", "cyan")
        model = model_loader(mode, model_name, **{"class_path": kwargs.get("class_path", None)})
        models_to_merge.append(model)
    
    models_to_merge.append(merged_model)
    if method not in ["average_merging"]:
        if mode == "sem_id":
            kwargs["resize_token_embeddings"] = merged_model.vocab_size
        log_with_color(logger, "INFO", f"Initialize model from {kwargs.get('base_model_path', None)}", "cyan")
        merged_model = model_loader(mode, None, **kwargs)
        
    log_with_color(logger, "INFO", f"Successfully loaded {len(models_to_merge)} models", "green")
    return merged_model, models_to_merge

def retain_hllm_userllm(merged_model, target_domain_model):
    """
    Copy user LLM parameters from target domain model to merged model for HLLM.
    """
    log_with_color(logger, "INFO", f"Retaining hllm userllm ", "red")
    with torch.no_grad():
    # Recursively copy the parameters from target_domain_model to merged_model
        state_dict = target_domain_model.user_llm.state_dict()
        merged_model.user_llm.load_state_dict(state_dict)
    return merged_model

def retain_sem_id_embedding(merged_model, target_domain_model, start_id: int = 151669):
    """
    Retain specialized semantic ID embeddings in the merged model.
    """
    log_with_color(logger, "INFO", f"Retaining sem_id embeddings from {target_domain_model.name_or_path}", "red")
    with torch.no_grad():
        merged_model.lm_head.weight[start_id:] = target_domain_model.lm_head.weight[start_id:]
        merged_model.model.embed_tokens.weight[start_id:] = target_domain_model.model.embed_tokens.weight[start_id:]
    return merged_model

def save_merged_model(merged_model, merged_model_path: str, output_path: str, mode: str, name: str):
    """
    Save the merged model and its configuration files.
    """
    log_with_color(logger, "INFO", f"Saving merged model to: {output_path}", "blue")
    
    if not os.path.exists(output_path):
        log_with_color(logger, "INFO", f"Creating output directory: {output_path}", "yellow")
        os.makedirs(output_path, exist_ok=True)
    
    # Save the model tensors
    if mode == "hllm":
        tensor_path = f"{output_path}/model.safetensors"
        save_file(merged_model.to(torch.bfloat16).state_dict(), tensor_path)
    else:
        merged_model.to(torch.bfloat16).save_pretrained(output_path, safe_serialization=True)
        os.rename(f"{output_path}/model.safetensors", f"{output_path}/model-00001-of-00001.safetensors")
    # Copy non-tensor files
    log_with_color(logger, "INFO", f"Copying non-tensor files from: {merged_model_path}", "cyan")
    copied_files = 0
    if mode in ["title", "sem_id"]:
        for file in os.listdir(merged_model_path):
            if (not file.endswith(".safetensors")) and (not file.endswith(".pth")) and (not file.endswith(".index.json")):
                src_path = f"{merged_model_path}/{file}"
                dst_path = f"{output_path}/{file}"
                shutil.copy(src_path, dst_path)
                copied_files += 1
        
        log_with_color(logger, "INFO", f"Copied {copied_files} non-tensor files", "green")
    log_with_color(logger, "INFO", f"Merged model name: <<<{name}>>>", "green")
    log_with_color(logger, "INFO", f"Successfully saved merged model to: {output_path}", "green")


def merge_models(mode, source_domain, target_domains, splits, method, base_model_path=None, hllm_class_path=None, merging_args=None):
    """
    Main function to perform model merging process
    
    Args:
        mode: str - The mode (title, sem_id, hllm)
        source_domain: str - Source domain name
        target_domains: list - List of target domain names
        splits: list - List of splits to merge
        method: str - Merging method (average_merging, ties_merging, mask_merging, task_arithmetic)
        base_model_path: str - Path to base model (optional)
        hllm_class_path: str - Path to HLLM class (optional)
        
    Returns:
        str - Output name of the merged model
    """

    log_with_color(logger, "INFO", "Starting model merging process", "magenta")
    init_seed(0, reproducibility=True)
    
    # Get model paths
    output_name = get_merged_name(mode, source_domain, target_domains, splits, method, merging_args)
    output_path = f"{os.environ['CKPT_PATH']}/{output_name}"

    if len(splits) == 1:
        merged_model_path = get_model_path(mode, splits[0], source_domain)
        models_to_merge_paths = [
            get_model_path(mode, splits[0], target_domain)
            for target_domain in target_domains
        ]
    else:
        merged_model_path = get_model_path(mode, splits[0], source_domain)
        models_to_merge_paths = [
            get_model_path(mode, split, source_domain)
            for split in splits[1:]
        ]

    # Load models
    merged_model, models_to_merge = get_models(
        mode=mode,
        merged_model_name=merged_model_path,
        models_to_merge_names=models_to_merge_paths,
        method=method,
        base_model_path=base_model_path,
        class_path=hllm_class_path
    )
    # Perform merging
    merged_model = MergingMethod(method).get_merged_model(
        merged_model=merged_model,
        models_to_merge=models_to_merge,
        exclude_param_names_regex=[],
        **merging_args
    )
    
    if mode == "sem_id":
        merged_model = retain_sem_id_embedding(merged_model, models_to_merge[-1])
    # elif mode == "hllm":
    #     merged_model = retain_hllm_userllm(merged_model, models_to_merge[-1])
    # Save merged model
    save_merged_model(
        merged_model=merged_model,
        merged_model_path=merged_model_path,
        output_path=output_path,
        mode=mode,
        name=output_name
    )
    del merged_model
    del models_to_merge
    torch.cuda.empty_cache()

    log_with_color(logger, "INFO", "Model merging process completed", "magenta")
    return output_name



if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="title")
    parser.add_argument("--source_domain", type=str, default="Movies_and_TV")
    parser.add_argument("--splits", nargs="+", default=["pretrain"])
    parser.add_argument("--target_domains", nargs="+", default=["Video_Games"])
    parser.add_argument("--base_model_path", type=str, default=None)
    parser.add_argument("--hllm_class_path", type=str, default=None)
    parser.add_argument("--method", type=str, default="average_merging", choices=["average_merging", "ties_merging", "mask_merging", "task_arithmetic"])
    args = parser.parse_args()
    

    # Call the function with parsed arguments
    merge_models(
        mode=args.mode,
        source_domain=args.source_domain,
        target_domains=args.target_domains,
        splits=args.splits,
        method=args.method,
        base_model_path=args.base_model_path,
        hllm_class_path=args.hllm_class_path,
        logger=logger
    )