#!/usr/bin/env python3
"""
Python implementation of merge.sh script
Performs model merging and generation with cleanup
"""

import os
import shutil
import sys
import torch
import pandas as pd
import json
from typing import Optional

sys.path.append("./processor")
from src.merging.merge import merge_models
from src.eval.generate import generate_data
from src.utils.utils import save_csv_with_precision, get_merged_name

# Fallback defaults
MODES = ["sem_id"]
SPLITS = ["phase2"]
SOURCE_DOMAIN = "Movies_and_TV"
TARGET_DOMAINS = ["Video_Games"]
METHODS = ["mask_merging"]

# Default Settings
BEAM_WIDTHS = {"sem_id": 20, "title": 5, "hllm": -1}


def setup_argparse():
    """Setup command line argument parser"""
    import argparse
    parser = argparse.ArgumentParser(description="Model merging and generation pipeline")
    
    # Core parameters
    parser.add_argument("--mode", type=str, default=MODES[0] if MODES else "sem_id",
                       choices=["title", "sem_id", "hllm"],
                       help="Generation mode")
    parser.add_argument("--source_domain", type=str, default=SOURCE_DOMAIN,
                       help="Source domain name")
    parser.add_argument("--splits", nargs="+", default=SPLITS,
                       help="List of splits to merge")
    parser.add_argument("--target_domains", nargs="+", default=TARGET_DOMAINS,
                       help="List of target domain names")
    parser.add_argument("--method", type=str, default=METHODS[0] if METHODS else "task_arithmetic",
                       choices=["average_merging", "ties_merging", "mask_merging", "task_arithmetic"],
                       help="Merging method")
    parser.add_argument("--merging_args", type=str, default="{}",
                       help="Merging arguments")
    
    # Model paths
    parser.add_argument("--base_model_path", type=str, 
                       default=f"{os.getenv('MODEL_ZOO_PATH', '')}/Qwen3-0.6B",
                       help="Path to base model")
    parser.add_argument("--hllm_class_path", type=str, default="",
                       help="Path to HLLM class")
    
    # Generation parameters
    parser.add_argument("--beam_width", type=int, default=None,
                       help="Beam width for generation (auto-set based on mode if not specified)")
    parser.add_argument("--sample_num", type=int, default=5000,
                       help="Number of samples to generate")
    parser.add_argument("--gpu_id", type=str, default="0",
                       help="GPU ID to use")
    
    # Eval Path
    parser.add_argument("--embed_model_path", type=str, default=f"{os.getenv('MODEL_ZOO_PATH', '')}/Qwen3-Embedding-8B",
                       help="Path to embedding model")
    parser.add_argument("--rescale", action="store_true",
                       help="Rescale the embeddings")
    # Control flags
    parser.add_argument("--skip_cleanup", action="store_true",
                       help="Skip cleanup of merged model checkpoint")
    parser.add_argument("--skip_merging", action="store_true",
                       help="Skip merging")
    parser.add_argument("--skip_generation", action="store_true",
                       help="Skip generation")
    return parser


class ModelMerger:
    def __init__(self, args):
        self.mode = args.mode
        self.splits = args.splits
        self.source_domain = args.source_domain
        self.target_domains = args.target_domains
        self.method = args.method
        self.sample_num = args.sample_num
        self.base_model_path = args.base_model_path
        self.hllm_class_path = args.hllm_class_path
        self.merging_args = json.loads(args.merging_args) if args.merging_args and isinstance(args.merging_args, str) else args.merging_args
        
        self.gpu_id = args.gpu_id
        if args.beam_width is None:
            self.beam_width = BEAM_WIDTHS.get(self.mode, 5)
        else:
            self.beam_width = args.beam_width
        self.skip_merging = args.skip_merging
        self.skip_generation = args.skip_generation
        self.skip_cleanup = args.skip_cleanup

    
    def run_merging(self) -> Optional[str]:
        """Run the merging script and capture model name from output"""
        print("Running merging script...")
        if not self.skip_merging:
            return merge_models(
                mode=self.mode,
                source_domain=self.source_domain,
                target_domains=self.target_domains,
                splits=self.splits,
                method=self.method,
                base_model_path=args.base_model_path,
                hllm_class_path=args.hllm_class_path,
                merging_args=self.merging_args
            )
        else:
            return get_merged_name(
                self.mode,
                self.source_domain,
                self.target_domains,
                self.splits,
                self.method,
                self.merging_args
            )
    
    def run_generation(self, model_name: str):
        """Run the generation script"""
        print(f"Running generation for model: {model_name}")
        
        # Set checkpoint paths
        checkpoint_dir = f"{os.getenv('CKPT_PATH', '')}"
        checkpoint_path = f"{checkpoint_dir}/{model_name}"
        
        if self.mode == "hllm":
            import subprocess
            # Build the torchrun command
            cmd = [
                "torchrun",
                f"--master_port={29501+int(self.gpu_id)}",
                "--node_rank=0",
                "--nproc_per_node=1",
                "--nnodes=1",
                "run.py",
                "--config_file",
                "overall/LLM_ddp_full.yaml",
                "HLLM/HLLM-test.yaml"
            ]
            
            # Set environment variable for CUDA_VISIBLE_DEVICES
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = self.gpu_id
            env["DOMAIN"] = self.source_domain
            env["SPLIT"] = "merged"
            env["EVAL_NAME"] = model_name
            # Execute the command
            try:
                result = subprocess.run(cmd, env=env, check=True, capture_output=True, text=True, cwd=os.path.curdir + "/src/genrec/hllm")
                print("HLLM torchrun command executed successfully")
                print(result.stdout)
            except subprocess.CalledProcessError as e:
                print(f"Error executing HLLM torchrun command: {e}")
                print(f"Return code: {e.returncode}")
                print(f"Stdout: {e.stdout}")
                print(f"Stderr: {e.stderr}")
                raise
        else:
            # Build command arguments
            generate_data(
                model_path=checkpoint_path,
                mode=self.mode,
                split="merged",
                domain=self.source_domain,
                beam_width=self.beam_width,
                sample_num=self.sample_num,
                output_name=model_name
            )
    
    def cleanup_checkpoint(self, model_name: str):
        """Clean up the merged model checkpoint directory"""
        if not model_name:
            return
        
        checkpoint_dir = f"{os.getenv('CKPT_PATH', '')}"
        checkpoint_path = f"{checkpoint_dir}/{model_name}"
        
        # Check if directory exists and remove it (equivalent to bash conditional)
        if os.path.exists(checkpoint_path) and os.path.isdir(checkpoint_path):
            print(f"Removing {checkpoint_path}")
            try:
                shutil.rmtree(checkpoint_path)
                print("Cleanup completed successfully")
            except Exception as e:
                print(f"Error during cleanup: {e}")
        else:
            print(f"Checkpoint directory {checkpoint_path} does not exist or is not a directory")
    
    def run(self):
        """Main execution method"""
        print("Starting model merging and generation process...")
        
        # Step 1: Run merging and get model name
        model_name = self.run_merging()
        if not model_name:
            print("Failed to get model name from merging process")
            return False
        torch.cuda.empty_cache()
        # Step 2: Run generation
        if not args.skip_generation:
            self.run_generation(model_name)

        # Step 3: Cleanup (commented out in original script)
        if not args.skip_cleanup:
            self.cleanup_checkpoint(model_name)
        
        print("Process completed successfully")
        return model_name


def merge_and_eval(args, eval_groups, top_k=[5,10,20]):
    if args.mode == "title":
        from src.dataset.embed import initialize_model
        embed_model = initialize_model(
            model_path=args.embed_model_path,
            gpu_id="1",
            gpu_memory_utilization=0.8,
            max_model_len=2048
        )

    from src.eval.eval import title_eval, sem_id_eval
    output_path = f"data/archive/amazon.tsv"

    for source, eval_names in eval_groups.items():
        if args.mode == "title":
            metrics = title_eval(source, None, embed_model, top_k=top_k, beam_size=args.beam_width, rescale=True, eval_names=eval_names)
        elif args.mode == "sem_id":
            metrics = sem_id_eval(source, None, top_k=top_k, eval_names=eval_names)
        elif args.mode == "hllm":
            metrics = None
        else:
            raise ValueError(f"Invalid mode: {args.mode}")
        if metrics is not None:
            save_csv_with_precision(pd.DataFrame(metrics), output_path, precision=3, index=False, header=False, mode="a")

if __name__ == "__main__":
    """Main entry point"""
    parser = setup_argparse()
    args = parser.parse_args()
    all_eval_names = []

    if not args.beam_width:
        args.beam_width = BEAM_WIDTHS.get(args.mode, 5)

    def single_run(args, model_merger_class):
        merger = model_merger_class(args)
        output_name = merger.run()
        del merger
        torch.cuda.empty_cache()
        return output_name

    eval_groups = {args.source_domain: [("merged", single_run(args, ModelMerger))]}

    merge_and_eval(args, eval_groups)
