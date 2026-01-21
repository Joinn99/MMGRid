import os
import argparse
import pandas as pd
import torch
import numpy as np

from src.utils.logger import get_logger, log_with_color, setup_logger

# Configure logging with colors
logger = get_logger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate embeddings for Amazon product data')
    
    parser.add_argument(
        '--model_path',
        type=str,
        default="Qwen/Qwen3-Embedding-0.6B",
        help='Path to the embedding model'
    )
    
    parser.add_argument(
        '--domain',
        type=str,
        default='Cell_Phones_and_Accessories',
        help='Domain name'
    )
    
    parser.add_argument(
        '--gpu_id',
        type=str,
        default="3",
        help='GPU device ID to use'
    )
    
    parser.add_argument(
        '--gpu_memory_utilization',
        type=float,
        default=0.5,
        help='GPU memory utilization ratio'
    )
    
    parser.add_argument(
        '--max_model_len',
        type=int,
        default=1024,
        help='Maximum model length'
    )
    
    parser.add_argument(
        '--max_length',
        type=int,
        default=128,
        help='Maximum token length for input processing'
    )
    
    return parser.parse_args()


def initialize_model(model_path, gpu_id, gpu_memory_utilization, max_model_len):
    """Initialize the embedding model."""
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    from vllm import LLM
    model = LLM(
        model=model_path,
        task="embed",
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len
    )
    
    return model

def get_input(item, with_description=True):
    """Process input item and return formatted text for embedding."""
    instruct = "Compress the following sentence into embedding.\n"
    if with_description:
        text = f"{instruct}Title: {item['title']}\nDescription: {item['description']}"
    else:
        if isinstance(item, str):
            text = f"{instruct}Title: {item}"
        else:
            text = f"{instruct}Title: {item['title']}"
    return text


def generate_embeddings(model, data, with_description=True):
    """Generate embeddings for the input data."""
    # Process all items
    if isinstance(data, pd.Series):
        processed_inputs = data.apply(
            lambda x: get_input(x, with_description)
        ).tolist()
    else:
        processed_inputs = data.apply(
            lambda x: get_input(x, with_description), 
            axis=1
        ).tolist()
    
    # Generate embeddings
    outputs = model.embed(processed_inputs)
    
    # Convert to tensor
    embeddings = torch.tensor([o.outputs.embedding for o in outputs])
    
    return embeddings


def save_embeddings(embeddings, output_path):
    """Save embeddings to file."""
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, embeddings.cpu().numpy())
    log_with_color(logger, "INFO", f"Embeddings saved to {output_path}")


def main():
    """Main function to orchestrate the embedding generation process."""
    args = parse_arguments()
    
    input_csv = f"data/information/amazon_{args.domain}.csv.gz"
    output_file = f"data/embedding/amazon_{args.domain}.npy"
    log_with_color(logger, "INFO", f"Generating embeddings for items in {args.domain}", "magenta")

    log_with_color(logger, "INFO", f"Loading model from: {args.model_path}", "cyan")
    model = initialize_model(
        args.model_path, 
        args.gpu_id, 
        args.gpu_memory_utilization, 
        args.max_model_len
    )
    
    log_with_color(logger, "INFO", f"Loading data from: {input_csv}", "cyan")
    data = pd.read_csv(input_csv)
    log_with_color(logger, "INFO", f"Loaded {len(data)} items", "red")

    log_with_color(logger, "INFO", f"Generating embeddings for {len(data)} items...", "magenta")
    embeddings = generate_embeddings(model, data, with_description=True)
    
    log_with_color(logger, "INFO", f"Saving embeddings to: {output_file}", "cyan")
    save_embeddings(embeddings, output_file)
    
    log_with_color(logger, "INFO", "Embedding generation completed successfully!", "magenta")


if __name__ == "__main__":
    main()