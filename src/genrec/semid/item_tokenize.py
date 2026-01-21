import os
import torch
import numpy as np
import pandas as pd

from src.utils.logger import get_logger, log_with_color

def get_neighbor_indices(codes, neighbor_indices):
    new_codes = []
    n = neighbor_indices.shape[0]
    for code in codes:
        cur = code
        i = 0
        while i < n and neighbor_indices[cur, i] in new_codes:
            i += 1
        new_codes.append(neighbor_indices[cur, i % n])
    return np.array(new_codes)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", type=str, default="Cell_Phones_and_Accessories")
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--cluster_sizes", type=int, nargs="+", default=[256, 256, 256])
    parser.add_argument("--gpu_id", type=int, default=0)
    args = parser.parse_args()

    # Configure logging with colors
    logger = get_logger(__name__)
    log_with_color(logger, "INFO", f"Starting item tokenization for {args.domain}", "magenta")

    embedding_path = f"data/embedding/amazon_{args.domain}.npy"
    index_path = f"data/tokens/amazon_{args.domain}_index.jsonl"
    model_path = f"data/tokens/amazon_{args.domain}_model.pth"

    if not os.path.exists(os.path.dirname(index_path)):
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
    if not os.path.exists(os.path.dirname(model_path)):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
    if not os.path.exists(os.path.dirname(embedding_path)):
        os.makedirs(os.path.dirname(embedding_path), exist_ok=True)

    log_with_color(logger, "INFO", f"Loading embeddings from {embedding_path}", "cyan")
    embeddings = np.load(embedding_path)
    log_with_color(logger, "INFO", f"Loaded {len(embeddings)} embeddings", "red")

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embeddings = torch.from_numpy(embeddings).float().to(device)
    
    from src.utils.utils import init_seed
    init_seed(0, reproducibility=True)

    log_with_color(logger, "INFO", f"Fitting model with {args.n_layers} layers and cluster sizes {args.cluster_sizes}", "magenta")
    try:
        from vector_quantize_pytorch import ResidualVQ
        torch.random.manual_seed(0)
        residual_vq = ResidualVQ(
            dim = embeddings.shape[-1],
            num_quantizers = 4,      # specify number of quantizers
            codebook_size = 256,    # codebook size
            kmeans_init = True,   # set to True
            kmeans_iters = 10,     # number of kmeans iterations to calculate the centroids for the codebook on init
            # stochastic_sample_codes = True,
            # sample_codebook_temp = 5e-4,         # temperature for stochastically sampling codes, 0 would be equivalent to non-stochastic
        ).to(device)
        quantized, indices, commit_loss = residual_vq(embeddings)
        log_with_color(logger, "INFO", f"Commit loss: {commit_loss}", "red")
    except Exception as e:
        log_with_color(logger, "ERROR", f"Item tokenization failed: {e}", "red")
        raise

    index = pd.DataFrame(indices.to("cpu").numpy()).reset_index()
    code_embed = residual_vq.state_dict()["layers.3._codebook.embed"] # Num_Head * Num_Code * Codebook_Size
    # Calculate L2 distances between all code embeddings
    distances = torch.cdist(code_embed.view(-1, code_embed.shape[-1]), code_embed.view(-1, code_embed.shape[-1]), p=2)
    neighbor_indices = torch.topk(distances, k=distances.shape[1], dim=1, largest=False).indices.cpu().numpy()

    tmp_index = index.groupby(list(range(args.n_layers-1))).agg(list)
    max_cluster_size = tmp_index.loc[:, args.n_layers-1].apply(len).max()
    tmp_index.loc[:, args.n_layers-1] = tmp_index.loc[:, args.n_layers-1].apply(lambda x: get_neighbor_indices(x, neighbor_indices))
    index = tmp_index.reset_index().explode(["index", args.n_layers-1]).set_index("index").sort_index()

    log_with_color(logger, "INFO", f"Saving index to {index_path}, max cluster size: {max_cluster_size}", "red")

    index = index.rename(columns={c: f"ID_{chr(c+97)}" for c in range(len(index.columns))})
    index.loc[:, "sem_id"] = index.apply(lambda x: "".join([f"<{chr(c+97)}_{x.iloc[c]}>" for c in range(args.n_layers)]), axis=1)
    index.to_json(index_path, orient="records", lines=True)
    log_with_color(logger, "INFO", f"Saved index to {index_path}", "cyan")

    log_with_color(logger, "INFO", f"Saving model to {model_path}", "cyan")
    torch.save(residual_vq.state_dict(), model_path)

    log_with_color(logger, "INFO", f"Item tokenization completed for {args.domain}", "magenta")