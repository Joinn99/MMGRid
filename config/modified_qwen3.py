from torchtune.models.qwen3._component_builders import qwen3
from torchtune.modules import TransformerDecoder
import torch

def qwen3_0_6b_instruct() -> TransformerDecoder:
    """
    Builder for creating a Qwen3 0.6B instruct model initialized w/ the default parameter values
    from https://huggingface.co/Qwen/Qwen3-0.6B

    Returns:
        TransformerDecoder: Instantiation of Qwen3 0.6B instruct model
    """
    return qwen3(
        vocab_size=152704,
        num_layers=28,
        num_heads=16,
        num_kv_heads=8,
        embed_dim=1024,
        intermediate_dim=3072,
        max_seq_len=40960,
        head_dim=128,
        attn_dropout=0.0,
        norm_eps=1e-6,
        rope_base=1000000.0,
        tie_word_embeddings=True,
        q_proj_bias=False,
        k_proj_bias=False,
        v_proj_bias=False,
        q_norm=True,
        k_norm=True,
    )

def qwen3_0_6b_instruct_fixed() -> TransformerDecoder:
    model = qwen3(
        vocab_size=151936,
        num_layers=28,
        num_heads=16,
        num_kv_heads=8,
        embed_dim=1024,
        intermediate_dim=3072,
        max_seq_len=40960,
        head_dim=128,
        attn_dropout=0.0,
        norm_eps=1e-6,
        rope_base=1000000.0,
        tie_word_embeddings=True,
        q_proj_bias=False,
        k_proj_bias=False,
        v_proj_bias=False,
        q_norm=True,
        k_norm=True,
    )

    for layer in model.layers:
        layer.requires_grad_(False)
    model.norm.requires_grad_(False)
    return model

def qwen3_1_7b_instruct() -> TransformerDecoder:
    """
    Builder for creating a Qwen3 1.7B instruct model initialized w/ the default parameter values
    from https://huggingface.co/Qwen/Qwen3-1.7B

    Returns:
        TransformerDecoder: Instantiation of Qwen3 1.7B instruct model
    """
    return qwen3(
        vocab_size=152704,
        num_layers=28,
        num_heads=16,
        num_kv_heads=8,
        embed_dim=2048,
        intermediate_dim=6144,
        max_seq_len=40960,
        head_dim=128,
        attn_dropout=0.0,
        norm_eps=1e-6,
        rope_base=1000000.0,
        tie_word_embeddings=True,
        q_proj_bias=False,
        k_proj_bias=False,
        v_proj_bias=False,
        q_norm=True,
        k_norm=True,
    )

def qwen3_4b_instruct() -> TransformerDecoder:
    """
    Builder for creating a Qwen3 4B instruct model initialized w/ the default parameter values
    from https://huggingface.co/Qwen/Qwen3-4B

    Returns:
        TransformerDecoder: Instantiation of Qwen3 4B instruct model
    """
    return qwen3(
        vocab_size=152704,
        num_layers=36,
        num_heads=32,
        num_kv_heads=8,
        embed_dim=2560,
        intermediate_dim=9728,
        max_seq_len=40960,
        head_dim=128,
        attn_dropout=0.0,
        norm_eps=1e-6,
        rope_base=1000000.0,
        tie_word_embeddings=True,
        q_proj_bias=False,
        k_proj_bias=False,
        v_proj_bias=False,
        q_norm=True,
        k_norm=True,
    )