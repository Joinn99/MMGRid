import os
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from src.utils.logger import get_logger, log_with_color

tqdm.pandas()
from vllm import LLM
from src.genrec.semid.logit_processor import SimpleTrieConstrainedProcessor
# Configure logging with colors
logger = get_logger(__name__)
logger.propagate = False

from transformers import AutoTokenizer

def build_trie_processor(domain="Cell_Phones_and_Accessories", model_path="Qwen3-0.6B", start_id=151669):
    """
    Build a trie-based logit processor for constrained generation in Semantic ID mode.
    
    Args:
        domain: The recommendation domain.
        model_path: Path to the model.
        start_id: The starting token ID for item codes.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    os.environ["RQ_START_ID"] = str(start_id)
    os.environ["RQ_CODES_PER_LAYER"] = "256"
    os.environ["RQ_NUM_LAYERS"] = "4"
    os.environ["RQ_ITEMS_PATH"] = f"data/tokens/amazon_{domain}_index.jsonl"
    os.environ["RQ_EOS_ID"] = str(tokenizer.eos_token_id)  # 或者设为 -1 表示不用
    os.environ["RQ_VOCAB_SIZE"] = str(152704)
    return SimpleTrieConstrainedProcessor

def get_llm(
    model_path,
    beam_width=None,
    mode="title",
    domain="Cell_Phones_and_Accessories",
):
    """
    Initialize a vLLM engine for generation.
    
    Args:
        model_path: Path to the model checkpoint.
        beam_width: Beam search width.
        mode: "title" (text-based) or "sem_id" (semantic ID).
        domain: The domain for context-specific logit processing.
    """
    if mode == "sem_id":
        llm_class = LLM
        assert os.path.exists(f"data/tokens/amazon_{domain}_index.jsonl"), f"Index file for {domain} not found"
        logits_processors = [build_trie_processor(domain, model_path)]
    else:
        llm_class = LLM
        logits_processors = None
    max_logprobs = 2 * beam_width if beam_width else 20
    llm = llm_class(
        model=model_path,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.4,
        max_model_len=4096,
        trust_remote_code=True,
        enforce_eager=True,
        max_logprobs=max_logprobs,
        logprobs_mode="processed_logprobs" if mode == "sem_id" else "raw_logprobs",
        logits_processors=logits_processors,
    )
    return llm

def batch_chat(
    llm,
    messages,
    beam_width=20,
    max_tokens=32,
):
    from vllm import SamplingParams
    sampling_params=SamplingParams(
        n=beam_width,
        temperature=0.6,
        max_tokens=max_tokens,
        stop=["\n"],
    )
    output = llm.chat(
        messages=messages,
        sampling_params=sampling_params,
        continue_final_message=True,
        add_generation_prompt=False,
        chat_template_kwargs={"enable_thinking": False},
    )
    return [[e.outputs[i].text for i in range(beam_width)] for e in output]

from vllm import LLM

def batch_beam_search(
    llm: LLM,
    messages,
    beam_width=20,
    max_tokens=4,
):
    from vllm.sampling_params import BeamSearchParams
    sampling_params=BeamSearchParams(
        temperature=0.0,
        beam_width=beam_width,
        max_tokens=max_tokens,
    )
    tokenizer = llm.get_tokenizer()
    prompts = [
        {"prompt_token_ids": e} for e in tokenizer.apply_chat_template(
            messages, add_generation_prompt=False, continue_final_message=True
        )
    ]
    small_batch_size = 256
    outputs = []
    for i in tqdm(range(0, len(prompts), small_batch_size)):
        batch_prompts = prompts[i:i+small_batch_size]
        batch_outputs = llm.beam_search(
            prompts=batch_prompts,
            params=sampling_params,
        )
        outputs.extend(batch_outputs)
    result_rankings = [
        [tokenizer.decode(s.tokens[-max_tokens:]) for s in output.sequences]
        for output in outputs
    ]
    return result_rankings


def generate_data(model_path, mode, split, domain, beam_width, sample_num, output_name=None):
    """
    Generate data using the specified model and parameters.
    
    Args:
        model_path: str - Path to the model
        mode: str - Generation mode (title, sem_id, etc.)
        split: str - Data split (phase1, phase2, etc.)
        domain: str - Domain name
        beam_width: int - Beam width for generation
        sample_num: int - Number of samples to generate (-1 for all)
        output_name: str - Optional output name override
    """
    np.random.seed(0)
    log_with_color(logger, "INFO", f"Generating data for {domain} in {split} with {mode} mode", "magenta")

    if mode == "title":
        input_path = f"data/messages/amazon_{domain}_test.jsonl.gz"
    else:
        input_path = f"data/sequences/amazon_{domain}_test.jsonl.gz"

    if output_name:
        output_path = f"data/outputs/amazon_{output_name}.jsonl"
    else:
        output_path = f"data/outputs/amazon_{domain}-{split}-{mode}.jsonl"

    log_with_color(logger, "INFO", f"Loading data from {input_path}", "cyan")
    df = pd.read_json(input_path, lines=True)
    if sample_num != -1 and sample_num < len(df):
        df = df.sample(n=sample_num, random_state=0)
    df = df.sort_index()
    log_with_color(logger, "INFO", f"Loaded {len(df)} rows", "red")
    llm = get_llm(model_path, beam_width=beam_width, mode=mode, domain=domain)
    
    prompt_prefix = "Recommended Item Title:" if mode == "title" else "Recommended Item Index: "
    
    log_with_color(logger, "INFO", f"Batch chatting...", "magenta")
    df["messages"] = df["messages"].apply(lambda x: x[:-1] + [{"role": "assistant", "content": prompt_prefix}])
    if mode == "title":
        # outputs = batch_beam_search(llm, df["messages"].tolist(), beam_width=beam_width, max_tokens=32)
        outputs = batch_chat(llm, df["messages"].tolist(), beam_width=beam_width, max_tokens=32)
    else:
        outputs = batch_beam_search(llm, df["messages"].tolist(), beam_width=beam_width, max_tokens=4)
        # outputs = batch_chat(llm, df["messages"].tolist(), beam_width=beam_width, max_tokens=4)

    log_with_color(logger, "INFO", f"Batch chatting done", "magenta")
    
    df["output"] = outputs
    df.drop(columns=["messages", "history"], inplace=True)
    log_with_color(logger, "INFO", f"Saving to {output_path}", "cyan")
    df.reset_index().to_json(output_path, lines=True, orient="records")
    
    log_with_color(logger, "INFO", f"Generation completed", "magenta")

if __name__ == "__main__":
    import os
    DATA_PATH = os.getenv("DATA_PATH")
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=f"{MODEL_ZOO_PATH}/Qwen3-0.6B")
    parser.add_argument("--mode", type=str, default="title")
    parser.add_argument("--split", type=str, default="phase1")
    parser.add_argument("--domain", type=str, default="Cell_Phones_and_Accessories")
    parser.add_argument("--beam_width", type=int, default=20)
    parser.add_argument("--sample_num", type=int, default=2000)
    args = parser.parse_args()

    generate_data(
        model_path=args.model_path,
        mode=args.mode,
        split=args.split,
        domain=args.domain,
        beam_width=args.beam_width,
        sample_num=args.sample_num,
    )
