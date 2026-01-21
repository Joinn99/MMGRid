import json
from src.utils.logger import get_logger

logger = get_logger(__name__)

def add_token(level=4, codebook_size=256, start_id=151669):
    added_tokens = []
    for i in range(level):
        for j in range(codebook_size):
            added_tokens.append({
                "id": start_id + i * codebook_size + j,
                "content": f"<{chr(i+97)}_{j}>",
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True
            },)
    return added_tokens


def add_token_to_tokenizer(tokenizer_path, start_id=151669):
    tok = json.load(open(tokenizer_path, "r"))
    start_id = max([e["id"] for e in tok["added_tokens"]]) + 1
    new_tok = {
        "added_tokens": tok["added_tokens"] + add_token(level=4, codebook_size=256, start_id=start_id)
    }
    json.dump(new_tok, open("config/added_tokens.json", "w"), indent=1)

def change_shape(tokens):
    new = {}
    for token in tokens:
        id = token["id"]
        token.pop("id")
        new[str(id)] = token
    return new

def modify_checkpoint(base_model_path, checkpoint_path, start_id=151669, vocab_size=152704):
    import json
    config = json.load(open(base_model_path + "/tokenizer_config.json", "r"))
    config["added_tokens_decoder"].update(change_shape(add_token(level=4, codebook_size=256, start_id=start_id)))
    # config["additional_special_tokens"] += [e["content"] for e in add_token(level=4, codebook_size=256, start_id=start_id)]
    json.dump(config, open(checkpoint_path + "/tokenizer_config.json", "w"), indent=1)

    config = json.load(open(base_model_path + "/tokenizer.json", "r"))
    config["added_tokens"] += add_token(level=4, codebook_size=256, start_id=start_id)
    json.dump(config, open(checkpoint_path + "/tokenizer.json", "w"), indent=1)

    config = json.load(open(base_model_path + "/config.json", "r"))
    config["vocab_size"] = vocab_size
    json.dump(config, open(checkpoint_path + "/config.json", "w"), indent=1)


if __name__ == "__main__":
    import argparse
    import os
    CKPT_PATH = os.getenv("CKPT_PATH")
    MODEL_ZOO_PATH = os.getenv("MODEL_ZOO_PATH")
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, default=f"{CKPT_PATH}/")
    parser.add_argument("--domain", type=str, default="Video_Games")
    parser.add_argument("--split", type=str, default="pretrain")
    parser.add_argument("--epoch", type=int, default=2)
    parser.add_argument("--checkpoint_path", type=str, default=f"{CKPT_PATH}/Video_Games-pretrain-sem_id/epoch_2")
    parser.add_argument("--base_model_path", type=str, default=f"{MODEL_ZOO_PATH}/Qwen3-0.6B")
    parser.add_argument("--start_id", type=int, default=151669)
    parser.add_argument("--vocab_size", type=int, default=152704)
    args = parser.parse_args()
    # add_token_to_tokenizer(args.tokenizer_path)
    logger.info(f"Modifying checkpoint for domain: {args.domain}, split: {args.split}, epoch: {args.epoch}")
    checkpoint_path = f"{args.checkpoint_dir}/{args.domain}-{args.split}-sem_id/epoch_{args.epoch}" if args.checkpoint_path is None else args.checkpoint_path
    modify_checkpoint(base_model_path=args.base_model_path, checkpoint_path=checkpoint_path, start_id=args.start_id, vocab_size=args.vocab_size)
    logger.info(f"Checkpoint modification completed successfully")
