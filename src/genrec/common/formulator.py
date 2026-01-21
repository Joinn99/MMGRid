import os
import random
import argparse
import datetime
import pandas as pd
from tqdm import tqdm

from src.utils.utils import time_split_data, TIME_SPLIT
from src.genrec.common.prompt import prompt_template, item_template, prediction_template
from src.utils.logger import get_logger, log_with_color

tqdm.pandas()

# Configure logging with colors
logger = get_logger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate training data for recommendation models')
    
    parser.add_argument(
        '--domain',
        type=str,
        default='Clothing_Shoes_and_Jewelry',
        help='Domain name for the dataset'
    )
    
    parser.add_argument(
        '--max_len',
        type=int,
        default=30,
        help='Maximum length of historical items to consider'
    )
    
    parser.add_argument(
        '--min_len',
        type=int,
        default=5,
        help='Minimum length of historical items required'
    )
    
    parser.add_argument(
        '--output_format',
        type=str,
        default='json',
        choices=['json', 'csv'],
        help='Output format for the generated data'
    )

    parser.add_argument(
        '--max_user_sample',
        type=int,
        default=5,
        help='Maximum number of samples for each user'
    )

    parser.add_argument(
        '--index',
        type=str,
        default='title',
        choices=['title', 'sem_id'],
        help='Index of the item information to use'
    )

    return parser.parse_args()


def load_data(domain, time_filter=None):
    """Load and preprocess the dataset."""
    log_with_color(logger, "INFO", f"Loading dataset for domain: {domain}", "magenta")
    
    # Load interaction data
    interaction_file = f"data/dataset/amazon_{domain}.csv.gz"
    log_with_color(logger, "INFO", f"Loading interaction data from: {interaction_file}", "cyan")
    df = pd.read_csv(interaction_file)
    
    # Convert timestamp and sort
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    if time_filter:
        df = df[df["timestamp"] < time_filter]
    df = df.sort_values(by="timestamp").groupby("user_id").agg(list)
    
    # Load item information
    item_file = f"data/information/amazon_{domain}.csv.gz"
    log_with_color(logger, "INFO", f"Loading item information from: {item_file}", "cyan")
    item = pd.read_csv(item_file)

    sem_id_file = f"data/tokens/amazon_{domain}_index.jsonl"
    if os.path.exists(sem_id_file):
        sem_id = pd.read_json(sem_id_file, lines=True)
        item["sem_id"] = sem_id["sem_id"]
    item = item.set_index("item_id").fillna("NA")
    
    return df, item

def formulate_message(item_info, source_list, target, index, index_prompt=None):
    if not index_prompt:
        index_prompt = index
    sep = "\n\n" if index_prompt == "title" else "\n"
    source_infos = item_info.loc[source_list].apply(
        lambda x: item_template[index_prompt].format(**x), axis=1
    ).tolist()
    target_info = prediction_template[index].format(**{index: item_info.loc[target][index]})
    messages = [
        {"role": "user", "content": prompt_template.format(index="item title" if index == "title" else "item index") + "\n" + sep.join(source_infos)},
        {"role": "assistant", "content": "<think>\n\n</think>\n\n" + target_info}
    ]
    return messages

def assemble_user_data(
        item_id_list,
        timestamp_list,
        all_item_info,
        max_len=30,
        min_len=5,
        max_user_sample=5,
        index="title",
        test=True,
        valid_period=("2017-01-01", "2023-04-01"),
    ):
    """Assemble training data for a user's interaction sequence."""
    
    results = []
    item_info = all_item_info.loc[item_id_list]    
    start, end = datetime.datetime.strptime(valid_period[0], "%Y-%m-%d"), datetime.datetime.strptime(valid_period[1], "%Y-%m-%d")
    if test:
        valid_idx = [len(item_id_list)-1] if start <= timestamp_list[-1] < end else []
    else:
        valid_idx = []
        for i in range(min_len-1, len(item_id_list)):
            if timestamp_list[i] < start: continue
            elif timestamp_list[i] >= end: break
            else: valid_idx.append(i)

        if len(valid_idx) > max_user_sample:
            valid_idx = random.sample(valid_idx, max_user_sample)

    for i in valid_idx:
        idx_start = max(0, i-max_len)
        results.append(
            {
                "messages": formulate_message(item_info, item_id_list[idx_start:i], item_id_list[i], index),
                "timestamp": timestamp_list[i],
                "item_id": item_id_list[i],
                "history": item_id_list[idx_start:i],
                "aux": False
            }
        )
        if (not test) and (index == "sem_id"):
            results.append(
                {
                    "messages": formulate_message(item_info, item_id_list[idx_start:i], item_id_list[i], "title", index),
                    "timestamp": timestamp_list[i],
                    "item_id": item_id_list[i],
                    "history": None,
                    "aux": True
                }
            )
            results.append(
                {
                    "messages": formulate_message(item_info, item_id_list[idx_start:i], item_id_list[i], index, "title"),
                    "timestamp": timestamp_list[i],
                    "item_id": item_id_list[i],
                    "history": None,
                    "aux": True
                }
            )
    
    return results


def save_data(data, domain, phase, index, mode="w"):
    """Save the generated data to file."""
    if index == "title":
        output_file = f"data/messages/amazon_{domain}_{phase}.jsonl.gz"
    elif index == "sem_id":
        output_file = f"data/sequences/amazon_{domain}_{phase}.jsonl.gz"
    
    if not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # Convert to DataFrame and save as JSON
    data.to_json(output_file, orient="records", lines=True, compression="gzip", mode=mode)

    log_with_color(logger, "INFO", f"Data saved to: {output_file}, {data.shape[0]} data points", "cyan")


def main():
    """Main function to orchestrate the data generation process."""
    args = parse_arguments()
    
    log_with_color(logger, "INFO", f"Starting data generation for domain: {args.domain}", "magenta")
    
    # Load data
    df, item = load_data(args.domain, time_filter=TIME_SPLIT[-1][0][1])
    random.seed(0)
    # Generate training data
    log_with_color(logger, "INFO", f"Generating training data for {args.domain}", "magenta")

    for period, phase in TIME_SPLIT:
        output_data = df.progress_apply(
            lambda x: assemble_user_data(
                x["item_id"], x["timestamp"], item,
                max_len=args.max_len, min_len=args.min_len,
                max_user_sample=5 if phase == "pretrain" else 1e9,
                index=args.index,
                test=(phase == "test"),
                valid_period=period
            ), axis=1
        )
        output_data = output_data.explode().dropna()
        output_data = pd.DataFrame(
            output_data.tolist(),
            columns=["messages", "timestamp", "item_id", "history", "aux"],
            index=output_data.index
        )

        if phase != "test":
            output_data = output_data.drop(columns=["history"])
            if args.index == "sem_id":
                output_data = output_data.groupby(["user_id", "item_id", "timestamp", "aux"]).sample(n=1)
        output_data = output_data.reset_index().drop(columns=["aux"])
        save_data(output_data.sort_values(by="timestamp"), args.domain, phase, args.index)
    
    log_with_color(logger, "INFO", "Data generation completed successfully!", "magenta")

if __name__ == "__main__":
    main()