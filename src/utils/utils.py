import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
import os
import random
import torch
from typing import List

from src.utils.logger import get_logger, log_with_color

logger = get_logger(__name__)

TIME_SPLIT = [
    (("2017-07-01", "2022-07-01"), "pretrain"),
    (("2022-07-01", "2022-10-01"), "phase1"),
    (("2022-10-01", "2023-01-01"), "phase2"),
    (("2023-01-01", "2023-04-01"), "test"),
]


def time_split_data(df, time_split=TIME_SPLIT):
    log_with_color(logger, "INFO", f"Splitting data with {len(df)} records into {len(time_split)} time periods", "red")
    results = {}
    for (start, end), phase in time_split:
        df_time = df[(df['timestamp'] >= start) & (df['timestamp'] < end)]
        results[phase] = df_time
        log_with_color(logger, "INFO", f"Phase {phase}: {len(df_time)} records ({start} to {end})", "red")
    return results

def save_csv_with_precision(df, filepath, precision=3, **kwargs):
    """
    Save DataFrame to CSV with specified float precision.
    
    Args:
        df: pandas DataFrame to save
        filepath: path to save the CSV file
        precision: number of decimal places for float columns (default: 6)
        **kwargs: additional arguments to pass to to_csv()
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Format string for float precision
    float_format = f'%.{precision}f'
    
    # Save with specified float format
    df.to_csv(filepath, float_format=float_format, sep="\t", **kwargs)
    
    return filepath

def format_float_columns(df, precision=6):
    """
    Format float columns in DataFrame to specified precision.
    
    Args:
        df: pandas DataFrame
        precision: number of decimal places (default: 6)
    
    Returns:
        DataFrame with formatted float columns
    """
    df_formatted = df.copy()
    
    # Apply formatting to float columns
    for col in df_formatted.select_dtypes(include=[np.floating]).columns:
        df_formatted[col] = df_formatted[col].apply(lambda x: f'{x:.{precision}f}' if pd.notna(x) else x)
    
    return df_formatted

def get_merged_name(mode: str, source_domain: str, target_domains: List[str], splits: List[str], method: str, merging_args: dict = None):
    assert min(len(splits), len(target_domains)) <= 1, "Split and target domains cannot be > 1 at the same time"
    source_domain = source_domain[:3]
    target_domains = [domain[:3] for domain in target_domains]


    if len(splits) == 1:
        name = f"merged-{source_domain}-{''.join(target_domains)}-{splits[0]}-{mode}-{method[:4]}"
    else:
        name = f"merged-{source_domain}-{''.join(splits)}-{mode}-{method[:4]}"

    if merging_args is not None:
        name += f"-{'-'.join([f'{k[:3]}={v}' for k, v in merging_args.items()])}"
    return name


def init_seed(seed, reproducibility):
    r""" init random seed for random functions in numpy, torch, cuda and cudnn

    Args:
        seed (int): random seed
        reproducibility (bool): Whether to require reproducibility
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if reproducibility:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False