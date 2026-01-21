# Copyright (c) 2024 westlake-repl
# Copyright (c) 2024 Bytedance Ltd. and/or its affiliate
# SPDX-License-Identifier: MIT
# This file has been modified by Junyi Chen.
#
# Original file was released under MIT, with the full license text
# available at https://choosealicense.com/licenses/mit/.
#
# This modified file is released under the same license.

from cProfile import run
from logging import getLogger
import torch
import json
from REC.data import *
from REC.config import Config
from REC.utils import init_logger, get_model, init_seed, set_color
from REC.trainer import Trainer
import torch.distributed as dist
from safetensors.torch import save_file, load_file

import os
import numpy as np
import pandas as pd
import argparse
import torch.distributed as dist
import torch
from datetime import datetime, timezone, timedelta

def convert_str(s):
    try:
        if s.lower() == 'none':
            return None
        if s.lower() == 'true':
            return True
        if s.lower() == 'false':
            return False
        float_val = float(s)
        if float_val.is_integer():
            return int(float_val)
        return float_val
    except ValueError:
        print(f"Unable to convert the string '{s}' to None / Bool / Float / Int, retaining the original string.")
        return s


def run_loop(local_rank, config_file=None, saved=True, extra_args=[]):

    # configurations initialization
    config = Config(config_file_list=config_file)

    device = torch.device("cuda", local_rank)
    config['device'] = device
    if len(extra_args):
        for i in range(0, len(extra_args), 2):
            key = extra_args[i][2:]
            value = extra_args[i + 1]
            try:
                if '[' in value or '{' in value:
                    value = json.loads(value)
                    if isinstance(value, dict):
                        for k, v in value.items():
                            value[k] = convert_str(v)
                    else:
                        value = [convert_str(x) for x in value]
                else:
                    value = convert_str(value)
                if '.' in key:
                    k1, k2 = key.split('.')
                    config[k1][k2] = value
                else:
                    config[key] = value
            except:
                raise ValueError(f"{key} {value} invalid")

    init_seed(config['seed'], config['reproducibility'])

    # logger initialization
    init_logger(config)
    logger = getLogger()
    if 'text_path' in config:
        config['text_path'] = os.path.join(config['text_path'], config['dataset'] + '.csv.gz')
        logger.info(f"Update text_path to {config['text_path']}")

    # get model and data
    dataload = load_data(config)
    train_loader, valid_loader, test_loader = bulid_dataloader(config, dataload)
    print(f"{len(train_loader) = }")
    print(f"{len(valid_loader) = }")
    print(f"{len(test_loader) = }")

    model = get_model(config['model'])(config, dataload)
    # model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)

    world_size = torch.distributed.get_world_size()
    trainer = Trainer(config, model)

    logger.info(set_color('\nWorld_Size', 'pink') + f' = {world_size} \n')
    logger.info(config)
    logger.info(dataload)
    logger.info(model)

    if config['val_only']:
        if 'eval_name' in config and config['eval_name']:
            eval_name = config['eval_name']
            ckpt_path = os.path.join(config['checkpoint_dir'], eval_name, 'model.safetensors')
        else:
            eval_name = f"{config['domain']}-{config['split']}-hllm"
            ckpt_path = os.path.join(config['checkpoint_dir'], f"{config['domain']}-{config['split']}", 'model.safetensors')
        
        ckpt = load_file(ckpt_path, device='cpu')
        logger.info(f'Eval only model load from {ckpt_path}')
        msg = trainer.model.load_state_dict(ckpt, False)
        logger.info(f'{msg.unexpected_keys = }')
        logger.info(f'{msg.missing_keys = }')
        test_result = trainer.evaluate(test_loader, load_best_model=False, show_progress=config['show_progress'], init_model=True)
        logger.info(set_color('test result', 'yellow') + f': {test_result}')
        test_result = dict(test_result)
        for k, v in test_result.items():
            if isinstance(v, float):
                test_result[k] = round(100 * v, 7)
        test_result.update({
            "time": datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S"),
            "mode": "hllm", "split": config['split'], "domain": config['domain'], "name": eval_name 
        })
        test_result = pd.DataFrame([test_result])
        test_result = test_result[['ndcg@5','recall@5','mrr@5','ndcg@10','recall@10','mrr@10','ndcg@20',\
            'recall@20','mrr@20','time','mode','split','domain','name']]
        if config['save_result_path']:
            if not os.path.exists(config['save_result_path']):
                test_result.to_csv(config['save_result_path'], index=False, sep='\t', float_format='%.3f')
            else:
                test_result.to_csv(config['save_result_path'], index=False, mode='a', sep='\t', header=False, float_format='%.3f')
        logger.info(f'Test result saved to {config["save_result_path"]}')

    else:
        # training process
        if config['resume_checkpoint_dir']:
            ckpt_path = os.path.join(config['resume_checkpoint_dir'], 'model.safetensors')
            ckpt = load_file(ckpt_path, device='cpu')
            logger.info(f'Resume model load from {ckpt_path}')
            msg = trainer.model.load_state_dict(ckpt, False)
            logger.info(f'{msg.unexpected_keys = }')
            logger.info(f'{msg.missing_keys = }')
        best_valid_score, best_valid_result = trainer.fit(
            train_loader, valid_loader, saved=saved, show_progress=config['show_progress']
        )
        logger.info(f'Trianing Ended' + set_color('best valid ', 'yellow') + f': {best_valid_result}')

        # model evaluation
        test_result = trainer.evaluate(test_loader, load_best_model=saved, show_progress=config['show_progress'])

        logger.info(set_color('best valid ', 'yellow') + f': {best_valid_result}')
        logger.info(set_color('test result', 'yellow') + f': {test_result}')

        trainer.model.to(torch.bfloat16)
        save_file(trainer.model.state_dict(), os.path.join(config['checkpoint_dir'], 'model.safetensors'))

        return {
            'best_valid_score': best_valid_score,
            'valid_score_bigger': config['valid_metric_bigger'],
            'best_valid_result': best_valid_result,
            'test_result': test_result
        }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", nargs='+', type=str)
    args, extra_args = parser.parse_known_args()
    local_rank = int(os.environ['LOCAL_RANK'])
    config_file = args.config_file

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')

    run_loop(local_rank=local_rank, config_file=config_file, extra_args=extra_args)
