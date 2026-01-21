# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F

from torchtune.utils import get_logger
from torchtune.modules.loss.cross_entropy_loss import LinearCrossEntropyLoss

log = get_logger()


class MaskedCELoss(LinearCrossEntropyLoss):
    def __init__(
        self,
        num_output_chunks: int = 8,
        ignore_index: int = -100,
        tp_enabled: bool = False,
        mask_ignored_tokens: bool = True,
        start_id: int = 151669,
        layer_size: int = 256,
        num_layers: int = 4,
    ):
        super().__init__(
            num_output_chunks=num_output_chunks,
            ignore_index=ignore_index,
            tp_enabled=tp_enabled,
            mask_ignored_tokens=mask_ignored_tokens,
        )
        self.start_id = start_id
        self.layer_size = layer_size
        self.num_layers = num_layers
        self.end_id = start_id + num_layers * layer_size

    def compute_cross_entropy(
        self,
        hidden_chunk: torch.Tensor,     # []
        target_chunk: torch.Tensor,
    ) -> torch.Tensor:
        """
        [RQ-VAE版]
        分别计算常规Token和RQ-VAE分层Token的损失，然后相加。

        1.  常规Token (target < START_ID):
            - Logits: 全词汇表 [B*T, V_full]
            - Softmax: 在 V_full 上计算
        2.  RQ-VAE分层Token (target >= START_ID):
            - 每个item由num_layers个token表示
            - 每层token有独立的layer_size词汇表
            - Softmax: 仅在对应层的词汇表上计算
        """
        if self.linear_projection is None:
            raise AttributeError("forward called before update_model")

        # 1. 计算全部 logits
        # [batch_size, chunk_size, vocab_size]
        logits = self.linear_projection(hidden_chunk)

        # 2. 展平 logits 和 targets
        # [B*T, V_full]
        vocab_size = logits.shape[-1]
        logits_flat = logits.view(-1, vocab_size)
        # [B*T]
        targets_flat = target_chunk.view(-1)

        # 准备一个 0.0 的张量用于累加损失，确保设备和梯度正确
        total_loss = torch.tensor(0.0, device=logits.device, requires_grad=True)

        # --- 第一部分：计算 RQ-VAE 分层Token的损失 (每层独立 Softmax) ---
        
        rqvae_mask = (targets_flat >= self.start_id) & \
                     (targets_flat < self.end_id) & \
                     (targets_flat != self.ignore_index)
        rqvae_indices = torch.where(rqvae_mask)[0]

        if rqvae_indices.numel() > 0:
            # 筛选出相关的 targets
            # 形状: [num_rqvae]
            rqvae_targets_absolute = targets_flat[rqvae_indices]
            
            # 计算每个token属于哪一层
            # 形状: [num_rqvae]
            layer_indices = (rqvae_targets_absolute - self.start_id) // self.layer_size
            
            # 计算每层内的相对索引
            # 形状: [num_rqvae]
            rqvae_targets_relative = (rqvae_targets_absolute - self.start_id) % self.layer_size
            
            # 为每一层分别计算损失
            for layer in range(self.num_layers):
                # 找到属于当前层的token
                layer_mask = (layer_indices == layer)
                layer_token_indices = torch.where(layer_mask)[0]
                
                if layer_token_indices.numel() > 0:
                    # 获取当前层的全局token索引
                    global_indices = rqvae_indices[layer_token_indices]
                    
                    # 计算当前层的logits范围
                    layer_start = self.start_id + layer * self.layer_size
                    layer_end = layer_start + self.layer_size
                    
                    # 筛选出当前层的logits
                    # 形状: [num_layer_tokens, layer_size]
                    layer_logits = logits_flat[global_indices, layer_start:layer_end]
                    
                    # 筛选出当前层的targets
                    # 形状: [num_layer_tokens]
                    layer_targets = rqvae_targets_relative[layer_token_indices]
                    
                    # 计算当前层的损失
                    layer_loss = F.cross_entropy(
                        layer_logits.float(),
                        layer_targets,
                        reduction="sum",
                        ignore_index=self.ignore_index,
                    )
                    total_loss = total_loss + layer_loss

        # --- 第二部分：计算常规 Token 的损失 (完整 Softmax) ---

        total_loss = total_loss * 10.0

        regular_mask = (targets_flat < self.start_id) & \
                       (targets_flat != self.ignore_index)
        regular_indices = torch.where(regular_mask)[0]

        if regular_indices.numel() > 0:
            # 筛选出相关的 logits (取 V_full)
            # 形状: [num_regular, V_full]
            regular_logits = logits_flat[regular_indices]
            
            # 筛选出相关的 targets (它们已经是绝对索引)
            # 形状: [num_regular]
            regular_targets = targets_flat[regular_indices]

            loss_regular = F.cross_entropy(
                regular_logits.float(),
                regular_targets,
                reduction="sum",
                # 此处不再需要 ignore_index，因为已手动过滤
            )
            total_loss = total_loss + loss_regular

        return total_loss
