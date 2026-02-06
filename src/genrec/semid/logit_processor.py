import os
from typing import Optional, List, Tuple, Dict

import torch
import numpy as np
from vllm.config import VllmConfig
from vllm.v1.sample.logits_processor.interface import LogitsProcessor, BatchUpdate
import pandas as pd

class RQMapper:
    def __init__(self, start_id: int, codes_per_layer: int = 256, num_layers: int = 4):
        self.start = int(start_id)
        self.L = int(codes_per_layer)
        self.K = int(num_layers)
        self.end = self.start + self.K * self.L

    def in_window(self, tok: int) -> bool:
        return self.start <= tok < self.end

    def layer_code(self, tok: int) -> Tuple[int, int]:
        off = tok - self.start
        return int(off // self.L), int(off % self.L)

    def layer_span(self, k: int) -> Tuple[int, int]:
        base = self.start + k * self.L
        return base, base + self.L

class Trie:
    # 紧凑 trie：每个节点一个 256 位的下一跳掩码（按层内 code 索引），以及 children 哈希
    def __init__(self, mapper: RQMapper, vocab_size: int, device: torch.device):
        self.mapper = mapper
        self.device = device
        self.vocab_size = int(vocab_size)
        self.children: List[Dict[int, int]] = [dict()]
        self.next_masks = None  # torch.bool [K, N, L]

    def add(self, seq4_codes: List[int]):
        nid = 0
        for k, code in enumerate(seq4_codes):
            base, _ = self.mapper.layer_span(k)
            tok = base + int(code)
            nxt = self.children[nid].get(tok)
            if nxt is None:
                nxt = len(self.children)
                self.children[nid][tok] = nxt
                self.children.append({})
            nid = nxt

    def build(self):
        K, L = self.mapper.K, self.mapper.L
        N = len(self.children)
        masks = torch.zeros((K, N, L), dtype=torch.bool, device=self.device)
        for nid, ch in enumerate(self.children):
            for tok, nxt in ch.items():
                if self.mapper.in_window(tok):
                    k, code = self.mapper.layer_code(tok)
                    masks[k, nid, code] = True
        self.next_masks = masks

    def next_node(self, nid: int, tok: int) -> Optional[int]:
        return self.children[nid].get(tok)

class SimpleTrieConstrainedProcessor(LogitsProcessor):
    """
    约束解码到 RQ 四层 token 的已知 item 集：
    - 仅基于本 request 的 prompt/output tokens 推断当前层位与 trie 节点
    - 在 logits 上放行下一层允许的 code（映射到 vocab 上的 256 连续窗口）
    - 当已生成 4 层后，仅放行 eos_id（如提供）
    环境变量：
      RQ_START_ID (必需), RQ_VOCAB_SIZE (必需), RQ_ITEMS_PATH (必需, 形状 [N,4], code ∈ [0,255])
      RQ_CODES_PER_LAYER=256, RQ_NUM_LAYERS=4, RQ_EOS_ID=-1(禁用)
    """
    def __init__(self, vllm_config: VllmConfig, device: torch.device, is_pin_memory: bool):
        start_id = int(os.getenv("RQ_START_ID", "151669"))  
        vocab_size = int(os.getenv("RQ_VOCAB_SIZE", "152704"))
        L = int(os.getenv("RQ_CODES_PER_LAYER", "256"))
        K = int(os.getenv("RQ_NUM_LAYERS", "4"))
        eos_id_env = int(os.getenv("RQ_EOS_ID", "-1"))
        items_path = os.getenv("RQ_ITEMS_PATH")

        all_items = pd.read_json(items_path, lines=True)
        print(f"Loaded {len(all_items)} items from dataset")
        
        items_codes = []
        for _, row in all_items.iterrows():  # e.g. [(12,34,0,255), ...]
            ca, cb, cc, cd = row["ID_a"], row["ID_b"], row["ID_c"], row["ID_d"]
            items_codes.append((ca, cb, cc, cd))

        codes = np.array(items_codes)
        assert codes.ndim == 2 and codes.shape[1] == K, "items npy must be [N,K]"

        self.device = device
        self.mapper = RQMapper(start_id, L, K)
        self.K = K
        self.L = L
        self.V = vocab_size
        self.eos_id = (None if eos_id_env < 0 else int(eos_id_env))

        # 构建 trie
        self.trie = Trie(self.mapper, vocab_size, device)
        for row in codes:
            self.trie.add([int(x) for x in row.tolist()])
        self.trie.build()  # next_masks: [K, N, L]
        self.N_nodes = self.trie.next_masks.shape[1]

        # 每步仅依赖“本轮 added 的行”，避免跨步与跨 request 调度
        # row_state[row] = (pos, nid)
        self.row_state: Dict[int, Tuple[int, int]] = {}


    def _extract_tail(self, toks: List[int]) -> List[int]:
        # 从尾部提取在窗口内的连续 token，遇到窗口外的token就终止
        tail = []
        for t in reversed(toks):
            if self.mapper.in_window(int(t)):
                tail.append(int(t))
                if len(tail) >= self.K:
                    break
            else:
                break
        return list(reversed(tail))

    def update_state(self, batch_update: Optional[BatchUpdate]) -> None:
        # beam_search 每轮都是新批次，这里只读取 added 并重建 row_state
        self.row_state.clear()
        if not batch_update or not getattr(batch_update, "added", None):
            return
        for rec in batch_update.added:
            # rec: (row_index, sampling_params, prompt_tok_ids, output_tok_ids)
            try:
                row, _sp, prompt_ids, out_ids_so_far = rec
            except Exception:
                # 支持具名属性的变体
                row = int(rec.index if hasattr(rec, "index") else rec[0])
                prompt_ids = list(rec.prompt_token_ids)
                out_ids_so_far = list(rec.output_token_ids)

            # 仅用本 request 的 tokens
            prompt_ids = list(prompt_ids)
            out_ids_so_far = list(out_ids_so_far)
            # 已见序列 = prompt 尾部的 RQ 段 + 本轮已生成（通常为空，因为 max_tokens=1）
            tail = self._extract_tail(prompt_ids + out_ids_so_far)
            
            # 在 trie 上推进；不匹配即停（视为无效前缀 -> pos=0）
            nid = 0
            pos = 0
            for i, tok in enumerate(tail):
                if pos >= self.K:
                    break
                nxt = self.trie.next_node(nid, int(tok))
                if nxt is None:
                    pos = 0
                    nid = 0
                    break
                nid = nxt
                pos += 1
            # print(tail, pos, nid)
            self.row_state[int(row)] = (pos, nid)

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if logits.numel() == 0:
            return logits
        B, V = logits.shape
        assert V == self.V

        neg_inf = torch.tensor(float("-inf"), dtype=logits.dtype, device=logits.device)
        allow = torch.zeros((B, V), dtype=torch.bool, device=logits.device)

        for row in range(B):
            pos, nid = self.row_state.get(row, (0, 0))
            if pos < self.K:
                # 只放行该层窗口里，trie 允许的 codes
                base, hi = self.mapper.layer_span(pos)
                num_legal = 0
                if 0 <= nid < self.N_nodes:
                    codes_mask = self.trie.next_masks[pos, nid]
                    num_legal = int(codes_mask.sum().item())
                    if num_legal > 0:
                        allow[row, base:hi] = codes_mask[: (hi - base)]
                        if self.eos_id is not None and 0 <= self.eos_id < V:
                            allow[row, self.eos_id] = False
                if num_legal == 0:
                    # 兜底并打印诊断
                    print(f"[RQWARN] dead-end row={row} pos={pos} nid={nid} "
                        f"num_legal=0 -> force EOS")
                    if self.eos_id is not None and 0 <= self.eos_id < V:
                        allow[row, :] = False
                        allow[row, self.eos_id] = True
                    else:
                        allow[row, :] = False
            else:
                # 已到 4 层：仅允许 eos（如配置）
                if self.eos_id is not None and 0 <= self.eos_id < V:
                    allow[row, :] = False
                    allow[row, self.eos_id] = True
                else:
                    # 如果未提供 eos，则全部禁用（会导致无法继续采样）
                    allow[row, :] = False

        logits.masked_fill_(~allow, neg_inf)
        return logits

    def is_argmax_invariant(self) -> bool:
        return False
    
