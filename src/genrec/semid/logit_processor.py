import os
from typing import Optional, List, Tuple, Dict

import torch
import numpy as np
from vllm.config import VllmConfig
from vllm.v1.sample.logits_processor.interface import LogitsProcessor, BatchUpdate
import pandas as pd

class RQMapper:
    """
    Maps RQ (Residual Quantization) token IDs to layer/code pairs and vice versa.
    The vocabulary is divided into K layers, each containing L consecutive codes.
    """
    def __init__(self, start_id: int, codes_per_layer: int = 256, num_layers: int = 4):
        self.start = int(start_id)
        self.L = int(codes_per_layer)
        self.K = int(num_layers)
        self.end = self.start + self.K * self.L

    def in_window(self, tok: int) -> bool:
        """Check if a token ID falls within the RQ code range."""
        return self.start <= tok < self.end

    def layer_code(self, tok: int) -> Tuple[int, int]:
        """Convert token ID to (layer, code) tuple."""
        off = tok - self.start
        return int(off // self.L), int(off % self.L)

    def layer_span(self, k: int) -> Tuple[int, int]:
        """Return the [start, end) token range for layer k."""
        base = self.start + k * self.L
        return base, base + self.L

class Trie:
    """
    Compact trie for RQ codes.
    Each node stores a 256-bit mask indicating valid next codes within the layer,
    plus a hash map of child node indices.
    """
    def __init__(self, mapper: RQMapper, vocab_size: int, device: torch.device):
        self.mapper = mapper
        self.device = device
        self.vocab_size = int(vocab_size)
        self.children: List[Dict[int, int]] = [dict()]  # child maps token->node_id
        self.next_masks = None  # torch.bool [K, N, L]

    def add(self, seq4_codes: List[int]):
        """Insert a 4-code sequence into the trie."""
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
        """Build the next_masks tensor after all sequences are added."""
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
        """Return the child node index for the given token, or None."""
        return self.children[nid].get(tok)

class SimpleTrieConstrainedProcessor(LogitsProcessor):
    """
    Constrains decoding to known 4-layer RQ item codes.
    - Only uses prompt + already-generated tokens of the current request.
    - Masks logits to allow only the next-layer codes permitted by the trie.
    - After 4 layers, only allows eos_id (if provided).

    Environment variables:
      RQ_START_ID (required), RQ_VOCAB_SIZE (required), RQ_ITEMS_PATH (required, [N,4] codes in [0,255])
      RQ_CODES_PER_LAYER=256, RQ_NUM_LAYERS=4, RQ_EOS_ID=-1 (disabled)
    """
    def __init__(self, vllm_config: VllmConfig, device: torch.device, is_pin_memory: bool):
        start_id = int(os.getenv("RQ_START_ID", "151669"))
        vocab_size = int(os.getenv("RQ_VOCAB_SIZE", "152704"))
        L = int(os.getenv("RQ_CODES_PER_LAYER", "256"))
        K = int(os.getenv("RQ_NUM_LAYERS", "4"))
        eos_id_env = int(os.getenv("RQ_EOS_ID", "-1"))
        items_path = os.getenv("RQ_ITEMS_PATH")

        # Load item codes from JSONL
        all_items = pd.read_json(items_path, lines=True)
        print(f"Loaded {len(all_items)} items from dataset")

        items_codes = []
        for _, row in all_items.iterrows():
            ca, cb, cc, cd = row["ID_a"], row["ID_b"], row["ID_c"], row["ID_d"]
            items_codes.append((ca, cb, cc, cd))

        codes = np.array(items_codes)
        assert codes.ndim == 2 and codes.shape[1] == K, "items must be [N,K]"

        self.device = device
        self.mapper = RQMapper(start_id, L, K)
        self.K = K
        self.L = L
        self.V = vocab_size
        self.eos_id = (None if eos_id_env < 0 else int(eos_id_env))

        # Build trie
        self.trie = Trie(self.mapper, vocab_size, device)
        for row in codes:
            self.trie.add([int(x) for x in row.tolist()])
        self.trie.build()  # next_masks: [K, N, L]
        self.N_nodes = self.trie.next_masks.shape[1]

        # Per-request state: row -> (current_layer, trie_node_id)
        self.row_state: Dict[int, Tuple[int, int]] = {}

    def _extract_tail(self, toks: List[int]) -> List[int]:
        """
        Extract the longest suffix of RQ tokens (up to K) from the token list.
        Stops at the first out-of-window token.
        """
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
        """
        Rebuild row_state from the latest batch.
        Each beam-search step provides a fresh batch; we only use 'added' records.
        """
        self.row_state.clear()
        if not batch_update or not getattr(batch_update, "added", None):
            return
        for rec in batch_update.added:
            # Handle both tuple and named-record formats
            try:
                row, _sp, prompt_ids, out_ids_so_far = rec
            except Exception:
                row = int(rec.index if hasattr(rec, "index") else rec[0])
                prompt_ids = list(rec.prompt_token_ids)
                out_ids_so_far = list(rec.output_token_ids)

            # Merge prompt and current output tokens
            tail = self._extract_tail(prompt_ids + out_ids_so_far)

            # Walk the trie; stop on first mismatch
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
            self.row_state[int(row)] = (pos, nid)

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Mask logits so that only valid next codes (or EOS) can be sampled.
        """
        if logits.numel() == 0:
            return logits
        B, V = logits.shape
        assert V == self.V

        neg_inf = torch.tensor(float("-inf"), dtype=logits.dtype, device=logits.device)
        allow = torch.zeros((B, V), dtype=torch.bool, device=logits.device)

        for row in range(B):
            pos, nid = self.row_state.get(row, (0, 0))
            if pos < self.K:
                # Allow only codes permitted by trie for the current layer
                base, hi = self.mapper.layer_span(pos)
                num_legal = 0
                if 0 <= nid < self.N_nodes:
                    codes_mask = self.trie.next_masks[pos, nid]
                    num_legal = int(codes_mask.sum().item())
                    if num_legal > 0:
                        allow[row, base:hi] = codes_mask[: (hi - base)]
                        # Disable EOS if it falls inside this window
                        if self.eos_id is not None and 0 <= self.eos_id < V:
                            allow[row, self.eos_id] = False
                if num_legal == 0:
                    # Dead-end: force EOS if available
                    print(f"[RQWARN] dead-end row={row} pos={pos} nid={nid} "
                          f"num_legal=0 -> force EOS")
                    if self.eos_id is not None and 0 <= self.eos_id < V:
                        allow[row, :] = False
                        allow[row, self.eos_id] = True
                    else:
                        allow[row, :] = False
            else:
                # If we're at the end of the trie, only allow EOS
                if self.eos_id is not None and 0 <= self.eos_id < V:
                    allow[row, :] = False
                    allow[row, self.eos_id] = True
                else:
                    # If no EOS is provided, disable all logits (will cause no further sampling)
                    allow[row, :] = False

        logits.masked_fill_(~allow, neg_inf)
        return logits

    def is_argmax_invariant(self) -> bool:
        return False
    
