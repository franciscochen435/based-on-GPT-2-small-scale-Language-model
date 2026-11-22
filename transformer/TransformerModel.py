import torch
import torch.nn as nn
from .Embedding import Embedding
from .TransformerBlock import TransformerBlock

# GPT-style Transformer (decoder-only architecture)
# Perform next-token prediction (language modeling)
#     - Output probability distribution over vocabulary for each position
class TransformerModel(nn.Module):
    def __init__(self, vocab_size, max_seq_len, d_model, n_heads, n_layers, d_ff, dropout):
        super().__init__()
        self.embed = Embedding(vocab_size, d_model, max_seq_len, dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, max_seq_len, dropout)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.apply(self._init_weights)
        self.lm_head.weight = self.embed.token_embed.weight

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, past_kvs=None, use_cache=False):
        # past_kvs: optional list with one (K, V) per block, from a previous call with use_cache=True.
        # Training and the existing callers use the default path and still get logits only.
        if input_ids.dim() != 2:
            raise ValueError(
                f"input_ids must have shape(B, T), got shape {tuple(input_ids.shape)}."
            )
        start_pos = past_kvs[0][0].size(2) if past_kvs is not None else 0
        x = self.embed(input_ids, start_pos=start_pos)
        present_kvs = [] if use_cache else None
        for i, block in enumerate(self.blocks):
            if use_cache:
                past_kv = past_kvs[i] if past_kvs is not None else None
                x, kv = block(x, past_kv=past_kv, use_cache=True)
                present_kvs.append(kv)
            else:
                x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)   # (B, T, vocab_size)
        if use_cache:
            return logits, present_kvs
        return logits
