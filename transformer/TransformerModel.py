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

    def encode(self, input_ids):
        if input_ids.dim() != 2:
            raise ValueError(
                f"input_ids must have shape(B, T), got shape {tuple(input_ids.shape)}."
            )
        x = self.embed(input_ids)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return x

    def forward(self, input_ids):
        x = self.encode(input_ids)
        logits = self.lm_head(x)   # (B, T, vocab_size)
        return logits
