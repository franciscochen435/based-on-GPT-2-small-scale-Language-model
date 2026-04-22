import torch
import torch.nn as nn

from .TransformerModel import TransformerModel


class TransformerForSequenceClassification(TransformerModel):
    def __init__(
        self,
        vocab_size,
        max_seq_len,
        d_model,
        n_heads,
        n_layers,
        d_ff,
        dropout,
        num_labels=2,
    ):
        super().__init__(vocab_size, max_seq_len, d_model, n_heads, n_layers, d_ff, dropout)
        self.num_labels = num_labels
        self.classifier = nn.Linear(d_model, num_labels)

    def forward(self, input_ids, attention_mask=None):
        hidden_states = self.encode(input_ids)

        if attention_mask is None:
            pooled = hidden_states[:, -1, :]
        else:
            lengths = attention_mask.long().sum(dim=1).clamp(min=1) - 1
            batch_idx = torch.arange(hidden_states.size(0), device=hidden_states.device)
            pooled = hidden_states[batch_idx, lengths, :]

        return self.classifier(pooled)
