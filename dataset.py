import torch
from torch.utils.data import Dataset

class LMDataset(Dataset):
    def __init__(self, token_ids, seq_len, stride=64):
        self.token_ids = token_ids
        self.seq_len = seq_len
        self.stride = stride

    def __len__(self):
        if len(self.token_ids) < self.seq_len + 1:
            return 0
        return (len(self.token_ids) - self.seq_len - 1) // self.stride + 1

    def __getitem__(self, idx):
        start = idx * self.stride
        chunk = self.token_ids[start : start + self.seq_len + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


class ClassificationDataset(Dataset):
    def __init__(self, hf_split, tokenizer, max_seq_len, pad_token_id=0):
        self.hf_split = hf_split
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id

    def __len__(self):
        return len(self.hf_split)

    def __getitem__(self, idx):
        example = self.hf_split[idx]
        token_ids = self.tokenizer.encode(example["text"]).ids[: self.max_seq_len]
        attention_mask = [1] * len(token_ids)

        pad_len = self.max_seq_len - len(token_ids)
        if pad_len > 0:
            token_ids = token_ids + [self.pad_token_id] * pad_len
            attention_mask = attention_mask + [0] * pad_len

        return (
            torch.tensor(token_ids, dtype=torch.long),
            torch.tensor(attention_mask, dtype=torch.long),
            torch.tensor(example["label"], dtype=torch.long),
        )
