import json
import os
import torch
import torch.nn.functional as F

from tokenizers import Tokenizer

from config import vocab_size, max_seq_len, d_model, n_heads, n_layers, d_ff, dropout
from transformer.TransformerBuilder import TransformerModelBuilder

class TextGenerator:
    def __init__(self, model, tokenizer, device="cpu"):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()
        self.eos_id = self._first_token_id(["<eos>", "</s>"])
        self.suppressed_token_ids = [
            token_id
            for token_id in (
                self._first_token_id(["<pad>"]),
                self._first_token_id(["<unk>"]),
                self._first_token_id(["<mask>"]),
                self._first_token_id(["<s>"]),
            )
            if token_id is not None
        ]

    def _first_token_id(self, token_names):
        for token_name in token_names:
            token_id = self.tokenizer.token_to_id(token_name)
            if token_id is not None:
                return token_id
        return None

    def encode_prompt(self, prompt: str) -> torch.Tensor:
        encoding = self.tokenizer.encode(prompt)
        ids = encoding.ids
        if len(ids) == 0:
            raise ValueError("Prompt produced no token IDs. Try a different prompt.")
        return torch.tensor([ids], dtype=torch.long, device=self.device)

    def decode_tokens(self, token_ids):
        return self.tokenizer.decode(token_ids)

    def crop_context(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.size(1) > max_seq_len:
            input_ids = input_ids[:, -max_seq_len:]
        return input_ids

    def suppress_special_tokens(self, logits: torch.Tensor) -> torch.Tensor:
        if self.suppressed_token_ids:
            logits = logits.clone()
            logits[:, self.suppressed_token_ids] = float("-inf")
        return logits

    def greedy_decode(self, prompt: str, max_new_tokens: int = 50) -> str:
        input_ids = self.encode_prompt(prompt)

        with torch.no_grad():
            for _ in range(max_new_tokens):
                context_ids = self.crop_context(input_ids)
                logits = self.model(context_ids)              # [B, T, V]
                next_token_logits = logits[:, -1, :]        # [B, V]
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                input_ids = torch.cat([input_ids, next_token], dim=1)
                if self.eos_id is not None and next_token.item() == self.eos_id:
                    break

        return self.decode_tokens(input_ids[0].tolist())

    def top_k_decode(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        k: int = 50,
        temperature: float = 1.0
    ) -> str:
        input_ids = self.encode_prompt(prompt)

        with torch.no_grad():
            for _ in range(max_new_tokens):
                context_ids = self.crop_context(input_ids)
                logits = self.model(context_ids)
                next_token_logits = logits[:, -1, :] / temperature
                next_token_logits = self.suppress_special_tokens(next_token_logits)

                k = min(k, next_token_logits.size(-1))
                top_k_vals, top_k_idx = torch.topk(next_token_logits, k=k, dim=-1)
                probs = F.softmax(top_k_vals, dim=-1)
                sampled_idx = torch.multinomial(probs, num_samples=1)
                next_token = top_k_idx.gather(-1, sampled_idx)

                input_ids = torch.cat([input_ids, next_token], dim=1)
                if self.eos_id is not None and next_token.item() == self.eos_id:
                    break

        return self.decode_tokens(input_ids[0].tolist())

    def nucleus_decode(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        p: float = 0.9,
        temperature: float = 1.0
    ) -> str:
        input_ids = self.encode_prompt(prompt)

        with torch.no_grad():
            for _ in range(max_new_tokens):
                context_ids = self.crop_context(input_ids)
                logits = self.model(context_ids)
                next_token_logits = logits[:, -1, :] / temperature
                next_token_logits = self.suppress_special_tokens(next_token_logits)

                sorted_logits, sorted_indices = torch.sort(
                    next_token_logits, descending=True, dim=-1
                )
                sorted_probs = F.softmax(sorted_logits, dim=-1)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

                remove_mask = cumulative_probs > p
                remove_mask[..., 1:] = remove_mask[..., :-1].clone()
                remove_mask[..., 0] = False

                sorted_logits[remove_mask] = float("-inf")
                filtered_probs = F.softmax(sorted_logits, dim=-1)

                sampled_idx = torch.multinomial(filtered_probs, num_samples=1)
                next_token = sorted_indices.gather(-1, sampled_idx)

                input_ids = torch.cat([input_ids, next_token], dim=1)
                if self.eos_id is not None and next_token.item() == self.eos_id:
                    break

        return self.decode_tokens(input_ids[0].tolist())


def load_model(model_path: str, device: str):
    model = (
        TransformerModelBuilder()
        .with_vocab_size(vocab_size)
        .with_max_seq_len(max_seq_len)
        .with_d_model(d_model)
        .with_n_heads(n_heads)
        .with_n_layers(n_layers)
        .with_d_ff(d_ff)
        .with_dropout(dropout)
        .build()
    ).to(device)

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def save_results(results, json_path="generated_samples.json", txt_path="generated_samples.txt"):
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    with open(txt_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write("=" * 100 + "\n")
            f.write(f"PROMPT: {r['prompt']}\n\n")
            f.write("GREEDY:\n")
            f.write(r["greedy"] + "\n\n")
            f.write("TOP-K:\n")
            f.write(r["top_k"] + "\n\n")
            f.write("NUCLEUS:\n")
            f.write(r["nucleus"] + "\n\n")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer_path = "tokenizer/trained_tokenizer/tokenizer.json"
    model_path = "gpt_model.pt"

    if not os.path.isfile(tokenizer_path):
        legacy_tokenizer_path = os.path.join("tokenizer", "trained_tokenizer", "new_tokenizer.json")
        if os.path.isfile(legacy_tokenizer_path):
            tokenizer_path = legacy_tokenizer_path

    tokenizer = Tokenizer.from_file(tokenizer_path)
    model = load_model(model_path, device)

    prompts = [
        "The future of artificial intelligence",
        "In a small town near the river",
        "The main purpose of the experiment was",
        "Once the robot reached the door",
        "Machine learning models can be useful when",
    ]

    generator = TextGenerator(model, tokenizer, device=device)

    results = []
    for prompt in prompts:
        greedy_text = generator.greedy_decode(prompt, max_new_tokens=50)
        top_k_text = generator.top_k_decode(prompt, max_new_tokens=50, k=50, temperature=1.0)
        nucleus_text = generator.nucleus_decode(prompt, max_new_tokens=50, p=0.9, temperature=1.0)

        sample = {
            "prompt": prompt,
            "greedy": greedy_text,
            "top_k": top_k_text,
            "nucleus": nucleus_text,
        }
        results.append(sample)

        print("=" * 100)
        print("PROMPT:", prompt)
        print("\nGREEDY:\n", greedy_text)
        print("\nTOP-K:\n", top_k_text)
        print("\nNUCLEUS:\n", nucleus_text)
        print()

    save_results(results)
    print("Saved outputs to generated_samples.json and generated_samples.txt")


if __name__ == "__main__":
    main()
