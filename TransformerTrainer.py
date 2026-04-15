import os
import math
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from checkpoint import save_checkpoint, load_checkpoint
from TokenizerFactory import TokenizerFactory
from datasets import load_dataset

from config import *
from transformer.TransformerBuilder import TransformerModelBuilder
from dataset import LMDataset
from TrainingObserver import TrainingMonitor, ConsoleTrainingObserver
from BaseTrainer import BaseTrainer


def get_lr_scheduler(optimizer, warmup_steps, total_steps, last_epoch=-1, eta_min_ratio=0.01):
    if total_steps < 1:
        raise ValueError(f"total_steps must be >= 1, got {total_steps}.")
    warmup_steps = max(0, warmup_steps)

    def lr_lambda(step: int):
        if step < warmup_steps:
            return float(step + 1) / float(max(warmup_steps, 1))
        if step >= total_steps:
            return float(eta_min_ratio)
        i = step - warmup_steps
        cosine_steps = total_steps - warmup_steps
        if cosine_steps <= 0:
            return 1.0
        progress = float(i) / float(max(1, cosine_steps - 1))
        progress = min(1.0, progress)
        return float(eta_min_ratio) + (1.0 - float(eta_min_ratio)) * 0.5 * (
            1.0 + math.cos(math.pi * progress)
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)

class TransformerTrainer(BaseTrainer):
    def __init__(
        self,
        tokenizer_path: str = "tokenizer/trained_tokenizer/tokenizer.json",
        tokenizer_type: str = "huggingface",
        checkpoint_path: str = "checkpoints/latest.pt",
        best_model_path: str = "best_gpt_model.pt",
        final_model_path: str = "gpt_model.pt",
        learning_curve_path: str = "learning_curve.png",
    ):
        self.tokenizer_path = tokenizer_path
        self.tokenizer_type = tokenizer_type
        self.checkpoint_path = checkpoint_path
        self.best_model_path = best_model_path
        self.final_model_path = final_model_path
        self.learning_curve_path = learning_curve_path

        self.vocab_size_cfg = vocab_size
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.max_steps_per_epoch = max_steps_per_epoch
        self.warmup_steps = warmup_steps
        self.device_pref = device

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.dropout = dropout

        self.tokenizer = None
        self.train_dataloader = None
        self.val_dataloader = None
        self.test_dataloader = None
        self.max_steps_this_epoch = None
        self.run_device = None
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.start_epoch = 0
        self.total_training_steps = None

        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float("inf")
        self.monitor = None

    @staticmethod
    def encode_wikitext_split(tokenizer, hf_dataset, split_name, eos_token="<eos>"):
        eos_id = tokenizer.token_to_id(eos_token)
        token_ids = []
        for line in hf_dataset[split_name]["text"]:
            line = line.strip()
            if not line:
                continue
            token_ids.extend(tokenizer.encode(line))
            if eos_id is not None:
                token_ids.append(eos_id)
        return token_ids

    def _load_tokenizer_and_data(self):
        self.tokenizer = TokenizerFactory.create(self.tokenizer_type, self.tokenizer_path)
        tok_vocab = self.tokenizer.get_vocab_size()
        if tok_vocab != self.vocab_size_cfg:
            raise ValueError(
                f"config vocab_size ({self.vocab_size_cfg}) != tokenizer vocab size ({tok_vocab})."
            )

        wikitext = load_dataset("wikitext", "wikitext-103-raw-v1")
        token_ids_train = self.encode_wikitext_split(self.tokenizer, wikitext, "train")
        token_ids_val = self.encode_wikitext_split(self.tokenizer, wikitext, "validation")
        token_ids_test = self.encode_wikitext_split(self.tokenizer, wikitext, "test")

        print(f"Train tokens: {len(token_ids_train)}")
        print(f"Val tokens: {len(token_ids_val)}")
        print(f"Test tokens: {len(token_ids_test)}")

        train_dataset = LMDataset(token_ids_train, self.max_seq_len, stride=self.max_seq_len)
        val_dataset = LMDataset(token_ids_val, self.max_seq_len, stride=self.max_seq_len)
        test_dataset = LMDataset(token_ids_test, self.max_seq_len, stride=self.max_seq_len)

        self.train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        self.test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        self.max_steps_this_epoch = min(len(self.train_dataloader), self.max_steps_per_epoch)
        if self.max_steps_this_epoch < 1:
            raise RuntimeError(
                "No training steps: empty train set or batch_size too large for data length."
            )

    def _build_model(self):
        self.run_device = self.device_pref if torch.cuda.is_available() else "cpu"
        self.model = (
            TransformerModelBuilder()
            .with_vocab_size(self.vocab_size_cfg)
            .with_max_seq_len(self.max_seq_len)
            .with_d_model(self.d_model)
            .with_n_heads(self.n_heads)
            .with_n_layers(self.n_layers)
            .with_d_ff(self.d_ff)
            .with_dropout(self.dropout)
            .build()
        ).to(self.run_device)

    def _setup_optimizer_and_scheduler(self):
        self.optimizer = AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.total_training_steps = self.max_steps_this_epoch * self.epochs
        if self.total_training_steps < 1:
            raise RuntimeError("epochs must be >= 1 when there is at least one step per epoch.")

        if os.path.exists(self.checkpoint_path):
            self.model, self.optimizer, self.start_epoch, _ = load_checkpoint(
                self.model,
                self.optimizer,
                self.checkpoint_path,
                self.run_device,
            )
            print(f"Resuming at epoch {self.start_epoch + 1}/{self.epochs}")

        completed_opt_steps = self.start_epoch * self.max_steps_this_epoch
        scheduler_last_epoch = completed_opt_steps - 1 if completed_opt_steps > 0 else -1
        self.scheduler = get_lr_scheduler(
            self.optimizer,
            self.warmup_steps,
            self.total_training_steps,
            last_epoch=scheduler_last_epoch,
        )

    def train_one_epoch(self):
        self.model.train()
        total_loss = 0.0
        effective_steps = 0

        for _, (x, y) in enumerate(self.train_dataloader):
            if effective_steps >= self.max_steps_per_epoch:
                break

            x, y = x.to(self.run_device), y.to(self.run_device)
            self.optimizer.zero_grad()

            logits = self.model(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                y.view(-1),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()
            effective_steps += 1

            if effective_steps % 1000 == 0:
                print(f"step {effective_steps}, loss = {loss.item():.4f}")

        return total_loss / max(effective_steps, 1)

    def eval_loss(self, dataloader, max_eval_steps=None):
        self.model.eval()
        total_loss = 0.0
        steps = 0

        with torch.no_grad():
            for step, (x, y) in enumerate(dataloader):
                if max_eval_steps is not None and step >= max_eval_steps:
                    break

                x, y = x.to(self.run_device), y.to(self.run_device)
                logits = self.model(x)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    y.view(-1),
                )
                total_loss += loss.item()
                steps += 1

                if steps % 500 == 0:
                    print(f"eval step {steps}, loss = {loss.item():.4f}")

        return total_loss / max(steps, 1)

    def train(self):
        self.monitor = TrainingMonitor()
        self.monitor.subscribe(ConsoleTrainingObserver())
        self.monitor.start(self.epochs, self.max_steps_this_epoch, self.run_device)

        for epoch in range(self.start_epoch, self.epochs):
            avg_loss = self.train_one_epoch()
            avg_val_loss = self.eval_loss(self.val_dataloader, max_eval_steps=2000)

            self.train_losses.append(avg_loss)
            self.val_losses.append(avg_val_loss)

            is_new_best = avg_val_loss < self.best_val_loss
            if is_new_best:
                self.best_val_loss = avg_val_loss
                torch.save(self.model.state_dict(), self.best_model_path)

            self.monitor.epoch_end(
                epoch + 1,
                self.epochs,
                avg_loss,
                avg_val_loss,
                self.best_val_loss,
                is_new_best,
            )

            save_checkpoint(
                model=self.model,
                optimizer=self.optimizer,
                next_epoch=epoch + 1,
                loss=avg_val_loss,
                filepath=self.checkpoint_path,
            )

        if self.train_losses:
            plt.plot(range(1, len(self.train_losses) + 1), self.train_losses, label="Training Loss")
            plt.plot(range(1, len(self.val_losses) + 1), self.val_losses, label="Validation Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("Learning Curve")
            plt.legend()
            plt.grid(True)
            plt.savefig(self.learning_curve_path)
            plt.close()

        self.monitor.train_finish(self.train_losses, self.val_losses)

    def evaluate_test_and_save(self):
        if not os.path.isfile(self.best_model_path):
            raise FileNotFoundError(
                f"{self.best_model_path} not found; training may not have completed any epoch."
            )

        self.model.load_state_dict(torch.load(self.best_model_path, map_location=self.run_device))
        print("Loaded best model for test evaluation.")
        test_loss = self.eval_loss(self.test_dataloader, max_eval_steps=None)
        try:
            perplexity = math.exp(test_loss)
        except OverflowError:
            perplexity = float("inf")

        self.monitor.test(test_loss, perplexity)

        torch.save(self.model.state_dict(), self.final_model_path)
        print(f"training finished, model saved to {self.final_model_path}")

    def run(self):
        self._load_tokenizer_and_data()
        self._build_model()
        self._setup_optimizer_and_scheduler()
        self.train()
        self.evaluate_test_and_save()


# Backward-compatible alias, so existing imports still work.
LMTrainer = TransformerTrainer


def main():
    trainer = TransformerTrainer()
    trainer.run()


if __name__ == "__main__":
    main()
