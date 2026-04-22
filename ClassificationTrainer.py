import os

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from datasets import load_dataset
from tokenizers import Tokenizer
from torch.optim import AdamW
from torch.utils.data import DataLoader

from BaseTrainer import BaseTrainer
from TransformerTrainer import get_lr_scheduler
from checkpoint import load_checkpoint, save_checkpoint
from config import *
from dataset import ClassificationDataset
from transformer.TransformerBuilder import TransformerClassificationModelBuilder


class ClassificationTrainer(BaseTrainer):
    def __init__(
        self,
        tokenizer_path: str = "tokenizer/trained_tokenizer/tokenizer.json",
        pretrained_model_path: str = "gpt_model.pt",
        checkpoint_path: str = "checkpoints/classification_latest.pt",
        best_model_path: str = "best_imdb_classifier.pt",
        final_model_path: str = "imdb_classifier.pt",
        learning_curve_path: str = "classification_learning_curve.png",
        num_labels: int = 2,
        validation_ratio: float = 0.1,
        max_train_samples=None,
        max_eval_samples=None,
    ):
        self.tokenizer_path = tokenizer_path
        self.pretrained_model_path = pretrained_model_path
        self.checkpoint_path = checkpoint_path
        self.best_model_path = best_model_path
        self.final_model_path = final_model_path
        self.learning_curve_path = learning_curve_path
        self.num_labels = num_labels
        self.validation_ratio = validation_ratio
        self.max_train_samples = max_train_samples
        self.max_eval_samples = max_eval_samples

        self.run_device = None
        self.tokenizer = None
        self.pad_token_id = 0
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.train_dataloader = None
        self.val_dataloader = None
        self.test_dataloader = None
        self.start_epoch = 0
        self.max_steps_this_epoch = None
        self.best_val_loss = float("inf")
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []

    @staticmethod
    def _find_pad_token_id(tokenizer):
        for token in ("<pad>", "[PAD]", "<unk>", "[UNK]"):
            token_id = tokenizer.token_to_id(token)
            if token_id is not None:
                return token_id
        return 0

    def _load_tokenizer_and_data(self):
        self.tokenizer = Tokenizer.from_file(self.tokenizer_path)
        self.pad_token_id = self._find_pad_token_id(self.tokenizer)

        imdb = load_dataset("imdb")
        split = imdb["train"].train_test_split(
            test_size=self.validation_ratio,
            seed=42,
        )
        train_split = split["train"]
        val_split = split["test"]
        test_split = imdb["test"]

        if self.max_train_samples is not None:
            train_split = train_split.select(range(min(self.max_train_samples, len(train_split))))
        if self.max_eval_samples is not None:
            val_split = val_split.select(range(min(self.max_eval_samples, len(val_split))))
            test_split = test_split.select(range(min(self.max_eval_samples, len(test_split))))

        train_dataset = ClassificationDataset(
            train_split, self.tokenizer, max_seq_len, self.pad_token_id
        )
        val_dataset = ClassificationDataset(
            val_split, self.tokenizer, max_seq_len, self.pad_token_id
        )
        test_dataset = ClassificationDataset(
            test_split, self.tokenizer, max_seq_len, self.pad_token_id
        )

        self.train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        self.test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        self.max_steps_this_epoch = min(len(self.train_dataloader), max_steps_per_epoch)

    def _build_model(self):
        self.run_device = device if torch.cuda.is_available() else "cpu"
        self.model = (
            TransformerClassificationModelBuilder()
            .with_vocab_size(vocab_size)
            .with_max_seq_len(max_seq_len)
            .with_d_model(d_model)
            .with_n_heads(n_heads)
            .with_n_layers(n_layers)
            .with_d_ff(d_ff)
            .with_dropout(dropout)
            .with_num_labels(self.num_labels)
            .build()
        ).to(self.run_device)

        if os.path.isfile(self.pretrained_model_path):
            state_dict = torch.load(self.pretrained_model_path, map_location=self.run_device)
            missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
            print(f"Loaded pretrained GPT weights from {self.pretrained_model_path}")
            if missing:
                print(f"Missing new classification parameters: {missing}")
            if unexpected:
                print(f"Unexpected parameters ignored: {unexpected}")
        else:
            print(f"{self.pretrained_model_path} not found; fine-tuning starts from random weights.")

    def _setup_optimizer_and_scheduler(self):
        decay_params = []
        no_decay_params = []
        for _, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if param.dim() >= 2:
                decay_params.append(param)
            else:
                no_decay_params.append(param)

        self.optimizer = AdamW(
            [
                {"params": decay_params, "weight_decay": weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=lr,
        )

        if os.path.exists(self.checkpoint_path):
            self.model, self.optimizer, self.start_epoch, _ = load_checkpoint(
                self.model, self.optimizer, self.checkpoint_path, self.run_device
            )
            print(f"Resuming classification fine-tuning at epoch {self.start_epoch + 1}/{epochs}")

        total_steps = self.max_steps_this_epoch * epochs
        completed_steps = self.start_epoch * self.max_steps_this_epoch
        scheduler_last_epoch = completed_steps - 1 if completed_steps > 0 else -1
        self.scheduler = get_lr_scheduler(
            self.optimizer, warmup_steps, total_steps, last_epoch=scheduler_last_epoch
        )

    def train_one_epoch(self):
        self.model.train()
        total_loss = 0.0
        effective_steps = 0

        for input_ids, attention_mask, labels in self.train_dataloader:
            if effective_steps >= max_steps_per_epoch:
                break

            input_ids = input_ids.to(self.run_device)
            attention_mask = attention_mask.to(self.run_device)
            labels = labels.to(self.run_device)

            self.optimizer.zero_grad()
            logits = self.model(input_ids, attention_mask)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()
            effective_steps += 1

            if effective_steps % 200 == 0:
                print(f"classification step {effective_steps}, loss = {loss.item():.4f}")

        return total_loss / max(effective_steps, 1)

    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        steps = 0

        with torch.no_grad():
            for input_ids, attention_mask, labels in dataloader:
                input_ids = input_ids.to(self.run_device)
                attention_mask = attention_mask.to(self.run_device)
                labels = labels.to(self.run_device)

                logits = self.model(input_ids, attention_mask)
                loss = F.cross_entropy(logits, labels)
                predictions = torch.argmax(logits, dim=-1)

                total_loss += loss.item()
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
                steps += 1

        return total_loss / max(steps, 1), correct / max(total, 1)

    def train(self):
        print(
            f"[classification] device={self.run_device} | epochs={epochs} | "
            f"steps/epoch cap={self.max_steps_this_epoch}"
        )
        for epoch in range(self.start_epoch, epochs):
            train_loss = self.train_one_epoch()
            val_loss, val_acc = self.evaluate(self.val_dataloader)

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)

            is_new_best = val_loss < self.best_val_loss
            if is_new_best:
                self.best_val_loss = val_loss
                torch.save(self.model.state_dict(), self.best_model_path)

            print(
                f"Epoch {epoch + 1}/{epochs}, train_loss={train_loss:.4f}, "
                f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
            )

            save_checkpoint(
                model=self.model,
                optimizer=self.optimizer,
                next_epoch=epoch + 1,
                loss=val_loss,
                filepath=self.checkpoint_path,
            )

        if self.train_losses:
            plt.plot(range(1, len(self.train_losses) + 1), self.train_losses, label="Training Loss")
            plt.plot(range(1, len(self.val_losses) + 1), self.val_losses, label="Validation Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("IMDb Classification Fine-Tuning")
            plt.legend()
            plt.grid(True)
            plt.savefig(self.learning_curve_path)
            plt.close()

    def evaluate_test_and_save(self):
        if not os.path.isfile(self.best_model_path):
            raise FileNotFoundError(
                f"{self.best_model_path} not found; classification training did not save a model."
            )
        self.model.load_state_dict(torch.load(self.best_model_path, map_location=self.run_device))
        test_loss, test_acc = self.evaluate(self.test_dataloader)
        print(f"[classification] test_loss={test_loss:.4f} | test_acc={test_acc:.4f}")
        torch.save(self.model.state_dict(), self.final_model_path)
        print(f"classification model saved to {self.final_model_path}")

    def run(self):
        self._load_tokenizer_and_data()
        self._build_model()
        self._setup_optimizer_and_scheduler()
        self.train()
        self.evaluate_test_and_save()


def main():
    trainer = ClassificationTrainer()
    trainer.run()


if __name__ == "__main__":
    main()
