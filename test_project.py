import os
import tempfile
import unittest

import torch
import torch.nn.functional as F

from TransformerTrainer import TransformerTrainer, get_lr_scheduler
from checkpoint import load_checkpoint, save_checkpoint
from dataset import LMDataset
from TrainingObserver import TrainingMonitor, TrainingObserver
from transformer.Embedding import Embedding
from transformer.FeedForward import FeedForward
from transformer.SelfAttention import SelfAttention
from transformer.TransformerBuilder import TransformerModelBuilder
from transformer.TransformerBlock import TransformerBlock
from transformer.TransformerModel import TransformerModel


class RecordingObserver(TrainingObserver):
    def __init__(self):
        self.events = []

    def on_start(self, total_epochs, max_steps_per_epoch, run_device):
        self.events.append(("start", total_epochs, max_steps_per_epoch, run_device))

    def on_epoch_end(
        self,
        epoch,
        total_epochs,
        train_loss,
        val_loss,
        best_val_loss,
        is_new_best,
    ):
        self.events.append(
            ("epoch_end", epoch, total_epochs, train_loss, val_loss, best_val_loss, is_new_best)
        )

    def on_train_finish(self, train_losses, val_losses):
        self.events.append(("train_finish", list(train_losses), list(val_losses)))

    def on_test(self, test_loss, perplexity):
        self.events.append(("test", test_loss, perplexity))


class FakeTokenizer:
    def token_to_id(self, token):
        if token == "<eos>":
            return 99
        return None

    def encode(self, text):
        return [len(part) for part in text.split()]


class UnitTests(unittest.TestCase):
    def test_dataset_basic(self):
        dataset = LMDataset(token_ids=[10, 11, 12, 13, 14, 15, 16], seq_len=3, stride=2)

        self.assertEqual(len(dataset), 2)

        x0, y0 = dataset[0]
        x1, y1 = dataset[1]

        self.assertTrue(torch.equal(x0, torch.tensor([10, 11, 12])))
        self.assertTrue(torch.equal(y0, torch.tensor([11, 12, 13])))
        self.assertTrue(torch.equal(x1, torch.tensor([12, 13, 14])))
        self.assertTrue(torch.equal(y1, torch.tensor([13, 14, 15])))

    def test_dataset_short_input(self):
        dataset = LMDataset(token_ids=[1, 2, 3], seq_len=3, stride=1)
        self.assertEqual(len(dataset), 0)

    def test_embedding_shape(self):
        module = Embedding(vocab_size=50, d_model=16, max_seq_len=8, dropout=0.0)
        input_ids = torch.randint(0, 50, (2, 6))
        output = module(input_ids)
        self.assertEqual(output.shape, (2, 6, 16))

    def test_embedding_limit(self):
        module = Embedding(vocab_size=50, d_model=16, max_seq_len=4, dropout=0.0)
        input_ids = torch.randint(0, 50, (2, 5))
        with self.assertRaises(ValueError):
            module(input_ids)

    def test_attention_shape(self):
        module = SelfAttention(d_model=16, n_heads=4, max_seq_len=8, dropout=0.0)
        x = torch.randn(2, 6, 16)
        output = module(x)
        self.assertEqual(output.shape, (2, 6, 16))

    def test_attention_heads_check(self):
        with self.assertRaises(AssertionError):
            SelfAttention(d_model=10, n_heads=4, max_seq_len=8, dropout=0.0)

    def test_ffn_shape(self):
        module = FeedForward(d_model=16, d_ff=32, dropout=0.0)
        x = torch.randn(2, 5, 16)
        output = module(x)
        self.assertEqual(output.shape, (2, 5, 16))
        self.assertTrue(torch.isfinite(output).all())

    def test_block_shape(self):
        module = TransformerBlock(d_model=16, n_heads=4, d_ff=32, max_seq_len=8, dropout=0.0)
        x = torch.randn(2, 6, 16)
        output = module(x)
        self.assertEqual(output.shape, (2, 6, 16))

    def test_weight_tying(self):
        model = TransformerModel(
            vocab_size=64,
            max_seq_len=8,
            d_model=16,
            n_heads=4,
            n_layers=1,
            d_ff=32,
            dropout=0.0,
        )
        self.assertIs(model.lm_head.weight, model.embed.token_embed.weight)

    def test_model_shape(self):
        model = TransformerModel(
            vocab_size=64,
            max_seq_len=8,
            d_model=16,
            n_heads=4,
            n_layers=2,
            d_ff=32,
            dropout=0.0,
        )
        input_ids = torch.randint(0, 64, (3, 8))
        logits = model(input_ids)
        self.assertEqual(logits.shape, (3, 8, 64))

    def test_model_input_check(self):
        model = TransformerModel(
            vocab_size=64,
            max_seq_len=8,
            d_model=16,
            n_heads=4,
            n_layers=1,
            d_ff=32,
            dropout=0.0,
        )
        input_ids = torch.randint(0, 64, (2, 3, 4))
        with self.assertRaises(ValueError):
            model(input_ids)

    def test_builder(self):
        model = (
            TransformerModelBuilder()
            .with_vocab_size(128)
            .with_max_seq_len(12)
            .with_d_model(24)
            .with_n_heads(4)
            .with_n_layers(2)
            .with_d_ff(48)
            .with_dropout(0.0)
            .build()
        )

        self.assertIsInstance(model, TransformerModel)
        self.assertEqual(model.embed.max_seq_len, 12)
        self.assertEqual(model.embed.token_embed.num_embeddings, 128)

    def test_scheduler_basic(self):
        model = torch.nn.Linear(4, 4)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1.0)
        scheduler = get_lr_scheduler(optimizer, warmup_steps=2, total_steps=6)

        initial_lr = optimizer.param_groups[0]["lr"]
        lrs = []
        for _ in range(6):
            optimizer.step()
            scheduler.step()
            lrs.append(optimizer.param_groups[0]["lr"])

        self.assertLess(initial_lr, 1.0)
        self.assertAlmostEqual(max(lrs), 1.0, places=6)
        self.assertGreater(lrs[0], lrs[-1])
        self.assertGreaterEqual(lrs[-1], 0.01)

    def test_scheduler_bad_steps(self):
        model = torch.nn.Linear(4, 4)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1.0)
        with self.assertRaises(ValueError):
            get_lr_scheduler(optimizer, warmup_steps=1, total_steps=0)

    def test_encode_split(self):
        tokenizer = FakeTokenizer()
        hf_dataset = {
            "train": {
                "text": [
                    "hello world",
                    "   ",
                    "a bb ccc",
                ]
            }
        }

        token_ids = TransformerTrainer.encode_wikitext_split(tokenizer, hf_dataset, "train")
        self.assertEqual(token_ids, [5, 5, 99, 1, 2, 3, 99])

    def test_monitor(self):
        monitor = TrainingMonitor()
        observer = RecordingObserver()
        monitor.subscribe(observer)

        monitor.start(total_epochs=2, max_steps_per_epoch=5, run_device="cpu")
        monitor.epoch_end(
            epoch=1,
            total_epochs=2,
            train_loss=1.2,
            val_loss=1.0,
            best_val_loss=1.0,
            is_new_best=True,
        )
        monitor.train_finish([1.2], [1.0])
        monitor.test(test_loss=0.9, perplexity=2.46)

        self.assertEqual(observer.events[0], ("start", 2, 5, "cpu"))
        self.assertEqual(observer.events[1][0], "epoch_end")
        self.assertEqual(observer.events[2], ("train_finish", [1.2], [1.0]))
        self.assertEqual(observer.events[3], ("test", 0.9, 2.46))


class IntegrationTests(unittest.TestCase):
    def test_checkpoint_old_format(self):
        model = TransformerModel(
            vocab_size=32,
            max_seq_len=6,
            d_model=12,
            n_heads=4,
            n_layers=1,
            d_ff=24,
            dropout=0.0,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "legacy.pt")
            torch.save(
                {
                    "epoch": 4,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": 0.75,
                },
                path,
            )

            loaded_model = TransformerModel(
                vocab_size=32,
                max_seq_len=6,
                d_model=12,
                n_heads=4,
                n_layers=1,
                d_ff=24,
                dropout=0.0,
            )
            loaded_optimizer = torch.optim.AdamW(loaded_model.parameters(), lr=1e-3)

            _, _, start_epoch, saved_loss = load_checkpoint(
                loaded_model, loaded_optimizer, path, device="cpu"
            )

            self.assertEqual(start_epoch, 5)
            self.assertEqual(saved_loss, 0.75)

    def test_train_and_checkpoint(self):
        torch.manual_seed(7)

        token_ids = list(range(40))
        dataset = LMDataset(token_ids=token_ids, seq_len=8, stride=8)
        x, y = dataset[0]
        batch_x = x.unsqueeze(0)
        batch_y = y.unsqueeze(0)

        model = TransformerModel(
            vocab_size=64,
            max_seq_len=8,
            d_model=16,
            n_heads=4,
            n_layers=2,
            d_ff=32,
            dropout=0.0,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        before = model.embed.token_embed.weight.detach().clone()
        logits = model(batch_x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), batch_y.view(-1))
        loss.backward()
        optimizer.step()
        updated_logits = model(batch_x).detach()

        after = model.embed.token_embed.weight.detach()
        self.assertEqual(logits.shape, (1, 8, 64))
        self.assertTrue(torch.isfinite(loss))
        self.assertFalse(torch.allclose(before, after))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "checkpoints", "latest.pt")
            save_checkpoint(model, optimizer, next_epoch=3, loss=float(loss.item()), filepath=path)

            reloaded_model = TransformerModel(
                vocab_size=64,
                max_seq_len=8,
                d_model=16,
                n_heads=4,
                n_layers=2,
                d_ff=32,
                dropout=0.0,
            )
            reloaded_optimizer = torch.optim.AdamW(reloaded_model.parameters(), lr=1e-3)

            reloaded_model, reloaded_optimizer, start_epoch, saved_loss = load_checkpoint(
                reloaded_model,
                reloaded_optimizer,
                path,
                device="cpu",
            )

            reloaded_logits = reloaded_model(batch_x)
            self.assertEqual(start_epoch, 3)
            self.assertAlmostEqual(saved_loss, float(loss.item()), places=6)
            self.assertTrue(torch.allclose(updated_logits, reloaded_logits.detach(), atol=1e-4))


if __name__ == "__main__":
    unittest.main()
