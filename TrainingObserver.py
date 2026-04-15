from abc import ABC, abstractmethod

class TrainingObserver(ABC):
    @abstractmethod
    def on_start(self, total_epochs, max_steps_per_epoch, run_device):
        pass
    @abstractmethod
    def on_epoch_end(
        self,
        epoch,
        total_epochs,
        train_loss,
        val_loss,
        best_val_loss,
        is_new_best,
    ):
        pass
        
    @abstractmethod
    def on_train_finish(self, train_losses, val_losses):
        pass
        
    @abstractmethod
    def on_test(self, test_loss, perplexity):
        pass

# Subject
class TrainingMonitor:
    def __init__(self):
        self._observers = []

    def subscribe(self, observer):
        self._observers.append(observer)

    def unsubscribe(self, observer):
        self._observers.remove(observer)

    def start(self, total_epochs, max_steps_per_epoch, run_device):
        for o in self._observers:
            o.on_start(total_epochs, max_steps_per_epoch, run_device)

    def epoch_end(
        self,
        epoch,
        total_epochs,
        train_loss,
        val_loss,
        best_val_loss,
        is_new_best,
    ):
        for o in self._observers:
            o.on_epoch_end(epoch, total_epochs, train_loss, val_loss, best_val_loss, is_new_best)

    def train_finish(self, train_losses, val_losses):
        for o in self._observers:
            o.on_train_finish(train_losses, val_losses)

    def test(self, test_loss, perplexity):
        for o in self._observers:
            o.on_test(test_loss, perplexity)

# Concrete Observer
class ConsoleTrainingObserver(TrainingObserver):
    def on_start(self, total_epochs, max_steps_per_epoch, run_device):
        print(
            f"[monitor] device={run_device} | epochs={total_epochs} | steps/epoch (cap)={max_steps_per_epoch}"
        )

    def on_epoch_end(
        self,
        epoch,
        total_epochs,
        train_loss,
        val_loss,
        best_val_loss,
        is_new_best,
    ):
        print(f"Epoch {epoch}/{total_epochs}, train_loss = {train_loss:.4f}, val_loss = {val_loss:.4f}")
        if is_new_best:
            print(f"Best model saved with val_loss = {best_val_loss:.4f}")

    def on_train_finish(self, train_losses, val_losses):
        if train_losses:
            print(
                f"[monitor] train done | last train_loss={train_losses[-1]:.4f} | last val_loss={val_losses[-1]:.4f}"
            )

    def on_test(self, test_loss, perplexity):
        print(f"[monitor] test_loss={test_loss:.4f} | perplexity={perplexity:.4f}")
