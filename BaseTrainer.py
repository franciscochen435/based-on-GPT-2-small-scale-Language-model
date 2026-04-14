from abc import ABC, abstractmethod


class BaseTrainer(ABC):
    @abstractmethod
    def train(self):
        """Run the training loop."""
        raise NotImplementedError

    @abstractmethod
    def evaluate_test_and_save(self):
        """Evaluate on test set and save final artifacts."""
        raise NotImplementedError

    @abstractmethod
    def run(self):
        """Execute the full trainer lifecycle."""
        raise NotImplementedError
