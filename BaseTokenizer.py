from abc import ABC, abstractmethod
from typing import List, Optional


class BaseTokenizer(ABC):
    @abstractmethod
    def encode(self, text: str) -> List[int]:
        pass

    @abstractmethod
    def decode(self, ids: List[int]) -> str:
        pass

    @abstractmethod
    def get_vocab_size(self) -> int:
        pass

    @abstractmethod
    def token_to_id(self, token: str) -> Optional[int]:
        pass
