from typing import List, Optional

from tokenizers import Tokenizer as _HFTokenizer

from BaseTokenizer import BaseTokenizer


class HuggingFaceTokenizerAdapter(BaseTokenizer):
    """Adapter that wraps the HuggingFace tokenizers.Tokenizer behind BaseTokenizer.

    The HuggingFace library returns an Encoding object from encode(), requiring
    callers to access .ids explicitly.  This adapter hides that detail so the
    rest of the codebase only depends on BaseTokenizer.
    """

    def __init__(self, tokenizer_path: str):
        self._tokenizer = _HFTokenizer.from_file(tokenizer_path)

    def encode(self, text: str) -> List[int]:
        return self._tokenizer.encode(text).ids

    def decode(self, ids: List[int]) -> str:
        return self._tokenizer.decode(ids)

    def get_vocab_size(self) -> int:
        return self._tokenizer.get_vocab_size()

    def token_to_id(self, token: str) -> Optional[int]:
        return self._tokenizer.token_to_id(token)
