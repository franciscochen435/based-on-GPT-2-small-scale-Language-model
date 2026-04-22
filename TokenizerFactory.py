from BaseTokenizer import BaseTokenizer
from HFTokenizerAdapter import HuggingFaceTokenizerAdapter

class TokenizerFactory:
    _registry = {
        "huggingface": HuggingFaceTokenizerAdapter,
    }

    @classmethod
    def create(cls, tokenizer_type: str, tokenizer_path: str) -> BaseTokenizer:
        adapter_cls = cls._registry.get(tokenizer_type)
        if adapter_cls is None:
            raise KeyError(tokenizer_type)
        return adapter_cls(tokenizer_path)
