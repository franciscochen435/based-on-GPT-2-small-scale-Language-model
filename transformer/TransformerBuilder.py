from .TransformerModel import TransformerModel

class TransformerModelBuilder:
    def __init__(self):
        self._vocab_size = None
        self._max_seq_len = None
        self._d_model = None
        self._n_heads = None
        self._n_layers = None
        self._d_ff = None
        self._dropout = None

    def with_vocab_size(self, n: int):
        self._vocab_size = n
        return self

    def with_max_seq_len(self, n: int):
        self._max_seq_len = n
        return self

    def with_d_model(self, n: int):
        self._d_model = n
        return self

    def with_n_heads(self, n: int):
        self._n_heads = n
        return self

    def with_n_layers(self, n: int):
        self._n_layers = n
        return self

    def with_d_ff(self, n: int):
        self._d_ff = n
        return self

    def with_dropout(self, p: float):
        self._dropout = p
        return self

    def build(self):
        return TransformerModel(
            vocab_size=self._vocab_size,
            max_seq_len=self._max_seq_len,
            d_model=self._d_model,
            n_heads=self._n_heads,
            n_layers=self._n_layers,
            d_ff=self._d_ff,
            dropout=self._dropout,
        )

class TransformerClassificationModelBuilder(TransformerModelBuilder):
    def __init__(self):
        super().__init__()
        self._num_labels = 2

    def with_num_labels(self, n: int):
        self._num_labels = n
        return self

    def build(self):
        return TransformerForSequenceClassification(
            vocab_size=self._vocab_size,
            max_seq_len=self._max_seq_len,
            d_model=self._d_model,
            n_heads=self._n_heads,
            n_layers=self._n_layers,
            d_ff=self._d_ff,
            dropout=self._dropout,
            num_labels=self._num_labels,
        )
