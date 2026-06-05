from .. import backend as BE

class PositionalEmbedding:
    """
    Positional Embedding layer for Transformers.
    Supports:
    - learned positional embeddings
    - sinusoidal embeddings (no gradients)
    """

    def __init__(self, seq_len, dim, learned=True):
        self.seq_len = seq_len
        self.dim = dim
        self.learned = learned

        if learned:
            limit = 1.0 / BE.xp.sqrt(dim)
            self.W = BE.xp.random.uniform(-limit, limit, (seq_len, dim))
            self.dW = BE.xp.zeros_like(self.W)
        else:
            self.W = self._build_sinusoidal_embeddings(seq_len, dim)
            self.dW = None

    def _build_sinusoidal_embeddings(self, seq_len, dim):
        pe = BE.xp.zeros((seq_len, dim))
        position = BE.xp.arange(seq_len)[:, BE.xp.newaxis]

        # Use xp.log instead of math.log
        div_term = BE.xp.exp(
            BE.xp.arange(0, dim, 2) * (-BE.xp.log(10000.0) / dim)
        )

        pe[:, 0::2] = BE.xp.sin(position * div_term)
        pe[:, 1::2] = BE.xp.cos(position * div_term)

        return pe

    def forward(self, x):
        self.x = x
        return x + self.W[BE.xp.newaxis, :, :]

    def backward(self, grad_output):
        if self.learned:
            self.dW += BE.xp.sum(grad_output, axis=0)
        return grad_output

    def zero_grad(self):
        if self.learned:
            self.dW[...] = 0