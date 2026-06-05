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
            self.d_W = BE.xp.zeros_like(self.W)
        else:
            self.W = self._build_sinusoidal_embeddings(seq_len, dim)
            self.d_W = None

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
            self.d_W += BE.xp.sum(grad_output, axis=0)
        return grad_output

    def zero_grad(self):
        if self.learned:
            self.d_W[...] = 0

class TokenEmbedding:
    def __init__(self, vocab_size, embed_dim):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

        self.W = BE.xp.random.uniform(-0.01, 0.01, (vocab_size, embed_dim))
        self.d_W = BE.xp.zeros_like(self.W)

        self.tokens = None

    def forward(self, tokens):
        """
        tokens: (batch, seq_len) int indices
        returns: (batch, seq_len, embed_dim)
        """
        self.tokens = tokens
        return self.W[tokens]

    def backward(self, grad_output):
        """
        grad_output: (batch, seq_len, embed_dim)
        accumulates gradients into d_W
        """
        B, L, E = grad_output.shape
        #zero local grad buffer for safety
        #(global zero_grad will clear d_W between steps)
        for b in range(B):
            for l in range(L):
                idx = int(self.tokens[b, l])
                self.d_W[idx] += grad_output[b, l]

    def zero_grad(self):
        self.d_W[...] = 0