# saann/transformer/blocks.py

from .. import backend as BE
from .attention import MultiHeadAttention

class FeedForward:
    """
    Position-wise Feed-Forward Network (FFN)
    FFN(x) = max(0, xW1 + b1) W2 + b2
    """

    def __init__(self, embed_dim, hidden_dim):
        """
        embed_dim: input/output dimension
        hidden_dim: inner layer dimension (usually 4 * embed_dim)
        """

        # Parameters
        self.W1 = BE.xp.random.uniform(-0.01, 0.01, (embed_dim, hidden_dim))
        self.b1 = BE.xp.zeros((hidden_dim,))

        self.W2 = BE.xp.random.uniform(-0.01, 0.01, (hidden_dim, embed_dim))
        self.b2 = BE.xp.zeros((embed_dim,))

        # Gradients
        self.dW1 = BE.xp.zeros_like(self.W1)
        self.db1 = BE.xp.zeros_like(self.b1)

        self.dW2 = BE.xp.zeros_like(self.W2)
        self.db2 = BE.xp.zeros_like(self.b2)

    def forward(self, x):
        """
        x: (batch, seq_len, embed_dim)
        returns: (batch, seq_len, embed_dim)
        """
        raise NotImplementedError

    def backward(self, grad_output):
        """
        grad_output: (batch, seq_len, embed_dim)
        returns: gradient wrt input x
        """
        raise NotImplementedError

    def zero_grad(self):
        self.dW1[...] = 0
        self.db1[...] = 0
        self.dW2[...] = 0
        self.db2[...] = 0

class LayerNorm:
    """
    Layer Normalization (applied over the last dimension)
    y = (x - mean) / sqrt(var + eps) * gamma + beta
    """

    def __init__(self, embed_dim, eps=1e-5):
        """
        embed_dim: dimension of the last axis to normalize
        eps: numerical stability constant
        """

        # Learnable parameters
        self.gamma = BE.xp.ones((embed_dim,))
        self.beta  = BE.xp.zeros((embed_dim,))

        # Gradients
        self.dgamma = BE.xp.zeros_like(self.gamma)
        self.dbeta  = BE.xp.zeros_like(self.beta)

        # Cache for backward
        self.eps = eps
        self.mean = None
        self.var = None
        self.x_centered = None
        self.inv_std = None

    def forward(self, x):
        """
        x: (batch, seq_len, embed_dim)
        returns: normalized tensor with same shape
        """
        raise NotImplementedError

    def backward(self, grad_output):
        """
        grad_output: (batch, seq_len, embed_dim)
        returns: gradient wrt input x
        """
        raise NotImplementedError

    def zero_grad(self):
        self.dgamma[...] = 0
        self.dbeta[...] = 0

class ResidualConnection:
    """
    Residual connection wrapper:
    y = x + sublayer(x)
    """

    def __init__(self):
        # Cache for backward
        self.x = None
        self.sublayer_out = None

    def forward(self, x, sublayer_fn):
        """
        x: (batch, seq_len, embed_dim)
        sublayer_fn: a callable that takes x and returns output
        returns: x + sublayer_fn(x)
        """
        raise NotImplementedError

    def backward(self, grad_output):
        """
        grad_output: (batch, seq_len, embed_dim)
        returns: gradient wrt input x
        """
        raise NotImplementedError
    
class TransformerBlock:
    """
    A single Transformer decoder block:
    
    x = x + MHA(LN(x))
    x = x + FFN(LN(x))
    """

    def __init__(self, embed_dim, num_heads, ff_hidden_dim):
        """
        embed_dim: model dimension
        num_heads: number of attention heads
        ff_hidden_dim: hidden dimension of the FFN (usually 4 * embed_dim)
        """

        # Sublayers
        self.ln1 = LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.res1 = ResidualConnection()

        self.ln2 = LayerNorm(embed_dim)
        self.ffn = FeedForward(embed_dim, ff_hidden_dim)
        self.res2 = ResidualConnection()

        # Cache for backward
        self.x = None

    def forward(self, x, mask=None):
        """
        x: (batch, seq_len, embed_dim)
        mask: optional (batch, 1, seq_len, seq_len)
        returns: (batch, seq_len, embed_dim)
        """
        raise NotImplementedError

    def backward(self, grad_output):
        """
        grad_output: (batch, seq_len, embed_dim)
        returns: gradient wrt input x
        """
        raise NotImplementedError

    def zero_grad(self):
        self.ln1.zero_grad()
        self.attn.zero_grad()
        self.ln2.zero_grad()
        self.ffn.zero_grad()