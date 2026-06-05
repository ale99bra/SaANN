# saann/transformer/blocks.py

from .. import backend as BE
from .attention import MultiHeadAttention
from .. import activation_functions as AF

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
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        # parameters (weights, bias) for the 2 layers
        self.W1 = BE.xp.random.uniform(-0.01, 0.01, (embed_dim, hidden_dim))
        self.b1 = BE.xp.zeros((hidden_dim,))

        self.W2 = BE.xp.random.uniform(-0.01, 0.01, (hidden_dim, embed_dim))
        self.b2 = BE.xp.zeros((embed_dim,))

        # gradients of parameters
        self.d_W1 = BE.xp.zeros_like(self.W1)
        self.d_b1 = BE.xp.zeros_like(self.b1)

        self.d_W2 = BE.xp.zeros_like(self.W2)
        self.d_b2 = BE.xp.zeros_like(self.b2)

        self.x = None
        self.h = None
        self.a = None

    def forward(self, x):
        """
        x: (batch, seq_len, embed_dim)
        returns: (batch, seq_len, embed_dim)
        """
        self.x = x
        self.h = BE.xp.matmul(x, self.W1) + self.b1 #pre-activation layer
        self.a = AF.reLU(self.h) #return BE.xp.maximum(0, x)

        out = BE.xp.matmul(self.a, self.W2) + self.b2

        return out

    def backward(self, grad_output):
        """
        grad_output: (batch, seq_len, embed_dim)
        returns: gradient wrt input x
        """
        B, L, E = grad_output.shape

        # update gradients of layer 2
        a_2D = self.a.reshape(B*L, self.hidden_dim)
        grad_out_2D = grad_output.reshape(B*L, E)
        self.d_W2 += BE.xp.matmul(a_2D.T, grad_out_2D)
        self.d_b2 += BE.xp.sum(grad_output, axis = (0,1))

        # gradient of act
        d_a = BE.xp.matmul(grad_output, BE.xp.transpose(self.W2, (1,0)))

        # ReLU backward
        d_h = AF.reLU_der(self.h) * d_a #reLU_der(f) -> return BE.xp.where(f > 0, 1, 0)

        # d_W1, d_b1
        x_2D = self.x.reshape(B*L, E)
        d_h_2D = d_h.reshape(B*L, self.hidden_dim)
        self.d_W1 += BE.xp.matmul(x_2D.T, d_h_2D)
        self.d_b1 += BE.xp.sum(d_h, axis=(0,1))

        d_x = BE.xp.matmul(d_h, BE.xp.transpose(self.W1, (1,0)))
        return d_x


    def zero_grad(self):
        self.d_W1[...] = 0
        self.d_b1[...] = 0
        self.d_W2[...] = 0
        self.d_b2[...] = 0

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
        self.embed_dim = embed_dim
        self.eps = eps

        # learnable parameters
        self.gamma = BE.xp.ones((embed_dim,))
        self.beta  = BE.xp.zeros((embed_dim,))

        # gradients
        self.d_gamma = BE.xp.zeros_like(self.gamma)
        self.d_beta  = BE.xp.zeros_like(self.beta)

        # Cache for backward
        self.x = None
        self.mean = None
        self.var = None
        self.x_hat = None
        self.inv_std = None

    def forward(self, x):
        """
        x: (batch, seq_len, embed_dim)
        returns: normalized tensor with same shape
        """
        self.x = x
        self.mean = BE.xp.mean(x, axis=-1, keepdims=True)
        self.var = BE.xp.var(x, axis=-1, keepdims=True)
        self.inv_std = 1.0 / BE.xp.sqrt(self.var + self.eps)
        self.x_hat = (x - self.mean) * self.inv_std
        out = self.x_hat * self.gamma + self.beta

        return out

    def backward(self, grad_output):
        """
        grad_output: (batch, seq_len, embed_dim)
        returns: gradient wrt input x
        """
        B, L, E = grad_output.shape

        # update d_gamma and d_beta
        self.d_gamma += BE.xp.sum(grad_output * self.x_hat, axis=(0,1))
        self.d_beta += BE.xp.sum(grad_output, axis=(0,1))

        # find d_x_hat
        d_x_hat = grad_output * self.gamma

        # backward per token
        N = E
        sum_d = BE.xp.sum(d_x_hat, axis=-1, keepdims=True)
        sums = BE.xp.sum(d_x_hat * self.x_hat, axis=-1, keepdims=True)

        d_x = (d_x_hat * N - sum_d - self.x_hat * sums) * (self.inv_std/N)

        return d_x

    def zero_grad(self):
        self.d_gamma[...] = 0
        self.d_beta[...] = 0

class ResidualConnection:
    """
    Residual connection wrapper:
    y = x + sublayer(x)
    """

    def __init__(self):
        self.x = None
        self.sublayer_out = None

    def forward(self, x, sublayer_fn):
        """
        x: (batch, seq_len, embed_dim)
        sublayer_fn: a callable that takes x and returns output
        returns: x + sublayer_fn(x)
        """
        self.x = x
        self.sublayer_out = sublayer_fn(x)

        return self.x + self.sublayer_out

    def backward(self, grad_output, sublayer_backward_fn):
        """
        grad_output: (batch, seq_len, embed_dim)
        returns: gradient wrt input x
        """
        grad_identity = grad_output # identity patch
        grad_sublayer = sublayer_backward_fn(grad_output)

        return grad_identity + grad_sublayer
    
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
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_hidden_dim = ff_hidden_dim

        # Sublayers
        self.ln1 = LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.res1 = ResidualConnection()

        self.ln2 = LayerNorm(embed_dim)
        self.ffn = FeedForward(embed_dim, ff_hidden_dim)
        self.res2 = ResidualConnection()

        # Cache for backward
        self.x = None
        self.x1 = None
        self.x2 = None

    def forward(self, x, mask=None):
        """
        x: (batch, seq_len, embed_dim)
        mask: optional (batch, 1, seq_len, seq_len)
        returns: (batch, seq_len, embed_dim)
        """
        self.x = x

        # block 2: LayerNorm -> MultiHeadAttention -> Residual
        ln1_out = self.ln1.forward(x)
        attn_out = self.attn.forward(ln1_out, mask)

        self.x1 = self.res1.forward(x, lambda x_: attn_out)

        # block 1: LayerNorm -> FFN -> Residual
        ln2_out = self.ln2.forward(self.x1)
        ffn_out = self.ffn.forward(ln2_out)

        self.x2 = self.res2.forward(self.x1, lambda x_: ffn_out)

        return self.x2

    def backward(self, grad_output):
        """
        grad_output: (batch, seq_len, embed_dim)
        returns: gradient wrt input x
        """

        # define backwards
        def ffn_backward(f):
            return self.ffn.backward(self.ln2.backward(f))
        
        def attn_backward(f):
            return self.attn.backward(self.ln1.backward(f))
        
        # backwards through res2
        grad_x1 = self.res2.backward(grad_output, ffn_backward)
        grad_x0 = self.res1.backward(grad_x1, attn_backward)

        return grad_x0

    def zero_grad(self):
        self.ln1.zero_grad()
        self.attn.zero_grad()
        self.ln2.zero_grad()
        self.ffn.zero_grad()