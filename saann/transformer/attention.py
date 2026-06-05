from .. import backend as BE
from .. import activation_functions as AF


class ScaledDotProductAttention:
    """
    Core attention mechanism:
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
    """

    def __init__(self):
        self.Q = None
        self.K = None
        self.V = None
        self.scores = None
        self.attention = None
        self.mask = None
        self.scale = None

    def forward(self, Q, K, V, mask=None):
        """
        Q, K, V: (batch, heads, seq_len, head_dim)
        mask: optional (batch, 1, seq_len, seq_len)
        returns: (batch, heads, seq_len, head_dim)
        """
        self.Q = Q
        self.K = K
        self.V = V
        self.mask = mask

        d_k = Q.shape[-1]
        self.scale = 1 / BE.xp.sqrt(d_k)

        K_T = BE.xp.transpose(K, (0, 1, 3, 2))
        scores = BE.xp.matmul(Q, K_T) * self.scale

        if mask is not None:
            scores = BE.xp.where(mask == 0, -1e9, scores)
        
        # reshape to 2D for the softmax
        B, H, L, _ = scores.shape
        scores_2D = scores.reshape(B*H*L, L)

        attention_2D = AF.softmax(scores_2D)

        self.attention = attention_2D.reshape(B, H, L, L)
        self.scores = scores

        return BE.xp.matmul(self.attention, V)

    def backward(self, grad_output):
        """
        grad_output: same shape as forward output
        returns: gradients wrt Q, K, V
        """

        # derivative of V (d_V) is attention^T x grad_output
        attention_T = BE.xp.transpose(self.attention, (0,1,3,2))
        d_V = BE.xp.matmul(attention_T, grad_output)

        # derivative of attention (d_att) is grad_output x V^T
        V_T = BE.xp.transpose(self.V, (0,1,3,2))
        d_att = BE.xp.matmul(grad_output, V_T)

        # softmax derivative for attention
        sum_dat_at = BE.xp.sum(d_att * self.attention, axis=1, keepdims=True)
        d_scores = (d_att - sum_dat_at) * self.attention

        if self.mask is not None:
            d_scores = BE.xp.where(self.mask == 0, 0, d_scores)

        d_scores *= self.scale

        # derivative of Q = d_scores x K
        d_Q = BE.xp.matmul(d_scores, self.K)

        # derivative of K = d_scores^T x Q
        d_scores_T = BE.xp.transpose(d_scores, (0,1,3,2))
        d_K = BE.xp.matmul(d_scores_T, self.Q)

        return d_Q, d_K, d_V

class MultiHeadAttention:
    """
    Multi-Head Attention layer:
    - Linear projections for Q, K, V
    - Scaled dot-product attention
    - Output projection
    """

    def __init__(self, embed_dim, num_heads):
        """
        embed_dim: total embedding dimension
        num_heads: number of attention heads
        head_dim = embed_dim // num_heads
        """

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Weight matrices: (embed_dim, embed_dim)
        self.W_q = BE.xp.random.uniform(-0.01, 0.01, (embed_dim, embed_dim))
        self.W_k = BE.xp.random.uniform(-0.01, 0.01, (embed_dim, embed_dim))
        self.W_v = BE.xp.random.uniform(-0.01, 0.01, (embed_dim, embed_dim))
        self.W_o = BE.xp.random.uniform(-0.01, 0.01, (embed_dim, embed_dim))

        # Gradients
        self.dW_q = BE.xp.zeros_like(self.W_q)
        self.dW_k = BE.xp.zeros_like(self.W_k)
        self.dW_v = BE.xp.zeros_like(self.W_v)
        self.dW_o = BE.xp.zeros_like(self.W_o)

        # Internal attention module
        self.attn = ScaledDotProductAttention()

    def _split_heads(self, x):
        """
        x: (batch, seq_len, embed_dim)
        returns: (batch, heads, seq_len, head_dim)
        """
        raise NotImplementedError

    def _merge_heads(self, x):
        """
        x: (batch, heads, seq_len, head_dim)
        returns: (batch, seq_len, embed_dim)
        """
        raise NotImplementedError

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
        self.dW_q[...] = 0
        self.dW_k[...] = 0
        self.dW_v[...] = 0
        self.dW_o[...] = 0
