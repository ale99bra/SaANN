# attention.py
# Copyright (c) 2026 Alessio Branda
# Licensed under the MIT License

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
        self.scale = 1.0 / BE.xp.sqrt(d_k)

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
        sum_dat_at = BE.xp.sum(d_att * self.attention, axis=-1, keepdims=True)
        d_scores = (d_att - sum_dat_at) * self.attention

        if self.mask is not None:
            d_scores = BE.xp.where(self.mask == 0, 0.0, d_scores)

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

        if self.embed_dim % self.num_heads != 0:
            raise ValueError("Embedding dimension must be divisible by the number of heads.")

        # weights: (embed_dim, embed_dim)
        self.W_q = BE.xp.random.uniform(-0.01, 0.01, (embed_dim, embed_dim))
        self.W_k = BE.xp.random.uniform(-0.01, 0.01, (embed_dim, embed_dim))
        self.W_v = BE.xp.random.uniform(-0.01, 0.01, (embed_dim, embed_dim))
        self.W_o = BE.xp.random.uniform(-0.01, 0.01, (embed_dim, embed_dim))

        # gradients
        self.d_W_q = BE.xp.zeros_like(self.W_q)
        self.d_W_k = BE.xp.zeros_like(self.W_k)
        self.d_W_v = BE.xp.zeros_like(self.W_v)
        self.d_W_o = BE.xp.zeros_like(self.W_o)

        # attention module
        self.attn = ScaledDotProductAttention()

        self.x = None
        self.Q_lin = None
        self.K_lin = None
        self.V_lin = None
        self.Q_heads = None
        self.K_heads = None
        self.V_heads = None
        self.context_heads = None
        self.context_merged = None

    def _split_heads(self, x):
        """
        x: (batch, seq_len, embed_dim)
        returns: (batch, heads, seq_len, head_dim)
        """
        B, L, E = x.shape
        H = self.num_heads
        D = self.head_dim

        x = x.reshape(B, L, H, D)
        x = BE.xp.transpose(x, (0,2,1,3))

        return x

    def _merge_heads(self, x):
        """
        x: (batch, heads, seq_len, head_dim)
        returns: (batch, seq_len, embed_dim)
        """
        B, H, L, D = x.shape
        E = H * D

        x = BE.xp.transpose(x, (0,2,1,3))
        x = x.reshape(B, L, E)

        return x

    def forward(self, x, mask=None):
        """
        x: (batch, seq_len, embed_dim)
        mask: optional (batch, 1, seq_len, seq_len)
        returns: (batch, seq_len, embed_dim)
        """
        self.x = x

        # linear projections
        self.Q_lin = BE.xp.matmul(x, self.W_q)
        self.K_lin = BE.xp.matmul(x, self.W_k)
        self.V_lin = BE.xp.matmul(x, self.W_v)

        # split -> heads
        self.Q_heads = self._split_heads(self.Q_lin)
        self.K_heads = self._split_heads(self.K_lin)
        self.V_heads = self._split_heads(self.V_lin)

        # scaled dot product
        self.context_heads = self.attn.forward(self.Q_heads, self.K_heads, self.V_heads, mask)

        # merge
        self.context_merged = self._merge_heads(self.context_heads)

        out = BE.xp.matmul(self.context_merged, self.W_o)

        return out

    def backward(self, grad_output):
        """
        grad_output: (batch, seq_len, embed_dim)
        returns: gradient wrt input x
        """
        B, L, E = grad_output.shape

        # output
        context_2D = self.context_merged.reshape(B*L, E)
        grad_out_2D = grad_output.reshape(B*L, E)
        self.d_W_o += BE.xp.matmul(context_2D.T, grad_out_2D)
        
        # d_context_merge = grad_out x W_o^T
        d_context_merged = BE.xp.matmul(grad_output, BE.xp.transpose(self.W_o, (1,0)))

        # backwards to merge heads
        d_context_heads = self._split_heads(d_context_merged)

        # back to attention
        d_Q_heads, d_K_heads, d_V_heads = self.attn.backward(d_context_heads)

        # each head
        d_Q_lin = self._merge_heads(d_Q_heads)
        d_K_lin = self._merge_heads(d_K_heads)
        d_V_lin = self._merge_heads(d_V_heads)

        # back through linear projections
        x_2D = self.x.reshape(B*L, E)
        d_Q_2D = d_Q_lin.reshape(B*L, E)
        d_K_2D = d_K_lin.reshape(B*L, E)
        d_V_2D = d_V_lin.reshape(B*L, E)

        # update der of weights
        self.d_W_q += BE.xp.matmul(x_2D.T, d_Q_2D)
        self.d_W_k += BE.xp.matmul(x_2D.T, d_K_2D)
        self.d_W_v += BE.xp.matmul(x_2D.T, d_V_2D)

        # contributions for the x tensor -> d_x_p = d_P_lin x W_p^T
        d_x_q = BE.xp.matmul(d_Q_lin, BE.xp.transpose(self.W_q, (1,0)))
        d_x_k = BE.xp.matmul(d_K_lin, BE.xp.transpose(self.W_k, (1,0)))
        d_x_v = BE.xp.matmul(d_V_lin, BE.xp.transpose(self.W_v, (1,0)))

        d_x = d_x_q + d_x_k + d_x_v

        return d_x

    def zero_grad(self):
        self.d_W_q[...] = 0
        self.d_W_k[...] = 0
        self.d_W_v[...] = 0
        self.d_W_o[...] = 0
