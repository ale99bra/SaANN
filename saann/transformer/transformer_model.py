from .. import backend as BE
from .blocks import TransformerBlock
from .embeddings import TokenEmbedding, PositionalEmbedding

def causal_mask(seq_len, batch_size, xp):
    """
    returns: (batch, 1, seq_len, seq_len) with 1 for allowed, 0 for masked
    """
    i = xp.arange(seq_len)[:, None]
    j = xp.arange(seq_len)[None, :]
    mask = (j <= i).astype(xp.int32)
    mask = mask[None, None, :, :]
    mask = xp.broadcast_to(mask, (batch_size, 1, seq_len, seq_len))
    return mask

class TransformerModel:
    """
    GPT-style Transformer model:
    - token + positional embeddings
    - N TransformerBlocks
    - final linear projection to vocab logits
    """

    def __init__(self, vocab_size, embed_dim, num_heads, ff_hidden_dim, num_layers, max_seq_len, learned_positional):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_hidden_dim = ff_hidden_dim
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len

        self.token_emb = TokenEmbedding(vocab_size, embed_dim)
        self.pos_emb = PositionalEmbedding(max_seq_len, embed_dim, learned=learned_positional)

        self.blocks = [
            TransformerBlock(embed_dim, num_heads, ff_hidden_dim)
            for _ in range(num_layers)
        ]

        #final projection: embed_dim -> vocab_size
        self.W_out = BE.xp.random.uniform(-0.01, 0.01, (embed_dim, vocab_size))
        self.b_out = BE.xp.zeros((vocab_size,))

        self.d_W_out = BE.xp.zeros_like(self.W_out)
        self.d_b_out = BE.xp.zeros_like(self.b_out)

        #cache
        self.input_tokens = None
        self.x_embed = None
        self.x_blocks = None  #list of intermediate outputs

    def forward(self, input_tokens):
        """
        input_tokens: (batch, seq_len) int indices
        returns: logits (batch, seq_len, vocab_size)
        """
        self.input_tokens = input_tokens
        B, L = input_tokens.shape

        if L > self.max_seq_len:
            raise ValueError("Sequence length exceeds max_seq_len")

        tok = self.token_emb.forward(input_tokens)
        x = self.pos_emb.forward(tok)
        self.x_embed = x

        mask = causal_mask(L, B, BE.xp)

        self.x_blocks = []
        for block in self.blocks:
            x = block.forward(x, mask)
            self.x_blocks.append(x)

        # final projection
        B, L, E = x.shape
        x_2D = x.reshape(B * L, E)
        logits_2D = BE.xp.matmul(x_2D, self.W_out) + self.b_out  # (B*L, V)
        logits = logits_2D.reshape(B, L, self.vocab_size)
        return logits

    def backward(self, grad_logits):
        """
        grad_logits: (batch, seq_len, vocab_size)
        returns: None (gradients stored in parameters)
        """
        B, L, V = grad_logits.shape
        E = self.embed_dim

        # final projection grads
        grad_logits_2D = grad_logits.reshape(B * L, V)
        last_x = self.x_blocks[-1]
        last_x_2D = last_x.reshape(B * L, E)

        self.d_W_out += BE.xp.matmul(last_x_2D.T, grad_logits_2D)
        self.d_b_out += BE.xp.sum(grad_logits_2D, axis=0)

        # grad wrt last block output
        d_x = BE.xp.matmul(grad_logits_2D, BE.xp.transpose(self.W_out, (1, 0)))
        d_x = d_x.reshape(B, L, E)

        # back through blocks (reverse order)
        for i in reversed(range(self.num_layers)):
            d_x = self.blocks[i].backward(d_x)

        # back through embeddings
        # x_embed = token_emb + pos_emb
        self.token_emb.backward(d_x)
        self.pos_emb.backward(d_x)

    def zero_grad(self):
        self.token_emb.zero_grad()
        self.pos_emb.zero_grad()
        for block in self.blocks:
            block.zero_grad()
        self.d_W_out[...] = 0
        self.d_b_out[...] = 0