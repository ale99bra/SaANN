from . import backend as BE
from .losses import cross_entropy_logits, cross_entropy_logits_der
import os
from .transformer.transformer_model import TransformerModel
from .tokenizer import ByteTokenizer
from .gradients import AdamW
import numpy as np

def train_transformer_step(model, optimizer, tokens, targets):

    #forward pass
    logits = model.forward(tokens)

    #loss
    loss = cross_entropy_logits(logits=logits, target_ids=targets)

    #backward pass
    d_logits = cross_entropy_logits_der(logits=logits, target_ids=targets)
    model.backward(d_logits)

    #gradient clipping
    for name, p in optimizer.params.items():
        if name.startswith("d"):
            BE.xp.clip(p, -1.0, 1.0, out=p)

    #optimizer
    optimizer.step()

    #reset
    model.zero_grad()

    return loss

class LRScheduler:
    def __init__(self, optimizer, warmup_steps, total_steps,
                 max_lr=1e-4, min_lr=1e-5):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.step_num = 0

    def step(self):
        self.step_num += 1

        # Use BE.xp for all math operations
        xp = BE.xp

        if self.step_num < self.warmup_steps:
            # Linear warmup: lr = max_lr * (step / warmup_steps)
            lr = self.max_lr * (self.step_num / self.warmup_steps)

        else:
            # Cosine decay
            progress = (self.step_num - self.warmup_steps) / \
                       (self.total_steps - self.warmup_steps)

            # cosine = 0.5 * (1 + cos(pi * progress))
            cosine = 0.5 * (1.0 + xp.cos(xp.pi * progress))

            lr = self.min_lr + (self.max_lr - self.min_lr) * cosine

        # Update optimizer LR
        self.optimizer.lr = float(lr)
        return lr


def train_transformer(
    model,
    optimizer,
    data,
    batch_size,
    seq_len,
    epochs=1000,
    checkpoint_every=50,
    checkpoint_dir="checkpoints",
    tokenizer=None
):

    os.makedirs(checkpoint_dir, exist_ok=True)

    batches = make_batches(tokens=data, batch_size=batch_size, seq_len=seq_len)

    tot_steps = epochs * len(batches)

    log_interval = max(1, epochs // 10)

    scheduler = LRScheduler(
        optimizer=optimizer,
        warmup_steps=int(0.05*tot_steps),
        total_steps=tot_steps,
        max_lr=1e-4,
        min_lr=1e-5
    )


    for epoch in range(epochs):
        for tokens, targets in batches:
            scheduler.step()
            loss = train_transformer_step(model, optimizer, tokens, targets)

        if epoch == 0:
            print(f"Initial loss: {loss:.4f}")
        elif epoch % log_interval == 0:
            print(f"{int((epoch/epochs)*100)}% - Loss: {loss:.4f} (LR={optimizer.lr:.6f})")

        if (epoch + 1) % checkpoint_every == 0:
            save_model(
                model,
                optimizer,
                tokenizer,
                scheduler,
                path=os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.npz")
            )

    save_model(
        model,
        optimizer,
        tokenizer,
        scheduler,
        path=os.path.join(checkpoint_dir, f"checkpoint_final.npz")
    )

    print(f"Final loss: {loss:.4f}")

def train_transformer_full(model, optimizer, tokens, targets, steps = 1000):
    for step in range(1, steps+1):
        loss = train_transformer_step(model=model, optimizer=optimizer, tokens=tokens, targets=targets)
        if step == 1:
            print(f"Initial loss: {loss:.4f}")
        elif step % (steps//10) == 0:
            print(f"{int((step/steps)*100)}% - Loss: {loss:.4f}")
    
    print(f"Final loss: {loss:.4f}")

def make_batches(tokens, batch_size, seq_len):

    tot_tokens = tokens.shape[0]
    num_seq = (tot_tokens - 1)//seq_len

    tokens = tokens[: num_seq*seq_len +1]

    #sequences
    inputs = tokens[:-1].reshape(num_seq, seq_len)
    targets = tokens[1:].reshape(num_seq, seq_len)

    #batches
    batches = []
    for i in range(0, num_seq, batch_size):
        batch_inputs = inputs[i:i+batch_size]
        batch_targets = targets[i:i+batch_size]
        batches.append((batch_inputs, batch_targets))

    return batches

def save_model(model, optimizer, tokenizer, scheduler, path="checkpoint.npz"):
    arrays = {}

    #Architecture 
    arrays["arch_vocab_size"] = model.vocab_size
    arrays["arch_embed_dim"] = model.embed_dim
    arrays["arch_num_heads"] = model.num_heads
    arrays["arch_ff_hidden_dim"] = model.ff_hidden_dim
    arrays["arch_num_layers"] = model.num_layers
    arrays["arch_max_seq_len"] = model.max_seq_len
    arrays["arch_learned_positional"] = model.learned_positional

    # Token embedding
    arrays["W_tok"] = model.token_emb.W
    arrays["b_tok"] = model.token_emb.b if hasattr(model.token_emb, "b") else None

    # Positional embedding
    arrays["W_pos"] = model.pos_emb.W

    # Output projection
    arrays["W_out"] = model.W_out
    arrays["b_out"] = model.b_out

    # Scheduler
    arrays["scheduler_step"] = scheduler.step_num
    arrays["scheduler_warmup"] = scheduler.warmup_steps
    arrays["scheduler_total"] = scheduler.total_steps
    arrays["scheduler_max_lr"] = scheduler.max_lr
    arrays["scheduler_min_lr"] = scheduler.min_lr

    # Blocks
    for i, block in enumerate(model.blocks):
        arrays[f"W_q_{i}"] = block.attn.W_q
        arrays[f"W_k_{i}"] = block.attn.W_k
        arrays[f"W_v_{i}"] = block.attn.W_v
        arrays[f"W_o_{i}"] = block.attn.W_o

        arrays[f"gamma1_{i}"] = block.ln1.gamma
        arrays[f"beta1_{i}"] = block.ln1.beta

        arrays[f"gamma2_{i}"] = block.ln2.gamma
        arrays[f"beta2_{i}"] = block.ln2.beta

        arrays[f"W1_{i}"] = block.ffn.W1
        arrays[f"b1_{i}"] = block.ffn.b1
        arrays[f"W2_{i}"] = block.ffn.W2
        arrays[f"b2_{i}"] = block.ffn.b2

    # Optimizer state
    for k, v in optimizer.m.items():
        arrays[f"opt_m_{k}"] = v
    for k, v in optimizer.v.items():
        arrays[f"opt_v_{k}"] = v
    
    arrays["optimizer_lr"] = optimizer.lr
    arrays["optimizer_wd"] = optimizer.wd

    # Tokenizer
    arrays["tokenizer_stoi"] = tokenizer.stoi
    arrays["tokenizer_itos"] = tokenizer.itos

    BE.xp.savez(path, **arrays)


def load_model(path):
    data = BE.xp.load(path, allow_pickle=True)

    data = BE.to_numpy(data)

    data = np.load(path, allow_pickle=True)

    to_xp = BE.xp.asarray

    # 1. Reconstruct architecture
    model = TransformerModel(
        vocab_size=int(data["arch_vocab_size"]),
        embed_dim=int(data["arch_embed_dim"]),
        num_heads=int(data["arch_num_heads"]),
        ff_hidden_dim=int(data["arch_ff_hidden_dim"]),
        num_layers=int(data["arch_num_layers"]),
        max_seq_len=int(data["arch_max_seq_len"]),
        learned_positional=bool(data["arch_learned_positional"])
    )

    # 2. Reconstruct tokenizer
    tokenizer = ByteTokenizer()
    tokenizer.stoi = data["tokenizer_stoi"].item()
    tokenizer.itos = data["tokenizer_itos"].item()

    # 3. Reconstruct optimizer
    optimizer = AdamW(model.get_params(),
                      learning_rate=float(data["optimizer_lr"]),
                      wd=float(data["optimizer_wd"]))

    # Load optimizer moments
    for k in optimizer.m.keys():
        optimizer.m[k][...] = to_xp(data[f"opt_m_{k}"])
    for k in optimizer.v.keys():
        optimizer.v[k][...] = to_xp(data[f"opt_v_{k}"])

    # 4. Reconstruct scheduler
    scheduler = LRScheduler(
        optimizer=optimizer,
        warmup_steps=int(data["scheduler_warmup"]),
        total_steps=int(data["scheduler_total"]),
        max_lr=float(data["scheduler_max_lr"]),
        min_lr=float(data["scheduler_min_lr"])
    )
    scheduler.step_num = int(data["scheduler_step"])

    # Recompute LR for this step
    current = scheduler.step_num
    scheduler.step_num = current - 1
    scheduler.step()
    scheduler.step_num = current

    # 5. Load model weights
    model.token_emb.W[...] = to_xp(data["W_tok"])
    model.pos_emb.W[...] = to_xp(data["W_pos"])
    model.W_out[...] = to_xp(data["W_out"])
    model.b_out[...] = to_xp(data["b_out"])

    for i, block in enumerate(model.blocks):
        block.attn.W_q[...] = to_xp(data[f"W_q_{i}"])
        block.attn.W_k[...] = to_xp(data[f"W_k_{i}"])
        block.attn.W_v[...] = to_xp(data[f"W_v_{i}"])
        block.attn.W_o[...] = to_xp(data[f"W_o_{i}"])

        block.ln1.gamma[...] = to_xp(data[f"gamma1_{i}"])
        block.ln1.beta[...] = to_xp(data[f"beta1_{i}"])

        block.ln2.gamma[...] = to_xp(data[f"gamma2_{i}"])
        block.ln2.beta[...] = to_xp(data[f"beta2_{i}"])

        block.ffn.W1[...] = to_xp(data[f"W1_{i}"])
        block.ffn.b1[...] = to_xp(data[f"b1_{i}"])
        block.ffn.W2[...] = to_xp(data[f"W2_{i}"])
        block.ffn.b2[...] = to_xp(data[f"b2_{i}"])

    return model, optimizer, tokenizer, scheduler

