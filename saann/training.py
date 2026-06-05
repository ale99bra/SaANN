from . import backend as BE
from .losses import cross_entropy_logits, cross_entropy_logits_der
import os

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

    # Tokenizer
    arrays["tokenizer_stoi"] = tokenizer.stoi
    arrays["tokenizer_itos"] = tokenizer.itos

    BE.xp.savez(path, **arrays)


def load_model(path, model, optimizer, tokenizer, scheduler):
    data = BE.xp.load(path, allow_pickle=True)

    # Token embedding
    model.token_emb.W[...] = data["W_tok"]

    # Positional embedding
    model.pos_emb.W[...] = data["W_pos"]

    # Output projection
    model.W_out[...] = data["W_out"]
    model.b_out[...] = data["b_out"]

    # Blocks
    for i, block in enumerate(model.blocks):
        block.attn.W_q[...] = data[f"W_q_{i}"]
        block.attn.W_k[...] = data[f"W_k_{i}"]
        block.attn.W_v[...] = data[f"W_v_{i}"]
        block.attn.W_o[...] = data[f"W_o_{i}"]

        block.ln1.gamma[...] = data[f"gamma1_{i}"]
        block.ln1.beta[...] = data[f"beta1_{i}"]

        block.ln2.gamma[...] = data[f"gamma2_{i}"]
        block.ln2.beta[...] = data[f"beta2_{i}"]

        block.ffn.W1[...] = data[f"W1_{i}"]
        block.ffn.b1[...] = data[f"b1_{i}"]
        block.ffn.W2[...] = data[f"W2_{i}"]
        block.ffn.b2[...] = data[f"b2_{i}"]

    # Optimizer state
    for k in optimizer.m.keys():
        optimizer.m[k][...] = data[f"opt_m_{k}"]
    for k in optimizer.v.keys():
        optimizer.v[k][...] = data[f"opt_v_{k}"]

    # Tokenizer
    tokenizer.stoi = data["tokenizer_stoi"].item()
    tokenizer.itos = data["tokenizer_itos"].item()

    scheduler.step_num = data["scheduler_step"].item()

    scheduler.step()   # recompute LR for current step
    scheduler.step_num = data["scheduler_step"].item()
