# generation.py
# Copyright (c) 2026 Alessio Branda
# Licensed under the MIT License

from . import backend as BE
from . import activation_functions as AF

def generate_greedy(model, start_token, max_new_tokens):

    tokens = start_token.copy()

    for i in range(max_new_tokens):
        #forward
        logits = model.forward(tokens)

        next_logits = logits[:, -1, :]

        #greedy decoding (get the highest prob.)
        next_token = BE.xp.argmax(next_logits, axis=-1)

        next_token = next_token.reshape(1, 1)
        tokens = BE.xp.concatenate([tokens, next_token], axis=1)
    
    return tokens

def generate_temperature(model, start_tokens, max_new_tokens, temperature=1.0):
    tokens = start_tokens.copy()

    for _ in range(max_new_tokens):
        tokens = tokens.reshape(1, -1)
        logits = model.forward(tokens)
        next_logits = logits[:, -1, :]
        probs = AF.softmax(next_logits/temperature)
        next_token = BE.xp.random.choice(probs.shape[-1], p = probs[0])
        next_token = BE.xp.array([[next_token]])
        tokens = BE.xp.concatenate([tokens, next_token], axis=1)

        if tokens.shape[1] > model.max_seq_len:
            tokens = tokens[:, -model.max_seq_len]

    return tokens

def generate_top_k(model, start_tokens, max_new_tokens, k=10, temperature=1.0):

    tokens = start_tokens.copy()

    for _ in range(max_new_tokens):

        tokens = tokens.reshape(1, -1)

        #forward
        logits = model.forward(tokens)
        next_logits = logits[:, -1, :]

        #temperature
        next_logits = next_logits / temperature

        #top-k idxs
        top_k_indices = BE.xp.argsort(next_logits[0])[-k:]

        #top-k logits
        top_k_logits = next_logits[0, top_k_indices]

        #softmax
        top_k_probs = AF.softmax(top_k_logits.reshape(1, -1))[0]

        next_token = BE.xp.random.choice(top_k_indices, p=top_k_probs)
        next_token = BE.xp.array([[int(next_token)]])
        tokens = BE.xp.concatenate([tokens, next_token], axis=1)

    return tokens

def generate_top_p(model, start_tokens, max_new_tokens, p=0.9, temperature=1.0, rep_penalty = 1.2):

    tokens = start_tokens.copy()

    for _ in range(max_new_tokens):

        tokens = tokens.reshape(1, -1)

        # Forward
        logits = model.forward(tokens)
        next_logits = logits[:, -1, :] 

        #penalty
        next_logits = repetition_penalty(next_logits, tokens, rep_penalty)  

        # Apply temperature
        next_logits = next_logits / temperature

        #probabilities
        probs = AF.softmax(next_logits)[0]  # shape (V,)

        # Sort probabilities descending
        sorted_indices = BE.xp.argsort(-probs)
        sorted_probs = probs[sorted_indices]

        # Compute cumulative probability
        cumulative = BE.xp.cumsum(sorted_probs)

        # Find cutoff where cumulative >= p
        cutoff = BE.xp.searchsorted(cumulative, p)

        # Keep only tokens up to cutoff
        top_p_indices = sorted_indices[:cutoff + 1]
        top_p_probs = sorted_probs[:cutoff + 1]

        # Renormalize
        top_p_probs = top_p_probs / BE.xp.sum(top_p_probs)

        # Sample
        if BE.xp.__name__ == "cupy":
            next_token = BE.xp.random.choice(top_p_indices, size=1, p=top_p_probs)[0]
        else:
            next_token = BE.xp.random.choice(top_p_indices, p=top_p_probs)

        # Append
        next_token = BE.xp.array([[int(next_token)]])
        tokens = BE.xp.concatenate([tokens, next_token], axis=1)

    return tokens

def repetition_penalty(logits, tokens, penalty):
    # Flatten token list
    used_tokens = tokens[0]

    for t in used_tokens:
        t = int(t)
        if logits[0, t] > 0:
            logits[0, t] /= penalty
        else:
            logits[0, t] *= penalty

    return logits