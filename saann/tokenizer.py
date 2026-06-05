# tokenizer.py
# Copyright (c) 2026 Alessio Branda
# Licensed under the MIT License

class CharTokenizer:
    def __init__(self, text):
        # Build vocabulary
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)

        # Maps
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}

    def encode(self, text):
        """Convert string → list of token IDs"""
        return [self.stoi[ch] for ch in text]

    def decode(self, tokens):
        """Convert list of token IDs → string"""
        return ''.join(self.itos[t] for t in tokens)

class ByteTokenizer:
    def __init__(self):
        # 256 byte values → 256 tokens
        self.vocab_size = 256
        
        # stoi: byte → int
        self.stoi = {i: i for i in range(256)}
        
        # itos: int → byte
        self.itos = {i: i for i in range(256)}

    def encode(self, text: str):
        # Convert string → bytes → list of ints
        return list(text.encode("utf-8"))

    def decode(self, tokens):
        # Convert list of ints → bytes → string
        return bytes(tokens).decode("utf-8", errors="replace")

