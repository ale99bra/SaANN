import unittest
import numpy as np
import os
from saann.transformer.transformer_model import TransformerModel
from saann.generation import generate_top_p
from saann.training import train_transformer, load_model, create_optimizer
from saann.tokenizer import CharTokenizer, ByteTokenizer

class TestTransformers(unittest.TestCase):
    """Test suite for Transformers"""

    def setUp(self):
        self.text = "This is a dummy example. The model will not learn anything useful. This is just to show the API of SaANN."

    def TestTokenizer(self):
        tokenizer = CharTokenizer(self.text)
        self.tokenizer = ByteTokenizer()
        encoded_text = self.tokenizer.encode(self.text)
        data = np.array(encoded_text, dtype=np.int32)
        split = int(0.9 * len(data))
        self.train_data = data[:split]
        self.val_data   = data[split:]

        self.len_input = len(self.train_data)
    
    def TestTransformerModel(self):
        vocab_size = self.tokenizer.vocab_size
        embed_dim = 512
        num_heads = 8
        ff_hidden_dim = 2048
        num_layers = 8
        max_seq_len = 512

        model = TransformerModel(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            ff_hidden_dim=ff_hidden_dim,
            num_layers=num_layers,
            max_seq_len=max_seq_len,
            learned_positional=True
        )

    def TestOptimizer(self):
        self.optimizer = create_optimizer(self.model)

    def TestTraining(self):
        train_transformer(model=self.model,
                  optimizer=self.optimizer,
                  data=self.train_data,
                  batch_size=32,
                  seq_len=64,
                  epochs=2,
                  checkpoint_every = 1,
                  checkpoint_dir="checkpoints",
                  tokenizer=self.tokenizer)
        
        if os.path.isdir("checkpoints"):
            if not os.path.isfile("checkpoints/checkpoint_epoch_1.npz"):
                raise FileExistsError("Checkpoints not created.")
            if not os.path.isfile("checkpoints/checkpoint_final.npz"):
                raise FileExistsError("Final checkpoint not created.")
        else:
            raise FileExistsError("Directory for checkpoints not created.")
    
    def TestLoadingModel(self):
        self.model, optimizer, self.tokenizer, scheduler = load_model("checkpoints/checkpoint_final.npz")
    
    def ValidationModel(self):
        start = np.array([self.val_data], dtype=np.int32)
        generated = generate_top_p(model=self.model, start_tokens=start, max_new_tokens=30, p=0.9, temperature=0.5, rep_penalty=1.2)
        decoded_output = self.tokenizer.decode(generated[0].tolist())

if __name__ == '__main__':
    unittest.main()