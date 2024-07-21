import torch
import torch.nn as nn
from torch.nn import functional

class GPTConfig:
    block_size: int = 256
    vocab_size: int = 65
    n_layer: int = 6
    n_head: int = 6
    n_embed: int = 6

"""
One of the changes from transformer to GPT is that 
we want the residual to not be inside the layer normalization so that
during the backpropagation we can pass the gradiet to the residual
as well as the output of that layer equally.
"""
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.ln_1 = nn.LayerNorm(config.n_embed)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embed)
        self.mlp = MLP(config)
    
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        """
        Embedding(row, col):  # num_embeddings, embedding_dim
            row stands for the entry of the token in the mebedding table and the vector for that token
            column is how many embedding numbers for each token
        Block:
            block is the inside of the transformer 
        """
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embed),
            wpe = nn.Embedding(config.block_size, config.n_embed), 
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embed)
        ))
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)