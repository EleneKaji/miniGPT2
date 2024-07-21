import torch
import torch.nn as nn
from torch.nn import functional

# B T C = Batch size, Sequence length, Embedding dimension

class GPTConfig:
    block_size: int = 256
    vocab_size: int = 65
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 6

"""
One of the changes from transformer to GPT is that 
we want the residual to not be inside the layer normalization so that
during the backpropagation we can pass the gradiet to the residual
as well as the output of that layer equally. We should put it inside
the residual block.
"""
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()

        """
        Layer normalization and batch normalization are used to try and limit
        having really high gradient. Both of the are done by calculate the mean
        and dividing by the standard deviation. In layer normalization we do that
        to each column (input) to the next layer, while in batch normalization 
        we do it to the row (batch).
        """

        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config) # weighted sum, communicate
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config) # think individually
    
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()

        
        """
        With c_fc and c_proj we are trying to first add the dimentionality
        and complexity to the linear layer, then change it back to the old
        dimensions. With this the projection returns to the original (we 
        campute more filters in first, then return back)
        """
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.c_proj = nn.Linear(4 * config.embed, config.n_embd)

        """
        GELU is Gaussian error linear unit such that it avoid the issue with ReLU 
        which is it has no derivative at a certain point.
        Approximation is used for some reason that is not relevant and is not used
        anymore. However it is accurate to gelu and requires less time complexity.
        Erf in gelu is for error function and is basically changed with tanh
        """
        self.gelu = nn.GELU(approximate="tanh")

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
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
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd), 
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd)
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)