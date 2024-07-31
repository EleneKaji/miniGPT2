import torch
import torch.nn as nn
from torch.nn import functional as F

import math
from dataclasses import dataclass

# B T C = Batch size, Sequence length, Embedding dimension

@dataclass
class GPTConfig:
    block_size: int = 1024 # sequence length
    vocab_size: int = 50257 # number of tokens
    n_layer: int = 12 # number of layers
    n_head: int = 12 # number of heads
    n_embd: int = 768 # embedding dimension

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head
        self.n_embd = config.n_embd

        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()
        qvk = self.c_attn(x)
        q, k, v = qvk.split(self.n_embd, dim=2)

        """
        Number of tokens stays the same, only embeddings change by the size of number of heads.
        Taking only specific number of embeddings are like applying filters to certain embedding
        dimensions, so we can get various many filters (like kernels in CNN)
        """

        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, T, number of heads, embedding for each head) -> (B, number of heads, T, embedding for each head)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        attention_scores = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        attention_scores = attention_scores.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        attention_scores = F.softmax(attention_scores, dim=-1)

        y = attention_scores @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        
        return y

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
        self.attn = MultiHeadSelfAttention(config) # weighted sum, communicate
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
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

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
            row stands for the entry of the token in the mebedding table and the vector for that token. token number corresponds to index number of the entry
            column is how many embedding numbers for each token
        Block:
            block is the inside of the transformer 
        """
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd), # (50257, 768)
            wpe = nn.Embedding(config.block_size, config.n_embd), # (1024, 768)
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]), 
            ln_f = nn.LayerNorm(config.n_embd)
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, x):
        """
        transformer.wte.weight torch.Size([50257, 768]) = # of tokens, embedding dimensions (vector representing tokens)
        transformer.wpe.weight torch.Size([1024, 768]) = sequence length, embedding dimensions 
        """
        # x is Batch Size and Sequence Length (in tokens)
        B, T = x.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"

        positions = torch.arange(0, T, dtype=torch.long, device=x.device)
        position_embd = self.transformer.wpe(positions) # (sequence length, embedding dimensions)
        token_embd = self.transformer.wte(x) # (batch size, sequence length, embedding dimensions)
        x = token_embd + position_embd
        
        for block in self.transformer.h:
            x = block(x)
        
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        return logits

    @classmethod
    def from_pretrained(cls, model_type):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("Loading weights from pretrained GPT-2")

        config_args = {
            "gpt2": dict(n_layer=12, n_head=12, n_embd=768),
            "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),
            "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280),
            "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600)
        }[model_type]
        config_args["vocab_size"] = 50257
        config_args["block_size"] = 1024

        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith(".attn.bias")]

        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith(".attn.masked_bias")]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith(".attn.bias")]
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

device = ("cuda:0" if torch.cuda.is_available() else "cpu")    

model = GPT.from_pretrained("gpt2")
model.eval()
model.to(device)

import tiktoken
encoder = tiktoken.get_encoding("gpt2")
tokens = encoder.encode("Hello, I'm a language model,")
tokens = torch.tensor(tokens, dtype=torch.long)
tokens = tokens.unsqueeze(0).repeat(5, 1)
x = tokens.to(device)

max_length = 30
torch.manual_seed(42)
torch.cuda.manual_seed(42)
while x.size(1) < max_length: 
    with torch.no_grad():
        logits = model(x)
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        top_probs, top_indices = torch.topk(probs, 50, dim=1)
        selected_token = torch.multinomial(top_probs, 1) # selects a random token from each batch
        selected_index = torch.gather(top_indices, -1, selected_token)
        x = torch.cat((x, selected_index), dim=1)

for i in range(5):
    tokens = x[i, :max_length].tolist()
    decoded = encoder.decode(tokens)
    print(">", decoded)