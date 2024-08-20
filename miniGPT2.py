import torch
import torch.nn as nn
from torch.nn import functional as F

import math
from dataclasses import dataclass
import tiktoken
import time
import inspect

import random

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

import os
import numpy as np
from tqdm import tqdm

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# B T C = Batch size, Sequence length, Embedding dimension

@dataclass
class GPTConfig:
    block_size: int = 1024 # sequence length
    vocab_size: int = 50304 # number of tokens. Changed from 50257 so it is power of 2
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

        self.c_proj.RES_CONN_FLAG = 1 # flag to calculate to calculate residual with normalization -> sqrt n is prop cuz so we dont make it linear idk

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
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # we can transpose here as well but k.size(-1) changes
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # attention_scores = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # attention_scores = attention_scores.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        # attention_scores = F.softmax(attention_scores, dim=-1)
        # y = attention_scores @ v

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # flash attention that I should research more about 2x times less
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

        self.c_proj.RES_CONN_FLAG = 1

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

        # shares the weight such that it helps with the embeddings
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_params)

    def _init_params(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "RES_CONN_FLAG"):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, y=None):
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

        loss = None
        
        """
        We return B, T, vocab size as logits. Vocab size basically includes the probabilities of each token being predicted after a sequence
        of T tokens. We use the probabilities for cross entropy loss calculations which helps to get the likelyhood of one token being 
        predicted and is a much better use. 
        """
        if y is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1)) # (B*T, vocab_size) Vs (B*T)

        return logits, loss

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
    
    def configure_optimizers(self, weight_decay, learning_rate, device):
        param_dict = {pn: p for pn, p in self.named_parameters()} # names_parameters returns the name and the parameter
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nondecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nondecay_params, "weight_decay": 0.0}
        ]

        num_decay_params = sum(p.numel() for p in decay_params)
        num_nondecay_params = sum(p.numel() for p in nondecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nondecay_params)}, with {num_nondecay_params:,} parameters")
        
        fused_availability = "fused" in inspect.signature(torch.optim.AdamW).parameters # introspection is the ability to examine the type or properties of an object at runtime
        use_fused = fused_availability and "cuda" in device # "fused" means that multiple operations are combined into a single kernel
        print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer


def load_tokens(filename):
    npt = np.load(filename)
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

class FineWebDataset(Dataset):
    def __init__(self, T, split):
        self.T = T

        assert split in {"train", "val"}

        data_root = "edu_fineweb10B"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards

        assert len(shards) > 0, f"no shards found for split {split}"

        print(f"Found {len(shards)} shards for split {split}")

        self.reset()
    
    def reset(self):
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])

    def __len__(self):
        return len(self.tokens) // self.T

    def __getitem__(self, idx): # batch index
        T = self.T
        start = idx * T
        end = start + T + 1
        buf = self.tokens[start:end]
        x = buf[:-1]
        y = buf[1:]
        
        if end + T + 1 > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            
        return x, y

def produce_sentence(max_length, itr):
    print("Generated Text ")

    tokens = tokenizer.encode("I hope I marry this boy ")
    tokens = torch.tensor(tokens, dtype=torch.long)
    tokens = tokens.unsqueeze(0).repeat(itr, 1)
    x = tokens.to(device)

    while x.size(1) < max_length: 
        with torch.no_grad():
            logits, _ = model(x)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            top_probs, top_indices = torch.topk(probs, 50, dim=1)
            selected_token = torch.multinomial(top_probs, 1) # selects a random token from each batch
            selected_index = torch.gather(top_indices, -1, selected_token)
            x = torch.cat((x, selected_index), dim=1)

    for i in range(itr):
        tokens = x[i, :max_length].tolist()
        decoded = tokenizer.decode(tokens)
        print(">", decoded, "\n")

"""
Warm up for the lr to increase and then
cosine decay for the lr to decrease and converge to min lr
"""
def get_lr(step):
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step > max_steps:
        return min_lr
    
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

def shuffle_data(data, seq_length, test_perc=0.2, split_num=2):
    new_data = list(data)
    split = test_perc / split_num
    new_seq_length = seq_length + 1
    batch_size = int(len(data) // new_seq_length * split)
    test_data = []

    if batch_size * split_num * new_seq_length > len(data):
        raise ValueError("Not enough data to split into the requested test size")

    for _ in range(split_num):
        random_idx = random.randint(0, len(new_data) - batch_size * new_seq_length)

        test_sample = new_data[random_idx:random_idx + batch_size * new_seq_length]
        test_data.extend(test_sample)

        new_data = new_data[:random_idx] + new_data[random_idx + batch_size * new_seq_length:]

    return new_data, test_data


# def test():
#     tot_loss = 0
#     for _ in range(test_epoch):
#         with torch.no_grad():
#             x_test, y_test = test_dataloader.next_data()
#             with ctx:
#                 logits_test, loss_test = model(x_test, y_test) # loss at start should be around -ln(1/50257) cuz of cross entrophy function
#             tot_loss += loss_test
#             torch.cuda.synchronize()
#     return tot_loss / test_epoch

if __name__ == '__main__':
    B, T = 4, GPTConfig.block_size
    total_batch_size = 524_288
    assert total_batch_size % (B * T) == 0, "make sure total_batch_size is divisible by B * T"
    grad_accum_steps = total_batch_size // (B * T) # 64
    print(f"Calculated gradient accumulation steps: {grad_accum_steps}")

    device = ("cuda" if torch.cuda.is_available() else "cpu") # mps stands for apple silicon that has a gpu    
    ctx = torch.autocast('cuda', dtype=torch.float16)
    scaler = torch.GradScaler(enabled=True)
    torch.set_float32_matmul_precision("high") # TF32
    model = GPT.from_pretrained("gpt2")
    model.to(device)
    # model = torch.compile(model)
    tokenizer = tiktoken.get_encoding("gpt2")

    train_dataset = FineWebDataset(T, "train")
    val_dataset = FineWebDataset(T, "val")

    train_dataloader = DataLoader(train_dataset, batch_size=B, shuffle=True, num_workers=4, pin_memory=True, drop_last=True, persistent_workers=True)
    val_dataloader = DataLoader(val_dataset, batch_size=B, shuffle=False, num_workers=2, pin_memory=True, drop_last=True)

    # train_iterator = iter(train_dataloader)
    # val_iterator = iter(val_dataloader)

    max_steps = 1000
    warmup_steps = 50
    max_lr = 6e-4
    min_lr = max_lr * 0.1

    lrs = []

    optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=max_lr, device=device)

    train_losses = []
    val_losses = []
    time_values = []

    time0 = time.time()
    step = 0
    loss_val = -1

    try:
        while step < max_steps:
            if step % 200 == 0:
                print("Evaluating")
                model.eval()
                eval_count = 0
                val_loss_accum = 0
                val_max_steps = 10
                done = False
                with torch.no_grad():
                    while done == False:
                        for x_val,y_val in val_dataloader:
                            eval_count += 1
                            x_val, y_val = x_val.to(device, non_blocking=True), y_val.to(device,non_blocking=True)
                            with ctx:
                                _, val_loss = model(x_val, y_val)
                            val_loss = val_loss / val_max_steps
                            val_loss_accum += val_loss
                            if eval_count == val_max_steps:
                                done = True
                                break

                    print(f"val loss: {val_loss_accum.item():.05f}")
                    loss_val = val_loss_accum

                model.train()
            start_time = time.time()

            loss_accum = 0.0
            acum_count = 0
            
            for x,y in train_dataloader: # 100ms per 4 mini batch
                acum_count += 1
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                with ctx: # 15ms
                    logits, loss = model(x, y) # loss at start should be around -ln(1/50257) cuz of cross entrophy function
                loss = loss / grad_accum_steps
                loss_accum += loss.detach()
                scaler.scale(loss).backward() # 90ms
                if acum_count == grad_accum_steps:
                    break

            # 10ms
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            val_losses.append(loss_val.item())
            train_losses.append(loss_accum.item())

            lr = get_lr(step)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
            lrs.append(lr)
            
            # torch.cuda.synchronize()
            end_time = time.time()
            time_diff = (end_time - start_time) * 1000
            time_values.append(end_time)
            print(f"-> {step} | loss: {loss_accum.item():.05f} | norm: {norm:.4f}| lr: {lr:.4e} | time (ms): {time_diff:.05f} | tok/sec: {((B * T * grad_accum_steps) / time_diff * 1000):.05f} |")
            step += 1
    except KeyboardInterrupt:
        pass

    timef = time.time()
    print(f"Time Elapsed: {timef - time0}s")
    produce_sentence(20, 2)

    plt.figure(figsize=(10, 5))
    plt.plot(time_values, train_losses, label="Train Loss")
    plt.plot(time_values, val_losses, label="Val Loss", linestyle='--')
    plt.xlabel("Time (s)")
    plt.ylabel("Loss")
    plt.title("Time vs Train Loss and Val Loss")
    plt.legend()
    plt.show()

    word_embeddings = model.transformer.wte.weight.detach().cpu().numpy()  # (vocab_size, n_embd)
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(word_embeddings)

    plt.figure(figsize=(10, 10))
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], s=1)

    for i in range(100): # first 100 tokens
        plt.annotate(str(i), (reduced_embeddings[i, 0], reduced_embeddings[i, 1]))

    plt.title("2D Visualization of Word Embeddings")
    plt.show()

    plt.plot(lrs)
    plt.show()

