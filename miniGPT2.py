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

# B T C = Batch size, Sequence length, Embedding dimension

@dataclass
class GPTConfig:
    block_size: int = 512 # sequence length
    vocab_size: int = 50304 # number of tokens. Changed from 50257 so it is power of 2
    n_layer: int = 8 # number of layers
    n_head: int = 8 # number of heads
    n_embd: int = 128 # embedding dimension

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
            {"params": decay_params, "weight _decay": weight_decay},
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
            
class DataLoader:
    def __init__(self, B, T, data, epoch=True):
        self.B = B
        self.T = T
        self.start = 0
        self.epoch = 1
        self.det_epoch = epoch

        self.encoded_text = torch.tensor(data)
        self.encoded_text = self.encoded_text.to(device)

        if epoch:
            print(f"TRAIN: epoch {self.epoch}, {math.floor(self.encoded_text.size(-1) / (B * T))} loops for {self.B} batches, {len(self.encoded_text)} tokens, {B * T} tokens per loop")
        else:
            print(f"TEST: {math.floor(self.encoded_text.size(-1) / (B * T))} loops for {self.B} batches, {len(self.encoded_text)} tokens, {B * T} tokens per loop")

    def next_data(self):
        x = self.encoded_text[self.start : (self.start + self.B*self.T)].view(self.B, self.T) # 32 * 4 = 128
        y = self.encoded_text[(self.start + 1) : (self.start + self.B*self.T + 1)].view(self.B, self.T)

        self.start += self.B * self.T
        if self.start + self.B*self.T >= self.encoded_text.size(-1):
            self.start = 0

        if self.start + self.B * self.T >= self.encoded_text.size(-1) * grad_accum_steps:
            self.epoch += 1

            if self.det_epoch:
                print(f"Epoch {self.epoch}")
            

        return x, y

def produce_sentence(max_length, itr):
    print("Generated Text ")

    tokens = tokenizer.encode("ANTONIO:\n")
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


def test():
    tot_loss = 0
    for _ in range(test_epoch):
        with torch.no_grad():
            x_test, y_test = test_dataloader.next_data()
            with ctx:
                logits_test, loss_test = model(x_test, y_test) # loss at start should be around -ln(1/50257) cuz of cross entrophy function
            tot_loss += loss_test
            torch.cuda.synchronize()
    return tot_loss / test_epoch

B, T = 16, GPTConfig.block_size
total_batch_size = 131_072
assert total_batch_size % (B * T) == 0, "make sure total_batch_size is divisible by B * T"
grad_accum_steps = total_batch_size // (B * T) # 64
print(f"Calculated gradient accumulation steps: {grad_accum_steps}")

device = ("cuda:0" if torch.cuda.is_available() else "cpu") # mps stands for apple silicon that has a gpu    
ctx = torch.autocast('cuda', dtype=torch.float16)
scaler = torch.GradScaler(enabled=True)
torch.set_float32_matmul_precision("high") # TF32
model = GPT(GPTConfig())
model.eval()
model.to(device)

tokenizer = tiktoken.get_encoding("gpt2")

with open("miniShakespare.txt", "r") as f:
    text = f.read()
tokenized_text = tokenizer.encode(text)
train_data, test_data = shuffle_data(tokenized_text, T, test_perc=0.1, split_num=65)

train_dataloader = DataLoader(B, T, train_data)
test_dataloader = DataLoader(B, T, test_data, False)

train_steps = int(input("Loops per train epoch: "))
epoch_steps = train_steps // (B * T)
warmup_steps = epoch_steps
max_steps = epoch_steps * 28
max_lr = 3e-4
min_lr = max_lr * 0.1

test_steps = int(input("Loops per test epoch: "))
test_epoch = test_steps // (B * T)

lrs = []

optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=3e-4, device=device)

train_losses = []
test_losses = []
time_values = []

time0 = time.time()

try:
    for step in range(epoch_steps * 30):
        start_time = time.time()
        optimizer.zero_grad()

        loss_accum = 0.0
        for micro_step in range(grad_accum_steps):
            x, y = train_dataloader.next_data()
            with ctx:
                logits, loss = model(x, y) # loss at start should be around -ln(1/50257) cuz of cross entrophy function
            loss = loss / grad_accum_steps
            loss_accum += loss.detach()
            scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        train_losses.append(loss_accum.item())

        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        lrs.append(lr)
        
        torch.cuda.synchronize()
        end_time = time.time()
        time_diff = (end_time - start_time) * 1000
        time_values.append(end_time)
        
        loss_test = test()
        test_losses.append(loss_test.item())
        print(f"-> {step} | loss: {loss_accum.item():.05f} | norm: {norm:.4f}| lr: {lr:.4e} | time (ms): {time_diff:.05f} | tok/sec: {((B * T * grad_accum_steps) / time_diff * 1000):.05f} | test loss: {loss_test.item():.05f}")

except KeyboardInterrupt:
    test()

timef = time.time()
print(f"Time Elapsed: {timef - time0}s")
produce_sentence(20, 2)

plt.figure(figsize=(10, 5))
plt.plot(time_values, train_losses, label="Train Loss")
plt.plot(time_values, test_losses, label="Test Loss", linestyle='--')
plt.xlabel("Time (s)")
plt.ylabel("Loss")
plt.title("Time vs Train Loss and Test Loss")
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
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.show()

plt.plot(lrs)
plt.show()

