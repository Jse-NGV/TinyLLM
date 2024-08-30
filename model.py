# Author: Jserw
# Start: 2024/8/26
# Finish: 2024/8/31

import math
import struct
import inspect
from dataclasses import dataclass
from typing import Any, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import random
import os
import glob

DATA_CACHE_DIR = 'data'

@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 6
    n_heads: int = 6
    n_group: Optional[int] = 3
    vocab_size: int = 4096
    hidden_dim: Optional[int] = None
    multiple_of: int = 256 # MLP层隐层维度的指定计算参数
    norm_eps: float = 1e-5
    max_seq_len: int = 2048
    dropout: float = 0.0


class RMSNorm(nn.Module):
    def __init__(self, dim, eps):
        super(RMSNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(dim)) # (d)
        self.eps = eps

    def norm(self, x): # (b, h, n, hd) torch.rsqrt() (b, h, n, 1)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) # 此处用到了python的广播机制

    def forward(self, x):
        return self.weight * self.norm(x.float()).type_as(x)

# 大概能理解其中的意思了
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.):
    assert dim % 2 == 0
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim//2)].float() / dim)) # 1/ 10000^(2i/d) i from 0 to d/2-1
    t = torch.arange(0, end, 1) # 蕴含了位置信息, token的位置信息
    freqs = torch.outer(t, freqs).float() # (T, d/2)
    freqs_cos = torch.cos(freqs)
    freqs_sin = torch.sin(freqs)
    return freqs_cos, freqs_sin

def reshape_for_brodcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    # freqs (T, D) --> (1, T, 1, D)
    # x -- > (B, T, H, D)
    B, T, H, D = x.shape
    new_shape = [1, T, 1, D]
    return  freqs_cis.reshape(*new_shape)


def apply_rotary_emb(
        xq: torch.Tensor, # (B, T, H, D)
        xk: torch.Tensor,
        freqs_cos: torch.Tensor, # (B, T, H, D)
        freqs_sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    B, T, H, D = xq.shape
    # 重塑xq 和 xk，使其与复数表示相匹配
    xq_r, xq_i = xq.float().reshape(*(xq.shape[:-1]), -1, 2).unbind(-1) # (B, T, H, D) --> (B, T, H, D/2) * 2
    xk_r, xk_i = xk.float().reshape(*(xk.shape[:-1]), -1, 2).unbind(-1) # (B, T, H, D) --> (B, T, H, D/2) * 2

    freqs_cos = reshape_for_brodcast(freqs_cos, xq_r) # (1, T, 1, D/2)
    freqs_sin = reshape_for_brodcast(freqs_sin, xq_r)

    xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin # (B, T, H, D/2)
    xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos
    xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin
    xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos

    xq_out = torch.stack([xq_out_r, xq_out_i], dim=-1).reshape(xq.shape)
    xk_out = torch.stack([xk_out_r, xk_out_i], dim=-1).reshape(xk.shape)

    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:

    batch_size, seq_len, kv_head, head_dim = x.shape
    if n_rep == 1:
        return x
    x = x[:,:,:,None,:].expand(batch_size, seq_len, kv_head, n_rep, head_dim) # head_dim*kv_head --> head_dim*n_head
    return x.reshape(batch_size, seq_len, kv_head*n_rep, head_dim)

class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int, dropout: float):
        super(FeedForward, self).__init__()
        if hidden_dim is None:
            hidden_dim = 4*dim
            hidden_dim = int(2*hidden_dim/3)
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1)//multiple_of)
        self.w1 = nn.Linear(dim, hidden_dim)
        self.w2 = nn.Linear(hidden_dim, dim)
        self.w3 = nn.Linear(dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        y1 = F.silu(self.w1(x))
        y2 = self.w3(x)
        x = self.dropout(self.w2(y1 * y2))
        return x


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super(Attention, self).__init__()
        self.n_group = args.n_group
        self.n_head = args.n_heads
        self.head_dim = args.dim // self.n_head
        self.kv_head = self.n_head // self.n_group
        self.wq = nn.Linear(args.dim, self.head_dim * self.n_head, bias=False)
        self.wk = nn.Linear(args.dim, self.head_dim * self.kv_head, bias=False)
        self.wv = nn.Linear(args.dim, self.head_dim * self.kv_head, bias=False)
        self.wo = nn.Linear(self.head_dim * self.n_head, args.dim, bias=False)
        mask = torch.full((1, 1, args.max_seq_len , args.max_seq_len), float('-inf'))
        mask = torch.triu(mask, diagonal=1) # 下三角和对角线为0
        self.att_dropout = nn.Dropout(args.dropout)
        self.redis_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout
        self.register_buffer('mask', mask)

    def forward(
            self,
            x: torch.Tensor, # (b, t, d)
            freqs_cos: torch.Tensor, # (t, d/2)
            freqs_sin: torch.Tensor
    ):
        B, T, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x) # q (b, t, head_dim*n_head)  kv (b, t, head_dim*kv_head)
        xq = xq.reshape(B, T, self.n_head, self.head_dim)
        xk = xk.reshape(B, T, self.kv_head, self.head_dim)
        xv = xv.reshape(B, T, self.kv_head, self.head_dim)
        # RoPE
        xq, xk = apply_rotary_emb(xq, xk, freqs_cos, freqs_sin)

        # GQA expand keys and values
        xk = repeat_kv(xk, self.n_group) # (b, t, kv_head, head_dim) --> (b, t, n_head, head_dim) 其中 n_head = n_rep * kv_head
        xv = repeat_kv(xv, self.n_group)

        xq = xq.transpose(1, 2) # (b, n_head, t, head_dim)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        scale = xk.size(-1) ** -0.5
        att = (xq @ xk.transpose(-1, -2)) * scale # (b, n_head, t, t)
        att = att + self.mask[:, :, :T, :T]
        att = F.softmax(att, dim=-1)
        att = self.att_dropout(att)
        output = att @ xv # (b, n_head, t, head_dim)
        output = (output.transpose(1, 2)).reshape(B, T, self.n_head * self.head_dim)
        output = self.wo(output) # (b, t, d)
        output = self.redis_dropout(output)
        return output


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs, layerid):
        super(TransformerBlock, self).__init__()
        self.att = Attention(args)
        self.ffn = FeedForward(
            dim=args.dim,
            hidden_dim=args.hidden_dim,
            multiple_of=args.multiple_of,
            dropout=args.dropout
        )
        self.layerid = layerid
        self.att_norm = RMSNorm(args.dim, args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, args.norm_eps)

    def forward(self, x, freqs_cos, freqs_sin):
        # x (b, t, d)
        h = self.att(self.att_norm(x), freqs_cos, freqs_sin) + x
        h = self.ffn(self.ffn_norm(h)) + h
        return h

class Llama(nn.Module):
    last_loss: Optional[torch.Tensor]

    def __init__(self, params: ModelArgs):
        super(Llama, self).__init__()
        self.params = params
        self.dim = params.dim
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers
        self.token_embeddings = nn.Embedding(self.vocab_size, self.dim)
        self.dropout = nn.Dropout(params.dropout)
        self.layers = torch.nn.ModuleList()
        for layer_id in range(self.n_layers):
            self.layers.append(TransformerBlock(params, layer_id))
        self.norm = RMSNorm(self.dim, params.norm_eps)
        self.output = nn.Linear(self.dim, self.vocab_size, bias=False)

        # share the unembedding parameters with the embedding parameters
        self.token_embeddings.weight = self.output.weight

        freqs_cos, freqs_sin = precompute_freqs_cis(self.dim // self.params.n_head, self.params.max_seq_len)
        self.register_buffer('freqs_cos',freqs_cos)
        self.register_buffer('freqs_sin',freqs_sin)

        # init weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('w3.weight') or pn.endswith('wo.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2*params.n_layers))

        self.last_loss = None

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens: torch.Tensor, targets: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len = tokens.shape # (b, t)
        h = self.token_embeddings(tokens) # (b, t, d)
        freqs_cos = self.freqs_cos[:seq_len,:] # (t, d/2)
        freqs_sin = self.freqs_sin[:seq_len,:]

        for layer in self.layers:
            h = layer(h, freqs_cos, freqs_sin)
        h = self.norm(h)

        if targets is not None:
            # targets (b, t)
            logits = self.output(h) # (b, t, vocab_size)
            # input (N, C) target(N) logits (b,t,v)-->(b*t,v) target (b,t)-->(b*t)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1)
            self.last_loss = loss
        else:
            logits = self.output(h[:,[-1],:]) # (b, 1, v)
            self.last_loss = None
        return logits

    @torch.inference_mode()
    def generate(self, idx, max_new_tokens, temperature=1.0):
        # idx (b, t)
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1)<=self.params.max_seq_len else idx[:,-self.params.max_seq_len:]
            logits = self(idx_cond) # targets=None mode, (b, 1, v)
            logits = logits[:, -1, :] # (b, v) 取生成的最后一个token
            logits = logits / temperature
            probs = F.softmax(logits, dim=-1) # (b, v)
            idx_next = torch.multinomial(probs, num_samples=1) # (b, 1)
            idx = torch.cat((idx,idx_next), dim=1) # (b, t+1)

        return idx

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn:p for pn,p in param_dict.items() if p.requires_grad}
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't
        decay_params = [p for pn,p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for pn,p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay':weight_decay},
            {'params':nodecay_params, 'weight_decay':0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using AdamW: {use_fused}")

        return optimizer


class PretokDataset(torch.utils.data.IterableDataset):
    """Loads pretokenized examples from disk and yields them as PyTorch tensors."""
    def __init__(self, split, max_seq_len, vocab_size, vocab_source):
        super(PretokDataset, self).__init__()
        self.split = split
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.vocab_source = vocab_source
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        seed = worker_id + 42
        rng = random.Random(seed)
        print(f"Created a PretokDataset with rng seed {seed}")
        self.vocab_source = 'custom'
        # the .bin files are in tok{N} directory
        # bin_dir = os.path.join(DATA_CACHE_DIR, f"tok{self.vocab_size}")
        bin_dir = os.path.join(DATA_CACHE_DIR, 'demo')
        shared_filenames = sorted(glob.glob(os.path.join(bin_dir, "*.bin")))
        # train/test split. let's use only shard 0 for test split, rest train
        shared_filenames = shared_filenames[1:] if self.split=='train' else shared_filenames[:1]
        assert len(shared_filenames)>0, f"No bin files found in {bin_dir}"
        while True:
            rng.shuffle(shared_filenames)
            for shard in shared_filenames:
                # open the dataset for reading but keep it on disk with memmap
                m = np.memmap(shard, dtype=np.uint16, mode='r')
                num_batches = len(m) // self.max_seq_len
                num_batches -= 1  # drop the last partial batch
                assert num_batches > 0, "this shard is way too small? investigate."
                ixs = list(range(num_batches))
                rng.shuffle(ixs)
                for ix in ixs:
                    start = ix * self.max_seq_len
                    end = start + self.max_seq_len + 1
                    # calling .astype will copy the data into a new numpy array, now in RAM
                    chunk = torch.from_numpy((m[start:end]).astype(np.int64))
                    x = chunk[:-1] #(1234)
                    y = chunk[1:]  #(2345)
                    yield x, y


class Task:
    @staticmethod
    def iter_batches(batch_size, device, num_workers=0, **dataset_kwargs):
        ds = PretokDataset(**dataset_kwargs)
        dl = torch.utils.data.DataLoader(
            ds, batch_size=batch_size, pin_memory=True, num_workers=num_workers
        )
        for x,y in dl:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            yield x,y

if __name__ == '__main__':
    conf = ModelArgs()
    model = Llama(conf)
    inputs = torch.randint(low=10, high=50, size=(10,2))
    output = model(inputs)
    print(output.shape)

