import torch
import torch.nn.functional as F
from einops import rearrange
from jaxtyping import Float, Int
from torch import nn

from .config import Config


class Rotary(nn.Module):
    cos: torch.Tensor
    sin: torch.Tensor

    def __init__(self, head_dim: int, seq_len: int):
        super().__init__()
        exponents = torch.arange(0, head_dim, 2).float() / head_dim  # (head_dim/2,)
        inverse_freq = 1.0 / 10000**exponents
        angles = torch.outer(torch.arange(seq_len).float(), inverse_freq)  # (seq, head_dim/2)
        angles = torch.cat([angles, angles], dim=-1)  # (seq, head_dim)
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    def forward(
        self, x: Float[torch.Tensor, "batch heads seq head_dim"]
    ) -> Float[torch.Tensor, "batch heads seq head_dim"]:
        """Rotate by position."""
        length = x.shape[2]
        first, second = x.chunk(2, dim=-1)  # each (batch, heads, seq, head_dim/2)
        rotated_half = torch.cat([-second, first], dim=-1)  # (batch, heads, seq, head_dim)
        rotated = x * self.cos[:length] + rotated_half * self.sin[:length]
        return rotated.type_as(x)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.heads = config.heads
        self.qkv = nn.Linear(config.width, 3 * config.width, bias=False)
        self.proj = nn.Linear(config.width, config.width, bias=False)
        self.rotary = Rotary(config.width // config.heads, config.seq_len)

    def forward(
        self, x: Float[torch.Tensor, "batch seq width"]
    ) -> Float[torch.Tensor, "batch seq width"]:
        qkv = rearrange(
            self.qkv(x),
            "batch seq (three heads head_dim) -> three batch heads seq head_dim",
            three=3,
            heads=self.heads,
        )
        query, key, value = qkv  # each (batch, heads, seq, head_dim)
        query, key = self.rotary(query), self.rotary(key)
        out = F.scaled_dot_product_attention(query, key, value, is_causal=True)
        return self.proj(rearrange(out, "batch heads seq head_dim -> batch seq (heads head_dim)"))


class MLP(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.up = nn.Linear(width, 4 * width, bias=False)
        self.down = nn.Linear(4 * width, width, bias=False)

    def forward(
        self, x: Float[torch.Tensor, "batch seq width"]
    ) -> Float[torch.Tensor, "batch seq width"]:
        return self.down(F.gelu(self.up(x)))


class Block(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.width)
        self.attention = CausalSelfAttention(config)
        self.mlp_norm = nn.LayerNorm(config.width)
        self.mlp = MLP(config.width)

    def forward(
        self, x: Float[torch.Tensor, "batch seq width"]
    ) -> Float[torch.Tensor, "batch seq width"]:
        x = x + self.attention(self.attention_norm(x))
        return x + self.mlp(self.mlp_norm(x))


class GPT(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.seq_len = config.seq_len
        self.embedding = nn.Embedding(config.vocab_size, config.width)
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.depth))
        self.norm = nn.LayerNorm(config.width)
        self.head = nn.Linear(config.width, config.vocab_size, bias=False)
        for module in self.modules():
            if isinstance(module, nn.Linear | nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

    def forward(
        self, token_ids: Int[torch.Tensor, "batch seq"]
    ) -> Float[torch.Tensor, "batch seq vocab"]:
        x = self.embedding(token_ids)
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x))

    def parameter_counts(self) -> dict[str, int]:
        total = sum(parameter.numel() for parameter in self.parameters())
        embedding = self.embedding.weight.numel() + self.head.weight.numel()
        return {"embedding": embedding, "non_embedding": total - embedding}

    @torch.no_grad()
    def generate(
        self, prompt_ids: Int[torch.Tensor, "1 prompt"], new_tokens: int, generator: torch.Generator
    ) -> Int[torch.Tensor, "1 total"]:
        """Sample new_tokens continuing a (1, T) prompt; no KV cache."""
        token_ids = prompt_ids
        for _ in range(new_tokens):
            # forward, not __call__: skips the compiled wrapper so varying lengths don't recompile
            logits = self.forward(token_ids[:, -self.seq_len :])[:, -1]
            probabilities = F.softmax(logits.float(), dim=-1)
            next_id = torch.multinomial(probabilities, 1, generator=generator)
            token_ids = torch.cat([token_ids, next_id], dim=1)  # (1, seq + 1)
        return token_ids
