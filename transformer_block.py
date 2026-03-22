import torch
import torch.nn as nn
import math
from dataclasses import dataclass
from einops import rearrange, repeat

@dataclass
class TransformerBlockConfig:
    d_model: int = 256
    n_heads: int = 8
    d_hidden: int = 2048
    dropout_rate: float = 0.1

def scaled_dot_attention(Q, K, V, mask=None):
    # Attention = Softmax((Q @ K.T) / sqrt(d_k)) @ V
    d_k = Q.shape[-1]

    scores = Q @ K.transpose(-1, -2) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask, float('-inf'))

    similarity = torch.softmax(scores, dim=-1)
    attention = similarity @ V

    return attention


class MultiHeadAttentionForStudy(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.d_model % config.n_heads == 0

        self.d_model: int = config.d_model
        self.n_heads: int = config.n_heads
        self.d_k: int = config.d_model // config.n_heads

        self.W_Qs = nn.ModuleList([
            nn.Linear(self.d_model, self.d_k, bias=False) for _ in range(self.n_heads)
        ])

        self.W_Ks = nn.ModuleList([
            nn.Linear(self.d_model, self.d_k, bias=False) for _ in range(self.n_heads)
        ])

        self.W_Vs = nn.ModuleList([
            nn.Linear(self.d_model, self.d_k, bias=False) for _ in range(self.n_heads)
        ])

        self.W_O = nn.Linear(self.d_model, self.d_model, bias=False)

    def forward(self, Q, K=None, V=None, mask=None):
        if K is None and V is None:
            K = V = Q

        return self.W_O(torch.cat([
            scaled_dot_attention(W_Q(Q), W_K(K), W_V(V), mask=mask) for W_Q, W_K, W_V in zip(self.W_Qs, self.W_Ks, self.W_Vs)
        ], dim=-1))


class MutiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.d_model % config.n_heads == 0
        self.d_model: int = config.d_model
        self.n_heads: int = config.n_heads
        self.d_k: int = config.d_model // config.n_heads

        self.W_Q = nn.Linear(self.d_model, self.d_model, bias=False)
        self.W_K = nn.Linear(self.d_model, self.d_model, bias=False)
        self.W_V = nn.Linear(self.d_model, self.d_model, bias=False)
        self.W_O = nn.Linear(self.d_model, self.d_model, bias=False)

    def forward(self, Q, K=None, V=None, mask=None):
        if K is None and V is None:
            K = V = Q

        # b -> batch_size, s -> seq_len, h -> n_heads, d -> d_k, (h d) -> d_model
        Q = rearrange(self.W_Q(Q), 'b s (h d) -> b h s d', h=self.n_heads)
        K = rearrange(self.W_K(K), 'b s (h d) -> b h s d', h=self.n_heads)
        V = rearrange(self.W_V(V), 'b s (h d) -> b h s d', h=self.n_heads)

        B = Q.shape[0]

        if mask is not None:
            if len(mask.shape) == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)
            elif len(mask.shape) == 3:
                mask = mask.unsqueeze(1)

        attention = scaled_dot_attention(Q, K, V, mask=mask)
        attention = rearrange(attention, 'b h s d -> b s (h d)')

        return self.W_O(attention)

class EncoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attention = MultiHeadAttention(config)

        self.feed_forward = nn.Sequential(
            nn.Linear(config.d_model, config.d_hidden),
            nn.GELU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.d_hidden, config.d_model),
        )

        self.layer_norm1 = nn.LayerNorm(config.d_model)
        self.layer_norm2 = nn.LayerNorm(config.d_model)

    def forward(self, X, mask=None):
        X = X + self.self_attention(self.layer_norm1(X), mask=mask)

        X = X + self.feed_forward(self.layer_norm2(X))

        return X


class DecoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.masked_self_attention = MultiHeadAttention(config)

        self.cross_attention = MultiHeadAttention(config)

        self.feed_forward = nn.Sequential(
            nn.Linear(config.d_model, config.d_hidden),
            nn.GELU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.d_hidden, config.d_model),
        )

        self.layer_norm1 = nn.LayerNorm(config.d_model)
        self.layer_norm2 = nn.LayerNorm(config.d_model)
        self.layer_norm3 = nn.LayerNorm(config.d_model)


def forward(self, X, to_cross_attn=None, self_mask=None, cross_mask=None):

    masked_attn = self.masked_self_attention(self.layer_norm1(X), mask=self_mask)
    X = X + masked_attn

    if to_cross_attn is not None:
        cross_attn = self.cross_attention(Q=self.layer_norm2(X), K=to_cross_attn, V=to_cross_attn, mask=cross_mask)
        X = X + cross_attn

    feed_forward = self.feed_forward(self.layer_norm3(X))
    X = X + feed_forward

    return X

