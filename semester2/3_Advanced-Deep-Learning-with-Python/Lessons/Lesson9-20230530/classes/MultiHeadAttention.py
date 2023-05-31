import torch
import torch.nn as nn
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int) -> None:
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisable with num_heads"
        # set innerl module parameters
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # define matrix related parameters Qw, Kw, and Vw as linear transformation layers
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask=None) -> torch.Tensor:
        attn_scores = torch.matmul(
            Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        # Row-wise application of the softmax transfer function
        attn_probs = torch.softmax(attn_scores, -1)

        output = torch.matmul(attn_probs, V)
        return output

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, d_model = x.size()
        return x.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, seq_len, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask=None):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))

        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)

        output = self.W_o(self.combine_heads(attn_output))
        return output
