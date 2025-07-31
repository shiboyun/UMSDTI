# -*- coding: utf-8 -*-
import pdb
import math
from torch import Tensor
import torch.nn.init as init
from typing import Tuple, Optional
import torch
import torch.nn as nn
import numpy as np
from math import sqrt


class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask


class ProbMask():
    def __init__(self, B, H, L, index, scores, device="cpu"):
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[torch.arange(B)[:, None, None],
                    torch.arange(H)[None, :, None],
                    index, :].to(device)
        self._mask = indicator.view(scores.shape).to(device)

    @property
    def mask(self):
        return self._mask


class ProbAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=15, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):  # n_top: c*ln(L_q)
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        # 将（32，8，96，64）先变成（32，8，1，96，64）再变成（32，8，96，96，64）也就是将后面的维度的（96，64）看成一个整体进行复制96份
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        # 生成一个(L_Q, sample_k，矩阵中的整数取值范围是0 - L_K
        index_sample = torch.randint(L_K, (L_Q, sample_k))  # real U = U_part(factor*ln(L_k))*L_q
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze(-2)

        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   M_top, :]  # factor*ln(L_q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))  # factor*ln(L_q)*L_k

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            # V_sum = V.sum(dim=-2)
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else:  # use mask
            assert (L_Q == L_V)  # requires that L_Q == L_V, i.e. for self-attention only
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1)  # nn.Softmax(dim=-1)(scores)

        context_in[
            torch.arange(B)[:, None, None],
            torch.arange(H)[None, :, None],
            index,
            :
        ] = torch.matmul(attn, V).type_as(context_in)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V]) / L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)


    def forward(self, queries, keys, values, attn_mask):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)
        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item()  # c*ln(L_k)
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item()  # c*ln(L_q)

        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q

        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)

        # add scale factor
        scale = self.scale or 1. / sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values, L_Q)  # [bs, h, L_Q, dim]
        # update the context with selected top_k queries
        context, attn = self._update_context(context, values, scores_top, index, L_Q, attn_mask)

        return context.transpose(2, 1).contiguous(), attn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int = 512, max_len: int = 10000) -> None:
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model, requires_grad=False)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, length: int) -> Tensor:
        return self.pe[:, :length]

class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, inputs: Tensor) -> Tensor:
        return inputs * inputs.sigmoid()


class GLU(nn.Module):
    def __init__(self, dim: int) -> None:
        super(GLU, self).__init__()
        self.dim = dim

    def forward(self, inputs: Tensor) -> Tensor:
        outputs, gate = inputs.chunk(2, dim=self.dim)
        return outputs * gate.sigmoid()


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        init.xavier_uniform_(self.linear.weight)
        if bias:
            init.zeros_(self.linear.bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x)

class View(nn.Module):
    def __init__(self, shape: tuple, contiguous: bool = False):
        super(View, self).__init__()
        self.shape = shape
        self.contiguous = contiguous

    def forward(self, x: Tensor) -> Tensor:
        if self.contiguous:
            x = x.contiguous()

        return x.view(*self.shape)

class Transpose(nn.Module):
    def __init__(self, shape: tuple):
        super(Transpose, self).__init__()
        self.shape = shape

    def forward(self, x: Tensor) -> Tensor:
        return x.transpose(*self.shape)

class FeedForwardModule(nn.Module):
    def __init__(
            self,
            encoder_dim: int = 512,
            expansion_factor: int = 4,
            dropout_p: float = 0.1,
    ) -> None:
        super(FeedForwardModule, self).__init__()
        self.sequential = nn.Sequential(
            nn.LayerNorm(encoder_dim),
            Linear(encoder_dim, encoder_dim * expansion_factor, bias=True),
            Swish(),
            nn.Dropout(p=dropout_p),
            Linear(encoder_dim * expansion_factor, encoder_dim, bias=True),
            nn.Dropout(p=dropout_p),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.sequential(inputs)

class ConvModule(nn.Module):
    def __init__(
            self,
            in_channels: int,
            kernel_size: int = 5,
            dropout_p: float = 0.1,
    ) -> None:
        super(ConvModule, self).__init__()
        self.sequential = nn.Sequential(
            nn.LayerNorm(in_channels),
            Transpose(shape=(1, 2)),
            nn.Conv1d(in_channels=in_channels, out_channels=in_channels * 2,
                      kernel_size=1, stride=1, padding=0, bias=True, ),
            Swish(),
            nn.Conv1d(in_channels=in_channels * 2, out_channels=in_channels * 2, kernel_size=kernel_size,
                      stride=1, padding=(kernel_size - 1) // 2, bias=True,
                      ),
            GLU(dim=1),
            nn.BatchNorm1d(in_channels),
            nn.Conv1d(in_channels=in_channels, out_channels=in_channels,
                      kernel_size=1, stride=1, padding=0, bias=True, ),
            nn.Dropout(p=dropout_p),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.sequential(inputs).transpose(1, 2)

class ProbMultiHeadAttention(nn.Module):
    def __init__(
            self,
            d_model: int = 512,
            num_heads: int = 16,
    ):
        super(ProbMultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model % num_heads should be zero."
        self.d_model = d_model
        self.d_head = int(d_model / num_heads)
        self.num_heads = num_heads
        self.sqrt_dim = math.sqrt(d_model)

        self.query_proj = Linear(d_model, d_model)
        self.key_proj = Linear(d_model, d_model)
        self.value_proj = Linear(d_model, d_model)

        self.out_proj = Linear(d_model, d_model)
        self.prob_attention = ProbAttention()

    def forward(
            self,
            query: Tensor,
            key: Tensor,
            value: Tensor,
            mask: Tensor,
    ) -> Tensor:
        batch_size, seq_length, _ = value.size()
        query = self.query_proj(query).view(batch_size, -1, self.num_heads, self.d_head)
        key = self.key_proj(key).view(batch_size, -1, self.num_heads, self.d_head)
        value = self.value_proj(value).view(batch_size, -1, self.num_heads, self.d_head)
        context, _ = self.prob_attention(query, key, value, mask)

        return self.out_proj(context.view(batch_size, seq_length, -1))


class MultiHeadedSelfAttentionModule(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout_p: float = 0.1):
        super(MultiHeadedSelfAttentionModule, self).__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.attention = ProbMultiHeadAttention(d_model, num_heads)
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, inputs: Tensor, mask: Optional[Tensor] = None):

        inputs = self.layer_norm(inputs)
        outputs = self.attention(inputs, inputs, inputs, mask=mask)

        return self.dropout(outputs)

class CNNformerBlock(nn.Module):
    def __init__(
            self,
            encoder_dim: int = 512,
            num_attention_heads: int = 8,
            feed_forward_expansion_factor: int = 4,
            feed_forward_dropout_p: float = 0.1,
            attention_dropout_p: float = 0.1,
            conv_dropout_p: float = 0.1,
            conv_kernel_size: int = 31,
    ):
        super(CNNformerBlock, self).__init__()

        self.MHSA_model = MultiHeadedSelfAttentionModule(
            d_model=encoder_dim,
            num_heads=num_attention_heads,
            dropout_p=attention_dropout_p
        )
        self.CNN_model = ConvModule(
            in_channels=encoder_dim,
            kernel_size=conv_kernel_size,
            dropout_p=conv_dropout_p,
        )
        self.FF_model = FeedForwardModule(
            encoder_dim=encoder_dim,
            expansion_factor=feed_forward_expansion_factor,
            dropout_p=feed_forward_dropout_p,
        )

    def forward(self, inputs: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        MHSA_out = self.MHSA_model(inputs, mask) + inputs
        CNN_out = self.CNN_model(MHSA_out) + MHSA_out
        FFout = 0.5 * self.FF_model(CNN_out) + 0.5 * CNN_out
        return FFout

class CNNformerEncoder(nn.Module):
    def __init__(
            self,
            max_len: int = 10000,
            encoder_dim: int = 512,
            num_layers: int = 6,
            num_attention_heads: int = 8,
            feed_forward_expansion_factor: int = 4,
            feed_forward_dropout_p: float = 0.1,
            attention_dropout_p: float = 0.1,
            conv_dropout_p: float = 0.1,
            conv_kernel_size: int = 31,
    ):
        super(CNNformerEncoder, self).__init__()
        self.positional_encoding = PositionalEncoding(encoder_dim, max_len=max_len)
        self.CNNformerlayers = nn.ModuleList([CNNformerBlock(
            encoder_dim=encoder_dim,
            num_attention_heads=num_attention_heads,
            feed_forward_expansion_factor=feed_forward_expansion_factor,
            feed_forward_dropout_p=feed_forward_dropout_p,
            attention_dropout_p=attention_dropout_p,
            conv_dropout_p=conv_dropout_p,
            conv_kernel_size=conv_kernel_size
        ) for _ in range(num_layers)])

        self.FFlayers = nn.ModuleList([FeedForwardModule(
            encoder_dim=encoder_dim,
            expansion_factor=feed_forward_expansion_factor,
            dropout_p=feed_forward_dropout_p,
        ) for _ in range(num_layers)])

    def forward(self, inputs: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        _, seq_length, _ = inputs.size()
        pos_embedding = self.positional_encoding(seq_length)
        inputs += pos_embedding
        CNNformeroutputs = inputs
        for num in range(len(self.CNNformerlayers)):
            FF_output = 0.5 * self.FFlayers[num](CNNformeroutputs) + 0.5 * CNNformeroutputs
            CNNformeroutputs = self.CNNformerlayers[num](FF_output, mask)

        return CNNformeroutputs