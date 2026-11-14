import copy
import math
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from infer.lib.infer_pack import commons, modules
from infer.lib.infer_pack.modules import LayerNorm


class Encoder(nn.Module):
    def __init__(
        self,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size=1,
        p_dropout=0.0,
        window_size=10,
        **kwargs
    ):
        super(Encoder, self).__init__()
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = int(n_layers)
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.window_size = window_size

        self.drop = nn.Dropout(p_dropout)
        self.attn_layers = nn.ModuleList()
        self.norm_layers_1 = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.norm_layers_2 = nn.ModuleList()
        for i in range(self.n_layers):
            self.attn_layers.append(
                MultiHeadAttention(
                    hidden_channels,
                    hidden_channels,
                    n_heads,
                    p_dropout=p_dropout,
                    window_size=window_size,
                )
            )
            self.norm_layers_1.append(LayerNorm(hidden_channels))
            self.ffn_layers.append(
                FFN(
                    hidden_channels,
                    hidden_channels,
                    filter_channels,
                    kernel_size,
                    p_dropout=p_dropout,
                )
            )
            self.norm_layers_2.append(LayerNorm(hidden_channels))

    def forward(self, x, x_mask, lengths=None):
        attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
        x = x * x_mask
        zippep = zip(
            self.attn_layers, self.norm_layers_1, self.ffn_layers, self.norm_layers_2
        )
        for attn_layers, norm_layers_1, ffn_layers, norm_layers_2 in zippep:
            y = attn_layers(x, x, attn_mask, lengths)
            y = self.drop(y)
            x = norm_layers_1(x + y)

            y = ffn_layers(x, x_mask)
            y = self.drop(y)
            x = norm_layers_2(x + y)
        x = x * x_mask
        return x


class Decoder(nn.Module):
    def __init__(
        self,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size=1,
        p_dropout=0.0,
        proximal_bias=False,
        proximal_init=True,
        **kwargs
    ):
        super(Decoder, self).__init__()
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.proximal_bias = proximal_bias
        self.proximal_init = proximal_init

        self.drop = nn.Dropout(p_dropout)
        self.self_attn_layers = nn.ModuleList()
        self.norm_layers_0 = nn.ModuleList()
        self.encdec_attn_layers = nn.ModuleList()
        self.norm_layers_1 = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.norm_layers_2 = nn.ModuleList()
        for i in range(self.n_layers):
            self.self_attn_layers.append(
                MultiHeadAttention(
                    hidden_channels,
                    hidden_channels,
                    n_heads,
                    p_dropout=p_dropout,
                    proximal_bias=proximal_bias,
                    proximal_init=proximal_init,
                )
            )
            self.norm_layers_0.append(LayerNorm(hidden_channels))
            self.encdec_attn_layers.append(
                MultiHeadAttention(
                    hidden_channels, hidden_channels, n_heads, p_dropout=p_dropout
                )
            )
            self.norm_layers_1.append(LayerNorm(hidden_channels))
            self.ffn_layers.append(
                FFN(
                    hidden_channels,
                    hidden_channels,
                    filter_channels,
                    kernel_size,
                    p_dropout=p_dropout,
                    causal=True,
                )
            )
            self.norm_layers_2.append(LayerNorm(hidden_channels))

    def forward(self, x, x_mask, h, h_mask):
        """
        x: decoder input
        h: encoder output
        """
        self_attn_mask = commons.subsequent_mask(x_mask.size(2)).to(
            device=x.device, dtype=x.dtype
        )
        encdec_attn_mask = h_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
        x = x * x_mask
        for i in range(self.n_layers):
            y = self.self_attn_layers[i](x, x, self_attn_mask)
            y = self.drop(y)
            x = self.norm_layers_0[i](x + y)

            y = self.encdec_attn_layers[i](x, h, encdec_attn_mask)
            y = self.drop(y)
            x = self.norm_layers_1[i](x + y)

            y = self.ffn_layers[i](x, x_mask)
            y = self.drop(y)
            x = self.norm_layers_2[i](x + y)
        x = x * x_mask
        return x


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        channels,
        out_channels,
        n_heads,
        p_dropout=0.0,
        window_size=None,
        heads_share=True,
        block_length=None,
        proximal_bias=False,
        proximal_init=False,
    ):
        super(MultiHeadAttention, self).__init__()
        assert channels % n_heads == 0

        self.channels = channels
        self.out_channels = out_channels
        self.n_heads = n_heads
        self.p_dropout = p_dropout
        self.window_size = window_size
        self.heads_share = heads_share
        self.block_length = block_length
        self.proximal_bias = proximal_bias
        self.proximal_init = proximal_init
        self.attn = None

        self.k_channels = channels // n_heads
        self.conv_q = nn.Conv1d(channels, channels, 1)
        self.conv_k = nn.Conv1d(channels, channels, 1)
        self.conv_v = nn.Conv1d(channels, channels, 1)
        self.conv_o = nn.Conv1d(channels, out_channels, 1)
        self.drop = nn.Dropout(p_dropout)

        if window_size is not None:
            n_heads_rel = 1 if heads_share else n_heads
            rel_stddev = self.k_channels**-0.5
            self.emb_rel_k = nn.Parameter(
                torch.randn(n_heads_rel, window_size * 2 + 1, self.k_channels)
                * rel_stddev
            )
            self.emb_rel_v = nn.Parameter(
                torch.randn(n_heads_rel, window_size * 2 + 1, self.k_channels)
                * rel_stddev
            )

        nn.init.xavier_uniform_(self.conv_q.weight)
        nn.init.xavier_uniform_(self.conv_k.weight)
        nn.init.xavier_uniform_(self.conv_v.weight)
        if proximal_init:
            with torch.no_grad():
                self.conv_k.weight.copy_(self.conv_q.weight)
                self.conv_k.bias.copy_(self.conv_q.bias)

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None,
    ):
        q = self.conv_q(x)
        k = self.conv_k(c)
        v = self.conv_v(c)

        x, _ = self.attention(q, k, v, mask=attn_mask, lengths=lengths)

        x = self.conv_o(x)
        return x

    def attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None,
    ):
        # reshape [b, d, t] -> [b, n_h, t, d_k]
        # 注意: 时间维用 -1 让 ONNX 在运行时推断，避免 trace 时被冻结成导出时的长度
        b, d, t_s = key.size()
        t_t = query.size(2)
        query = query.view(b, self.n_heads, self.k_channels, -1).transpose(2, 3)
        key = key.view(b, self.n_heads, self.k_channels, -1).transpose(2, 3)
        value = value.view(b, self.n_heads, self.k_channels, -1).transpose(2, 3)

        scores = torch.matmul(query / math.sqrt(self.k_channels), key.transpose(-2, -1))
        if self.window_size is not None:
            assert (
                t_s == t_t
            ), "Relative attention is only available for self-attention."
            # 动态帧数: 导出时来自 phone_lengths 输入张量（[1]），保持动态不冻结
            if lengths is not None:
                length = lengths.view([])
            else:
                length = t_t
            # scores_local[i, j] = sum_k q[i,k] * emb_rel_k[(j-i)+window, k]
            rel_raw = self._matmul_with_relative_keys(
                query / math.sqrt(self.k_channels), self.emb_rel_k
            )
            scores_local = self._relative_position_to_absolute_position(rel_raw, length)
            scores = scores + scores_local
        if self.proximal_bias:
            assert t_s == t_t, "Proximal bias is only available for self-attention."
            scores = scores + self._attention_bias_proximal(t_s).to(
                device=scores.device, dtype=scores.dtype
            )
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e4)
            if self.block_length is not None:
                assert (
                    t_s == t_t
                ), "Local attention is only available for self-attention."
                block_mask = (
                    torch.ones_like(scores)
                    .triu(-self.block_length)
                    .tril(self.block_length)
                )
                scores = scores.masked_fill(block_mask == 0, -1e4)
        p_attn = F.softmax(scores, dim=-1)  # [b, n_h, t_t, t_s]
        p_attn = self.drop(p_attn)
        output = torch.matmul(p_attn, value)
        if self.window_size is not None:
            output = output + self._absolute_position_to_relative_position(
                p_attn, length
            )
        output = (
            output.transpose(2, 3).contiguous().view(b, d, -1)
        )  # [b, n_h, t_t, d_k] -> [b, d, t_t]
        return output, p_attn

    def _matmul_with_relative_keys(self, x, y):
        """
        x: [b, h, l, d]
        y: [h or 1, m, d]
        ret: [b, h, l, m]
        """
        ret = torch.matmul(x, y.unsqueeze(0).transpose(-2, -1))
        return ret

    def _relative_position_to_absolute_position(self, rel_raw, length):
        """
        相对注意力 -> 绝对注意力: scores_local[i, j] = rel_raw[i, (j - i) + window]（越界为 0）。
        rel_raw: [b, h, l, 2*window+1]（q 与 emb_rel_k 的 matmul 结果）
        返回: [b, h, l, l]
        length: 动态标量张量（导出时来自 phone_lengths，不会被 trace 冻结）
        """
        batch, heads, _, _ = rel_raw.size()
        window = self.window_size
        # 下标 (j - i) + window ∈ [0, 2*window]，越界补 0。
        # 等价于原 F.pad + flatten/reshape 的逻辑，但只用动态 Range + gather，
        # 不产生按导出长度固化的 Reshape 节点。
        idx = (
            torch.arange(length, device=rel_raw.device).view(1, -1)  # j
            - torch.arange(length, device=rel_raw.device).view(-1, 1)  # i
            + window
        )  # [l, l]
        idx_clamped = torch.clamp(idx, 0, 2 * window)
        valid = (idx >= 0) & (idx <= 2 * window)
        gathered = torch.gather(rel_raw, 3, idx_clamped.expand(batch, heads, -1, -1))
        return gathered * valid.to(rel_raw.dtype)

    def _absolute_position_to_relative_position(self, p_attn, length):
        """
        绝对注意力 -> 相对注意力 value 贡献:
        out[i, k] = sum_{c'} p_attn[i, c' + i - window] * emb_rel_v[c', k]，c' ∈ [0, 2*window]。
        p_attn: [b, h, l, l]
        返回: [b, h, l, d_k]
        length: 动态标量张量
        """
        batch, heads, _, _ = p_attn.size()
        window = self.window_size
        # 对每个 query 位置 i，取 p_attn 第 (c' + i - window) 列（越界补 0）
        col = (
            torch.arange(length, device=p_attn.device).view(-1, 1)  # i
            + torch.arange(2 * window + 1, device=p_attn.device).view(1, -1)  # c'
            - window
        )  # [l, 2*window+1]
        col_clamped = torch.clamp(col, 0, length - 1)
        valid = (col >= 0) & (col < length)
        gathered = torch.gather(
            p_attn, 3, col_clamped.expand(batch, heads, -1, -1)
        )
        gathered = gathered * valid.to(p_attn.dtype)
        # [b, h, l, 2*window+1] @ [1, 2*window+1, d_k] -> [b, h, l, d_k]
        return torch.matmul(gathered, self.emb_rel_v)

    def _attention_bias_proximal(self, length: int):
        """Bias for self-attention to encourage attention to close positions.
        Args:
          length: an integer scalar.
        Returns:
          a Tensor with shape [1, 1, length, length]
        """
        r = torch.arange(length, dtype=torch.float32)
        diff = torch.unsqueeze(r, 0) - torch.unsqueeze(r, 1)
        return torch.unsqueeze(torch.unsqueeze(-torch.log1p(torch.abs(diff)), 0), 0)


class FFN(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        filter_channels,
        kernel_size,
        p_dropout=0.0,
        activation: str = None,
        causal=False,
    ):
        super(FFN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.activation = activation
        self.causal = causal
        self.is_activation = True if activation == "gelu" else False
        # if causal:
        #     self.padding = self._causal_padding
        # else:
        #     self.padding = self._same_padding

        self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size)
        self.conv_2 = nn.Conv1d(filter_channels, out_channels, kernel_size)
        self.drop = nn.Dropout(p_dropout)

    def padding(self, x: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:
        if self.causal:
            padding = self._causal_padding(x * x_mask)
        else:
            padding = self._same_padding(x * x_mask)
        return padding

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor):
        x = self.conv_1(self.padding(x, x_mask))
        if self.is_activation:
            x = x * torch.sigmoid(1.702 * x)
        else:
            x = torch.relu(x)
        x = self.drop(x)

        x = self.conv_2(self.padding(x, x_mask))
        return x * x_mask

    def _causal_padding(self, x):
        if self.kernel_size == 1:
            return x
        pad_l: int = self.kernel_size - 1
        pad_r: int = 0
        # padding = [[0, 0], [0, 0], [pad_l, pad_r]]
        x = F.pad(
            x,
            #   commons.convert_pad_shape(padding)
            [pad_l, pad_r, 0, 0, 0, 0],
        )
        return x

    def _same_padding(self, x):
        if self.kernel_size == 1:
            return x
        pad_l: int = (self.kernel_size - 1) // 2
        pad_r: int = self.kernel_size // 2
        # padding = [[0, 0], [0, 0], [pad_l, pad_r]]
        x = F.pad(
            x,
            #   commons.convert_pad_shape(padding)
            [pad_l, pad_r, 0, 0, 0, 0],
        )
        return x
