# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Various positional encodings for the transformer.
"""
import math
import torch
from torch import nn

from util.misc import NestedTensor


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, tensor_list: NestedTensor):
        # 1. 输入完整特征图
        x = tensor_list.tensors
        mask = tensor_list.mask
        assert mask is not None
        # 2. 取其中未被 mask 的部分
        not_mask = ~mask
        # 3. 计算累积和得到坐标信息
        y_embed = not_mask.cumsum(1, dtype=torch.float32)    # 垂直方向累加
        x_embed = not_mask.cumsum(2, dtype=torch.float32)    # 水平方向累加
        # 4. 将坐标归一化到 [0,scale] 范围，scale 默认为 2pi
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        """
        5. 位置编码中的频率生成部分
        """
        # 5.1 生成一个序列
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        # 假设 num_pos_feats=64, 结果是: [0,1,2,3,...,63]
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        # 应用这个公式进一步生成一组指数递增的频率值，低位索引对应低频率(数值小)，高位索引对应高频率(数值大)

        # 6. 生成正弦波
        # 假设 num_pos_feats = 4 时，简化后的频率值 dim_t = [1, 1, 100, 100]
        pos_x = x_embed[:, :, :, None] / dim_t
        # 假设 x_embed = 5，pos_x = [5/1, 5/1, 5/100, 5/100] = [5, 5, 0.05, 0.05]
        pos_y = y_embed[:, :, :, None] / dim_t
        # 假设 y_embed= 5，pos_y = [5/1, 5/1, 5/100, 5/100] = [5, 5, 0.05, 0.05]
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        # [sin(5), cos(5), sin(0.05), cos(0.05)]
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        # [sin(5), cos(5), sin(0.05), cos(0.05)]
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)

        return pos


class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """
    def __init__(self, num_pos_feats=256):
        super().__init__()
        self.row_embed = nn.Embedding(50, num_pos_feats)
        self.col_embed = nn.Embedding(50, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors
        h, w = x.shape[-2:]
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        return pos


def build_position_encoding(args):
    N_steps = args.hidden_dim // 2
    if args.position_embedding in ('v2', 'sine'):
        # TODO find a better way of exposing other arguments
        position_embedding = PositionEmbeddingSine(N_steps, normalize=True)
    elif args.position_embedding in ('v3', 'learned'):
        position_embedding = PositionEmbeddingLearned(N_steps)
    else:
        raise ValueError(f"not supported {args.position_embedding}")

    return position_embedding
