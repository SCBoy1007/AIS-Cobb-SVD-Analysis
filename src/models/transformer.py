import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class EfficientWindowAttention(nn.Module):
    """内存优化的窗口注意力机制，支持非对称窗口"""

    def __init__(self, dim, window_height, window_width, num_heads=8, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_height = window_height
        self.window_width = window_width
        self.window_area = window_height * window_width
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # QKV投影
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # 相对位置偏置表 - 调整为支持非对称窗口
        height_size = 2 * window_height - 1
        width_size = 2 * window_width - 1
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(height_size * width_size, num_heads))

        # 位置索引 - 调整为支持非对称窗口
        coords_h = torch.arange(window_height)
        coords_w = torch.arange(window_width)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2

        # 调整相对位置偏置的计算方式
        relative_coords[:, :, 0] += window_height - 1  # 偏移到 [0, 2*Wh-1]
        relative_coords[:, :, 1] += window_width - 1  # 偏移到 [0, 2*Ww-1]
        relative_coords[:, :, 0] *= width_size
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww

        self.register_buffer("relative_position_index", relative_position_index)

        # 初始化
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x):
        B_, N, C = x.shape
        # QKV投影
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # B_, H, N, C//H

        # 注意力分数
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # B_, H, N, N

        # 添加相对位置偏置
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        # softmax归一化
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # 应用注意力
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class MemoryEfficientFeedForward(nn.Module):
    """内存高效的前馈网络"""

    def __init__(self, dim, hidden_dim=None, out_dim=None, dropout=0.):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        out_dim = out_dim or dim

        self.w1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.w2 = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.w1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.w2(x)
        x = self.dropout(x)
        return x


def window_partition(x, window_height, window_width):
    """将特征图划分为不重叠的窗口，支持非对称窗口尺寸"""
    B, C, H, W = x.shape
    x = x.view(B, C, H // window_height, window_height, W // window_width, window_width)
    windows = x.permute(0, 2, 4, 1, 3, 5).contiguous().view(-1, C, window_height, window_width)
    return windows


def window_reverse(windows, window_height, window_width, H, W):
    """将窗口重组为原始特征图格式，支持非对称窗口尺寸"""
    B = int(windows.shape[0] / (H * W / window_height / window_width))
    x = windows.view(B, H // window_height, W // window_width, -1, window_height, window_width)
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous().view(B, -1, H, W)
    return x


class EfficientTransformerBlock(nn.Module):
    """内存高效的Transformer块，支持非对称窗口和自定义移位"""

    def __init__(self, dim, num_heads, window_height=8, window_width=8,
                 shift_height=0, shift_width=0, mlp_ratio=4.,
                 qkv_bias=True, drop=0., attn_drop=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_height = window_height
        self.window_width = window_width
        self.shift_height = shift_height
        self.shift_width = shift_width
        self.mlp_ratio = mlp_ratio

        # 标准化和注意力
        self.norm1 = norm_layer(dim)
        self.attn = EfficientWindowAttention(
            dim, window_height=window_height, window_width=window_width,
            num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop)

        # 前馈网络
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MemoryEfficientFeedForward(dim=dim, hidden_dim=mlp_hidden_dim, dropout=drop)

    def forward(self, x, H, W):
        B, L, C = x.shape
        assert L == H * W, f"输入特征的空间大小与H×W不一致! 得到{L}，期望{H * W}"

        shortcut = x
        x = x.view(B, H, W, C)

        # 移位窗口
        if self.shift_height > 0 or self.shift_width > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_height, -self.shift_width), dims=(1, 2))
        else:
            shifted_x = x

        # 窗口划分
        x_windows = window_partition(shifted_x.permute(0, 3, 1, 2),
                                     self.window_height, self.window_width)  # B*nW, C, window_height, window_width
        x_windows = x_windows.flatten(2).transpose(1, 2)  # B*nW, window_area, C

        # 窗口注意力
        attn_windows = self.attn(self.norm1(x_windows))  # B*nW, window_area, C

        # 窗口合并
        attn_windows = attn_windows.view(-1, self.window_height, self.window_width, C)
        attn_windows = attn_windows.permute(0, 3, 1, 2)  # B*nW, C, window_height, window_width

        # 重组特征图
        shifted_x = window_reverse(attn_windows, self.window_height, self.window_width, H, W)  # B, C, H, W

        # 如果使用了移位，需要反向移位
        if self.shift_height > 0 or self.shift_width > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_height, self.shift_width), dims=(2, 3))
        else:
            x = shifted_x

        # 转换回 B, L, C 格式并应用残差连接
        x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)  # B, L, C
        x = shortcut + x

        # 前馈网络
        x = x + self.mlp(self.norm2(x))

        return x


class OptimizedTransformerEnhancement(nn.Module):
    """带有降采样的内存优化Transformer特征增强模块"""

    def __init__(self, in_channels, depth=4, num_heads=4,
                 window_height=8, window_width=8, shift_size=None,
                 channel_expansion=4, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0.1, attn_drop_rate=0.1, drop_path_rate=0.1,
                 downsample_factor=4):
        super().__init__()

        self.downsample_factor = downsample_factor

        # 降采样层
        self.downsample = nn.Sequential()
        current_dim = in_channels
        for i in range(int(math.log2(downsample_factor))):
            self.downsample.add_module(f"down_{i}_conv", nn.Conv2d(current_dim, current_dim * 2,
                                                                   kernel_size=3, stride=2, padding=1))
            self.downsample.add_module(f"down_{i}_norm", nn.BatchNorm2d(current_dim * 2))
            self.downsample.add_module(f"down_{i}_act", nn.GELU())
            current_dim = current_dim * 2

        # 计算降采样后的通道数
        downsampled_channels = in_channels * (2 ** int(math.log2(downsample_factor)))

        # 在降采样特征上的通道扩展
        expanded_channels = downsampled_channels * channel_expansion

        print(
            f"降采样Transformer: 输入通道={in_channels}, 降采样通道={downsampled_channels}, 扩展通道={expanded_channels}")
        print(f"降采样因子={downsample_factor}, 头数={num_heads}, 深度={depth}")
        print(f"窗口大小: 高={window_height}, 宽={window_width}")

        # 计算移位大小
        if shift_size is None:
            shift_height = window_height // 2
            shift_width = window_width // 2
        else:
            shift_height = shift_width = shift_size

        print(f"移位大小: 高={shift_height}, 宽={shift_width}")

        # 特征投影
        self.input_proj = nn.Sequential(
            nn.Conv2d(downsampled_channels, expanded_channels, kernel_size=1),
            nn.BatchNorm2d(expanded_channels),
            nn.GELU()
        )

        # 创建Transformer块
        self.transformer_blocks = nn.ModuleList()
        for i in range(depth):
            # 交替使用标准窗口和移位窗口
            use_shift = (i % 2 == 1)
            current_shift_height = shift_height if use_shift else 0
            current_shift_width = shift_width if use_shift else 0

            self.transformer_blocks.append(
                EfficientTransformerBlock(
                    dim=expanded_channels,
                    num_heads=num_heads,
                    window_height=window_height,
                    window_width=window_width,
                    shift_height=current_shift_height,
                    shift_width=current_shift_width,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate
                )
            )

        # 输出投影 - 将通道降回降采样后的通道数
        self.output_proj = nn.Sequential(
            nn.Conv2d(expanded_channels, downsampled_channels, kernel_size=1),
            nn.BatchNorm2d(downsampled_channels),
            nn.GELU()
        )

        # 上采样层
        self.upsample = nn.Sequential()
        current_dim = downsampled_channels
        for i in range(int(math.log2(downsample_factor))):
            self.upsample.add_module(f"up_{i}_conv", nn.ConvTranspose2d(current_dim, current_dim // 2,
                                                                        kernel_size=4, stride=2, padding=1))
            self.upsample.add_module(f"up_{i}_norm", nn.BatchNorm2d(current_dim // 2))
            self.upsample.add_module(f"up_{i}_act", nn.GELU())
            current_dim = current_dim // 2

        # 残差特征增强 - 在原始分辨率上
        self.feature_enhance = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.GELU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
        )

        # 标准化
        self.norm = nn.LayerNorm(expanded_channels)

        # 保存窗口参数供forward使用
        self.window_height = window_height
        self.window_width = window_width

        # 初始化权重
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 保存原始输入
        identity = x
        B, C, H, W = x.shape

        # 应用降采样
        x = self.downsample(x)
        _, C_down, H_down, W_down = x.shape

        # 特征投影
        x = self.input_proj(x)  # B, C*expansion, H_down, W_down
        _, C_exp, _, _ = x.shape

        # 确保尺寸是window_size的整数倍
        pad_h = (self.window_height - H_down % self.window_height) % self.window_height
        pad_w = (self.window_width - W_down % self.window_width) % self.window_width
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))

        _, _, padded_H, padded_W = x.shape

        # 转换格式并应用标准化
        x = x.permute(0, 2, 3, 1).reshape(B, padded_H * padded_W, C_exp)  # B, L, C
        x = self.norm(x)

        # 应用Transformer块
        for block in self.transformer_blocks:
            x = block(x, padded_H, padded_W)

        # 转换回空间格式
        x = x.reshape(B, padded_H, padded_W, C_exp).permute(0, 3, 1, 2)  # B, C, H, W

        # 移除填充
        if pad_h > 0 or pad_w > 0:
            x = x[:, :, :H_down, :W_down]

        # 输出投影
        x = self.output_proj(x)  # B, C_down, H_down, W_down

        # 上采样回原始分辨率
        x = self.upsample(x)  # B, C, H, W

        # 确保上采样后的尺寸与输入匹配
        if x.size(2) != H or x.size(3) != W:
            x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)

        # 特征增强和残差连接
        enhanced = self.feature_enhance(x)
        x = enhanced + identity

        return x


# 保持原始类名的别名以兼容性
TransformerEnhancement = OptimizedTransformerEnhancement