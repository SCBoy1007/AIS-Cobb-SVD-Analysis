import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    """LayerNorm for channels first tensors"""

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class SimpleMambaUnit(nn.Module):
    """极简的Mamba风格块，仅使用卷积和通道注意力"""

    def __init__(self, channels, expansion=2, dropout=0.0):
        super().__init__()
        self.channels = channels
        self.inner_channels = int(channels * expansion)

        # 投影层
        self.norm1 = nn.BatchNorm2d(channels)
        self.proj_in = nn.Conv2d(channels, self.inner_channels * 2, kernel_size=1)

        # 深度卷积
        self.depth_conv = nn.Conv2d(
            self.inner_channels,
            self.inner_channels,
            kernel_size=3,
            padding=1,
            groups=self.inner_channels
        )

        # 通道注意力 - 模拟全局上下文
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.inner_channels, self.inner_channels // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.inner_channels // 4, self.inner_channels, kernel_size=1),
            nn.Sigmoid()
        )

        # 输出投影
        self.norm2 = nn.BatchNorm2d(self.inner_channels)
        self.proj_out = nn.Conv2d(self.inner_channels, channels, kernel_size=1)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        # 保存输入用于残差连接
        shortcut = x

        # 规范化
        x = self.norm1(x)

        # 投影和门控
        x = self.proj_in(x)
        x1, x2 = torch.chunk(x, 2, dim=1)

        # 深度卷积处理
        x1 = self.depth_conv(x1)
        x1 = F.gelu(x1)

        # 通道注意力
        x1 = x1 * self.channel_attention(x1)

        # 第二次规范化
        x1 = self.norm2(x1)

        # 门控机制
        x1 = x1 * F.gelu(x2)

        # 输出投影
        x1 = self.proj_out(x1)
        x1 = self.dropout(x1)

        # 残差连接
        return x1 + shortcut


class MambaBlock(nn.Module):
    """一组Mamba单元块"""

    def __init__(
            self,
            dim,
            depth=1,
            d_state=16,  # 为保持API一致，但不实际使用
            expand=2,
            dropout=0.0,
            **kwargs  # 捕获任何其他参数
    ):
        super().__init__()
        self.blocks = nn.ModuleList([
            SimpleMambaUnit(
                channels=dim,
                expansion=expand,
                dropout=dropout
            ) for _ in range(depth)
        ])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class VisionMambaBackbone(nn.Module):
    """Vision Mamba骨干网络 - 极简版本"""

    def __init__(
            self,
            in_channels=3,
            depths=[2, 2, 9, 2],
            dims=[96, 192, 384, 768],
            patch_size=4,
            d_state=16,  # 为保持API一致，但不实际使用
            dropout=0.0,
            feature_reduction=True,
            **kwargs  # 捕获任何其他参数
    ):
        super().__init__()
        self.depths = depths
        self.dims = dims
        self.feature_reduction = feature_reduction

        # Patch embedding
        self.patch_embed = nn.Sequential(
            nn.Conv2d(in_channels, dims[0], kernel_size=patch_size, stride=patch_size),
            nn.BatchNorm2d(dims[0])
        )

        # 构建编码器阶段
        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        # 构建每个阶段
        for i in range(len(depths)):
            stage = MambaBlock(
                dim=dims[i],
                depth=depths[i],
                expand=2,
                dropout=dropout
            )
            self.stages.append(stage)

            # 添加下采样层(除了最后一个阶段)
            if i < len(depths) - 1:
                downsample = nn.Sequential(
                    nn.BatchNorm2d(dims[i]),
                    nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2)
                )
                self.downsamples.append(downsample)

        # 解码器阶段 - 上采样和融合
        self.decoders = nn.ModuleList()
        self.upsamples = nn.ModuleList()

        # 从最深层开始向上
        for i in range(len(depths) - 1):
            idx = len(depths) - i - 2  # 从倒数第二个开始

            # 上采样层
            upsample = nn.ConvTranspose2d(
                dims[idx + 1], dims[idx], kernel_size=2, stride=2
            )
            self.upsamples.append(upsample)

            # 融合层
            decoder = nn.Sequential(
                nn.Conv2d(dims[idx] * 2, dims[idx], kernel_size=3, padding=1),
                nn.BatchNorm2d(dims[idx]),
                nn.ReLU(inplace=True)
            )
            self.decoders.append(decoder)

        # 输出层 - 降维到32通道
        if feature_reduction:
            self.reduce = nn.Sequential(
                nn.Conv2d(dims[0], 64, kernel_size=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True)
            )
        else:
            self.reduce = nn.Identity()

    def forward(self, x):
        # 记录输入尺寸以确保输出尺寸正确
        input_shape = x.shape[-2:]
        target_shape = (input_shape[0] // 4, input_shape[1] // 4)

        # Patch embedding
        x = self.patch_embed(x)

        # 编码器路径 - 保存各级特征
        features = [x]

        # 编码器前向传播
        for i, (stage, downsample) in enumerate(zip(self.stages[:-1], self.downsamples)):
            x = stage(x)
            features.append(x)  # 保存当前特征
            x = downsample(x)  # 下采样

        # 最后一个阶段没有下采样
        x = self.stages[-1](x)
        features.append(x)

        # 解码器路径 - 上采样并融合
        for i, (upsample, decoder) in enumerate(zip(self.upsamples, self.decoders)):
            # 获取编码器对应的特征
            skip_idx = len(features) - i - 2
            skip = features[skip_idx]

            # 上采样
            x = upsample(x)

            # 处理大小不匹配
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)

            # 拼接特征并融合
            x = torch.cat([x, skip], dim=1)
            x = decoder(x)

        # 确保输出尺寸正确
        if x.shape[2:] != target_shape:
            x = F.interpolate(x, size=target_shape, mode='bilinear', align_corners=False)

        # 应用特征降维
        x = self.reduce(x)

        return x


def vision_mamba_backbone(pretrained=False):
    """创建简化版Vision Mamba骨干网络，完全避免维度问题"""
    return VisionMambaBackbone(
        in_channels=3,
        depths=[2, 2, 4, 2],  # 减少深度以加快训练
        dims=[96, 192, 384, 768],
        patch_size=4,
        dropout=0.0,
        feature_reduction=True
    )