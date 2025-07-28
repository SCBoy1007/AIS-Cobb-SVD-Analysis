import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import convnext_tiny, convnext_small, convnext_base


class LayerNorm2d(nn.Module):
    """特殊的LayerNorm变体，适用于通道优先(NCHW)的张量"""

    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class SpatialAttention(nn.Module):
    """增强版空间注意力模块 - 帮助模型关注空间上的特定点"""

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "空间注意力的卷积核大小必须是3或7"
        padding = 3 if kernel_size == 7 else 1

        # 标准注意力生成
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)

        # 增加锐化层，强调局部响应
        self.sharpen = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 生成空间注意力图
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention = torch.cat([avg_out, max_out], dim=1)
        attention = self.conv(attention)

        # 应用锐化，提高局部点的对比度
        attention = self.sharpen(attention)

        # 应用非线性变换，增强高响应区域
        attention_map = self.sigmoid(attention * 2.0)  # 增加系数放大对比度

        # 应用注意力
        return x * attention_map


class PointEnhancerModule(nn.Module):
    """点增强模块 - 专门设计用于突出局部最大值（点）"""

    def __init__(self, channels):
        super(PointEnhancerModule, self).__init__()

        # 局部响应增强
        self.local_enhancer = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
        )

        # 点特征提取器 - 使用小卷积核聚焦于点状特征
        self.point_extractor = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False, groups=channels),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

        # 最终融合
        self.fusion = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        # 局部增强
        local_features = self.local_enhancer(x)

        # 提取点特征
        point_features = self.point_extractor(x)

        # 应用局部抑制，突出点状响应
        enhanced = local_features * point_features

        # 融合原始特征
        result = self.fusion(enhanced) + x
        return self.activation(result)


class FocalBlock(nn.Module):
    """改进的聚焦块 - 帮助生成更点状而非线状的特征"""

    def __init__(self, channels):
        super(FocalBlock, self).__init__()

        # 第一阶段：广域上下文
        self.context_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False, groups=channels),
            LayerNorm2d(channels),
            nn.GELU()
        )

        # 第二阶段：局部聚焦 - 调整为更聚焦的结构
        self.focal_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            LayerNorm2d(channels),
            nn.GELU()
        )

        # 第三阶段：点增强 - 使用新的点增强模块
        self.point_enhancer = PointEnhancerModule(channels)

        # 注意力模块 - 使用改进的空间注意力
        self.attention = SpatialAttention(kernel_size=7)

        # 局部最大值增强 - 帮助生成更鲜明的点
        self.local_max_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        context = self.context_conv(x)
        focal = self.focal_conv(context)

        # 提取局部最大值作为点候选
        local_max = self.local_max_pool(focal)
        # 点增强 - 只有当像素值等于局部最大值时才保持高值
        point_mask = (focal == local_max).float()
        focal = focal * (0.7 + 0.3 * point_mask)  # 保留一些原始信息

        focused = self.attention(focal)
        enhanced = self.point_enhancer(focused)

        # 残差连接
        return x + enhanced


class LocalContrastNormalization(nn.Module):
    """局部对比度归一化 - 帮助突出局部峰值"""

    def __init__(self, kernel_size=5):
        super(LocalContrastNormalization, self).__init__()
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

    def forward(self, x):
        # 计算局部均值
        weight = torch.ones(1, 1, self.kernel_size, self.kernel_size).to(x.device) / (self.kernel_size ** 2)
        local_mean = F.conv2d(x, weight.repeat(x.size(1), 1, 1, 1), padding=self.padding, groups=x.size(1))

        # 计算局部标准差
        local_var = F.conv2d((x - local_mean) ** 2, weight.repeat(x.size(1), 1, 1, 1),
                             padding=self.padding, groups=x.size(1))
        local_std = torch.sqrt(local_var + 1e-5)

        # 归一化
        normalized = (x - local_mean) / (local_std + 1e-5)
        return normalized


class ConvNeXtFeatures(nn.Module):
    """提取ConvNeXt特征的封装器，返回单一点状特征图"""

    def __init__(self, model_type='tiny'):
        super(ConvNeXtFeatures, self).__init__()

        # 加载预训练的ConvNeXt模型
        if model_type == 'tiny':
            self.feature_channels = 768
            weights = None
            model = convnext_tiny(weights=weights)
        elif model_type == 'small':
            self.feature_channels = 768
            weights = None
            model = convnext_small(weights=weights)
        elif model_type == 'base':
            self.feature_channels = 1024
            weights = None
            model = convnext_base(weights=weights)
        else:
            raise ValueError(f"Unsupported ConvNeXt model type: {model_type}")

        # 手动加载预训练权重
        if model_type == 'tiny':
            pretrained_dict = torch.hub.load_state_dict_from_url(
                'https://download.pytorch.org/models/convnext_tiny-983f1562.pth',
                progress=True
            )
            model.load_state_dict(pretrained_dict)
        elif model_type == 'small':
            pretrained_dict = torch.hub.load_state_dict_from_url(
                'https://download.pytorch.org/models/convnext_small-0c510722.pth',
                progress=True
            )
            model.load_state_dict(pretrained_dict)
        elif model_type == 'base':
            pretrained_dict = torch.hub.load_state_dict_from_url(
                'https://download.pytorch.org/models/convnext_base-6075fbad.pth',
                progress=True
            )
            model.load_state_dict(pretrained_dict)

        # 提取ConvNeXt的特征提取部分
        self.features = model.features

        # 局部对比度归一化 - 用于增强点特征
        self.local_contrast_norm = LocalContrastNormalization(kernel_size=5)

        # 通道减少层 - 更多地关注于生成点状特征
        self.reduce_channels = nn.Sequential(
            # 初始降维
            nn.Conv2d(self.feature_channels, 256, kernel_size=1, bias=False),
            LayerNorm2d(256),
            nn.GELU(),

            # 应用局部对比度归一化
            nn.BatchNorm2d(256),

            # 第一级聚焦 - 使用点增强结构
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(128),
            nn.GELU(),
            nn.BatchNorm2d(128),

            # 第二级聚焦
            nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False, groups=8),  # 分组卷积保留局部结构
            LayerNorm2d(64),
            nn.GELU(),
            nn.BatchNorm2d(64),

            # 最终降维
            nn.Conv2d(64, 32, kernel_size=1, bias=False),
            LayerNorm2d(32),
            nn.GELU(),
            nn.BatchNorm2d(32),
        )

        # 特征聚焦模块 - 使用改进的FocalBlock
        self.point_focus = nn.Sequential(
            FocalBlock(32),
            FocalBlock(32)
        )

        # 点增强层 - 额外添加一个专门的点增强层
        self.final_point_enhancer = nn.Sequential(
            PointEnhancerModule(32),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        input_shape = x.shape[-2:]
        target_shape = (input_shape[0] // 4, input_shape[1] // 4)

        # 提取ConvNeXt特征
        for i, stage in enumerate(self.features):
            x = stage(x)

        # 应用局部对比度归一化
        x = self.local_contrast_norm(x)

        # 减少通道数
        x = self.reduce_channels(x)

        # 应用点聚焦模块
        x = self.point_focus(x)

        # 应用最终点增强
        x = self.final_point_enhancer(x)

        # 确保输出是输入的1/4大小
        if x.shape[-2:] != target_shape:
            x = F.interpolate(x, size=target_shape, mode='bilinear', align_corners=False)

        return x


def convnext_tiny_backbone(pretrained=True):
    """创建ConvNeXt Tiny作为backbone，返回单一特征图"""
    return ConvNeXtFeatures(model_type='tiny')


def convnext_small_backbone(pretrained=True):
    """创建ConvNeXt Small作为backbone，返回单一特征图"""
    return ConvNeXtFeatures(model_type='small')


def convnext_base_backbone(pretrained=True):
    """创建ConvNeXt Base作为backbone，返回单一特征图"""
    return ConvNeXtFeatures(model_type='base')