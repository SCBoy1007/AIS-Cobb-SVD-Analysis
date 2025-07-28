import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, efficientnet_b3, efficientnet_b5
from torchvision.models import EfficientNet_B0_Weights, EfficientNet_B3_Weights, EfficientNet_B5_Weights


class EfficientNetFeatures(nn.Module):
    def __init__(self, variant='b0', pretrained=True):
        super(EfficientNetFeatures, self).__init__()

        # 选择EfficientNet变体
        if variant == 'b0':
            weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
            base_model = efficientnet_b0(weights=weights)
            self.output_channels = 1280
            self.final_channels = 128  # 保留更多通道
        elif variant == 'b3':
            weights = EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
            base_model = efficientnet_b3(weights=weights)
            self.output_channels = 1536
            self.final_channels = 152  # 保留更多通道
        elif variant == 'b5':
            weights = EfficientNet_B5_Weights.IMAGENET1K_V1 if pretrained else None
            base_model = efficientnet_b5(weights=weights)
            self.output_channels = 2048
            self.final_channels = 192  # 保留更多通道
        else:
            raise ValueError(f"Unsupported EfficientNet variant: {variant}")

        # 提取特征提取部分 (去掉分类器)
        self.features = nn.Sequential(*list(base_model.children())[:-2])

        # 添加上采样层 - 从1/32上采样到1/4 (需要上采样8倍)
        # 第一次上采样2x
        self.upsample1 = nn.Sequential(
            nn.Conv2d(self.output_channels, self.output_channels // 2, kernel_size=1),
            nn.BatchNorm2d(self.output_channels // 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(self.output_channels // 2, self.output_channels // 4, kernel_size=4, stride=2, padding=1)
        )

        # 第二次上采样2x
        self.upsample2 = nn.Sequential(
            nn.BatchNorm2d(self.output_channels // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(self.output_channels // 4, self.output_channels // 8, kernel_size=4, stride=2, padding=1)
        )

        # 第三次上采样2x
        self.upsample3 = nn.Sequential(
            nn.BatchNorm2d(self.output_channels // 8),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(self.output_channels // 8, self.final_channels, kernel_size=4, stride=2, padding=1)
        )

        # 最后的特征整合层
        self.final_layer = nn.Sequential(
            nn.BatchNorm2d(self.final_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.final_channels, self.final_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.final_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # 获取特征（没有全局平均池化）
        x = self.features(x)

        # 上采样到1/16尺寸
        x = self.upsample1(x)

        # 上采样到1/8尺寸
        x = self.upsample2(x)

        # 上采样到1/4尺寸
        x = self.upsample3(x)

        # 最终特征整合
        x = self.final_layer(x)

        return x


def efficientnet_b0_backbone(pretrained=True):
    """
    EfficientNet-B0 model with proper upsampling for detection tasks
    """
    return EfficientNetFeatures('b0', pretrained)


def efficientnet_b3_backbone(pretrained=True):
    """
    EfficientNet-B3 model with proper upsampling for detection tasks
    """
    return EfficientNetFeatures('b3', pretrained)


def efficientnet_b5_backbone(pretrained=True):
    """
    EfficientNet-B5 model with proper upsampling for detection tasks
    """
    return EfficientNetFeatures('b5', pretrained)