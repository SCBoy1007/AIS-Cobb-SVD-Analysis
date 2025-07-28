import torch
import torch.nn as nn
import torchvision.models as models


class DenseNetFeatures(nn.Module):
    def __init__(self, variant='densenet121', pretrained=True):
        super(DenseNetFeatures, self).__init__()

        # 选择DenseNet变体
        if variant == 'densenet121':
            base_model = models.densenet121(pretrained=pretrained)
            self.output_channels = 1024
            self.final_channels = 128  # 保留更多通道
        elif variant == 'densenet169':
            base_model = models.densenet169(pretrained=pretrained)
            self.output_channels = 1664
            self.final_channels = 160  # 保留更多通道
        elif variant == 'densenet201':
            base_model = models.densenet201(pretrained=pretrained)
            self.output_channels = 1920
            self.final_channels = 192  # 保留更多通道
        else:
            raise ValueError(f"Unsupported DenseNet variant: {variant}")

        # 提取特征层
        self.features = base_model.features

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
        # 获取DenseNet的特征图 (1/32尺寸)
        features = self.features(x)

        # 上采样到1/16尺寸
        x = self.upsample1(features)

        # 上采样到1/8尺寸
        x = self.upsample2(x)

        # 上采样到1/4尺寸
        x = self.upsample3(x)

        # 最终特征整合
        x = self.final_layer(x)

        return x


def densenet121(pretrained=True):
    """
    DenseNet-121 model with upsampling to match the expected feature map size
    """
    return DenseNetFeatures('densenet121', pretrained)


def densenet169(pretrained=True):
    """
    DenseNet-169 model with upsampling to match the expected feature map size
    """
    return DenseNetFeatures('densenet169', pretrained)


def densenet201(pretrained=True):
    """
    DenseNet-201 model with upsampling to match the expected feature map size
    """
    return DenseNetFeatures('densenet201', pretrained)