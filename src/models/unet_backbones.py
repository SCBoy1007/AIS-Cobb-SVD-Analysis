import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(Conv2d -> BN -> ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, dropout_rate=0.1):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate) if dropout_rate > 0 else nn.Identity()
        )

    def forward(self, x):
        return self.double_conv(x)


class OptimizedUNetBackbone(nn.Module):
    """优化的UNet，减少通道数和简化结构"""

    def __init__(self, n_channels=3, feature_reduction=True, dropout_rate=0.1):
        super(OptimizedUNetBackbone, self).__init__()

        # 基础通道配置
        base_channels = 32

        # 编码器部分
        self.inc = DoubleConv(n_channels, base_channels, dropout_rate=0)
        self.pool1 = nn.MaxPool2d(2)
        self.enc1 = DoubleConv(base_channels, base_channels * 2, dropout_rate=dropout_rate)
        self.pool2 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(base_channels * 2, base_channels * 4, dropout_rate=dropout_rate)
        self.pool3 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(base_channels * 4, base_channels * 8, dropout_rate=dropout_rate)
        self.pool4 = nn.MaxPool2d(2)
        self.enc4 = DoubleConv(base_channels * 8, base_channels * 16, dropout_rate=dropout_rate)

        # 解码器部分
        self.up1 = nn.ConvTranspose2d(base_channels * 16, base_channels * 8, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(base_channels * 16, base_channels * 8, dropout_rate=dropout_rate)

        self.up2 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(base_channels * 8, base_channels * 4, dropout_rate=dropout_rate)

        self.up3 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(base_channels * 4, base_channels * 2, dropout_rate=dropout_rate)

        self.up4 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(base_channels * 2, base_channels, dropout_rate=dropout_rate)

        # 输出层
        if feature_reduction and base_channels > 32:
            self.reduce = nn.Sequential(
                nn.Conv2d(base_channels, 32, kernel_size=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True)
            )
        else:
            self.reduce = nn.Identity()

    def forward(self, x):
        # 记录输入尺寸以确保输出是输入的1/4
        input_shape = x.shape[-2:]
        target_shape = (input_shape[0] // 4, input_shape[1] // 4)

        # 编码器路径
        x0 = self.inc(x)
        x1 = self.enc1(self.pool1(x0))
        x2 = self.enc2(self.pool2(x1))
        x3 = self.enc3(self.pool3(x2))
        x4 = self.enc4(self.pool4(x3))

        # 解码器路径
        x = self.up1(x4)
        x = self.dec1(torch.cat([x3, x], dim=1))

        x = self.up2(x)
        x = self.dec2(torch.cat([x2, x], dim=1))

        x = self.up3(x)
        x = self.dec3(torch.cat([x1, x], dim=1))

        x = self.up4(x)
        x = self.dec4(torch.cat([x0, x], dim=1))

        # 确保输出尺寸正确
        out = self.reduce(x)
        if out.shape[-2:] != target_shape:
            out = F.interpolate(out, size=target_shape, mode='bilinear', align_corners=False)

        return out


class SimpleUNetPlusPlusBackbone(nn.Module):
    """简化版的UNet++，使用更通用的实现方法"""

    def __init__(self, n_channels=3, feature_reduction=True, dropout_rate=0.1):
        super(SimpleUNetPlusPlusBackbone, self).__init__()

        # 基础通道配置
        base_channels = 32

        # 编码器部分
        self.inc = DoubleConv(n_channels, base_channels, dropout_rate=0)
        self.pool1 = nn.MaxPool2d(2)
        self.enc1 = DoubleConv(base_channels, base_channels * 2, dropout_rate=dropout_rate)
        self.pool2 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(base_channels * 2, base_channels * 4, dropout_rate=dropout_rate)
        self.pool3 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(base_channels * 4, base_channels * 8, dropout_rate=dropout_rate)
        self.pool4 = nn.MaxPool2d(2)
        self.enc4 = DoubleConv(base_channels * 8, base_channels * 16, dropout_rate=dropout_rate)

        # L1层 - 底层到顶层路径
        self.up1_0 = nn.ConvTranspose2d(base_channels * 16, base_channels * 8, kernel_size=2, stride=2)
        self.dec1_0 = DoubleConv(base_channels * 16, base_channels * 8, dropout_rate=dropout_rate)

        self.up2_0 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.dec2_0 = DoubleConv(base_channels * 8, base_channels * 4, dropout_rate=dropout_rate)

        self.up3_0 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec3_0 = DoubleConv(base_channels * 4, base_channels * 2, dropout_rate=dropout_rate)

        self.up4_0 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec4_0 = DoubleConv(base_channels * 2, base_channels, dropout_rate=dropout_rate)

        # L2层 - 嵌套连接
        self.up1_1 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.dec2_1 = DoubleConv(base_channels * 8, base_channels * 4, dropout_rate=dropout_rate)

        self.up2_1 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec3_1 = DoubleConv(base_channels * 4, base_channels * 2, dropout_rate=dropout_rate)

        self.up3_1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec4_1 = DoubleConv(base_channels * 2, base_channels, dropout_rate=dropout_rate)

        # L3层 - 更深的嵌套连接
        self.up1_2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec3_2 = DoubleConv(base_channels * 4, base_channels * 2, dropout_rate=dropout_rate)

        self.up2_2 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec4_2 = DoubleConv(base_channels * 2, base_channels, dropout_rate=dropout_rate)

        # L4层 - 最深的嵌套连接
        self.up1_3 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec4_3 = DoubleConv(base_channels * 2, base_channels, dropout_rate=dropout_rate)

        # 输出层
        if feature_reduction and base_channels > 32:
            self.reduce = nn.Sequential(
                nn.Conv2d(base_channels, 32, kernel_size=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True)
            )
        else:
            self.reduce = nn.Identity()

    def forward(self, x):
        # 记录输入尺寸以确保输出是输入的1/4
        input_shape = x.shape[-2:]
        target_shape = (input_shape[0] // 4, input_shape[1] // 4)

        # 编码器路径
        x0 = self.inc(x)
        x1 = self.enc1(self.pool1(x0))
        x2 = self.enc2(self.pool2(x1))
        x3 = self.enc3(self.pool3(x2))
        x4 = self.enc4(self.pool4(x3))

        # L1层 - 标准UNet路径
        x0_1 = self.dec1_0(torch.cat([x3, self.up1_0(x4)], dim=1))
        x0_2 = self.dec2_0(torch.cat([x2, self.up2_0(x3)], dim=1))
        x0_3 = self.dec3_0(torch.cat([x1, self.up3_0(x2)], dim=1))
        x0_4 = self.dec4_0(torch.cat([x0, self.up4_0(x1)], dim=1))

        # L2层 - 嵌套连接
        x1_1 = self.dec2_1(torch.cat([x2, self.up1_1(x0_1)], dim=1))
        x1_2 = self.dec3_1(torch.cat([x1, self.up2_1(x0_2)], dim=1))
        x1_3 = self.dec4_1(torch.cat([x0, self.up3_1(x0_3)], dim=1))

        # L3层 - 更深嵌套
        x2_1 = self.dec3_2(torch.cat([x1, self.up1_2(x1_1)], dim=1))
        x2_2 = self.dec4_2(torch.cat([x0, self.up2_2(x1_2)], dim=1))

        # L4层 - 最深嵌套
        x3_1 = self.dec4_3(torch.cat([x0, self.up1_3(x2_1)], dim=1))

        # 使用最深层的输出
        out = x3_1

        # 确保输出尺寸正确
        out = self.reduce(out)
        if out.shape[-2:] != target_shape:
            out = F.interpolate(out, size=target_shape, mode='bilinear', align_corners=False)

        return out


# 替换现有的unet_backbone函数，使用优化版UNet
def unet_backbone(pretrained=False):
    """创建优化版UNet作为backbone"""
    return OptimizedUNetBackbone(n_channels=3, feature_reduction=True, dropout_rate=0.1)


# 更新的unetplusplus_backbone函数使用简化版UNet++
def unetplusplus_backbone(pretrained=False):
    """创建简化版UNet++作为backbone"""
    return SimpleUNetPlusPlusBackbone(n_channels=3, feature_reduction=True, dropout_rate=0.1)