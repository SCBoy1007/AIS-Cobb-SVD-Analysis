import torch.nn as nn
import torch
import math
from .ffca import *
from .ipg import *
from .transformer import OptimizedTransformerEnhancement as TransformerEnhancement


class Decodeing(nn.Module):
    def __init__(self, final_kernel, head_conv, channel, use_gnn=False, use_trans=False, dropout_rate=0.1,
                 backbone='hrnet18', trans_config=None):
        super(Decodeing, self).__init__()

        self.dropout_rate = dropout_rate
        self.backbone = backbone

        self.use_gnn = use_gnn
        self.use_trans = use_trans

        # 判断是否是多尺度backbone
        self.is_multi_scale = backbone.startswith('hrnet')

        # 为不同backbone设置合适的通道数，保留更多原始信息
        if backbone.startswith('hrnet'):
            if backbone == 'hrnet18':
                self.dec_c2 = FFCA(36, 18, batch_norm=True)
                self.dec_c3 = FFCA(72, 36, batch_norm=True)
                self.dec_c4 = FFCA(144, 72, batch_norm=True)
                self.final_channels = 18
            elif backbone == 'hrnet32':
                self.dec_c2 = FFCA(64, 32, batch_norm=True)
                self.dec_c3 = FFCA(128, 64, batch_norm=True)
                self.dec_c4 = FFCA(256, 128, batch_norm=True)
                self.final_channels = 32
        # 为不同的骨干网络配置适当的通道数
        elif backbone.startswith('densenet'):
            if backbone == 'densenet121':
                self.final_channels = 128  # 从1024压缩到128，保留更多信息
            elif backbone == 'densenet169':
                self.final_channels = 160  # 从1664压缩到160
            elif backbone == 'densenet201':
                self.final_channels = 192  # 从1920压缩到192
            else:
                self.final_channels = 128  # 默认值
        elif backbone.startswith('efficientnet'):
            if backbone == 'efficientnet_b0':
                self.final_channels = 128  # 从1280压缩到128
            elif backbone == 'efficientnet_b3':
                self.final_channels = 152  # 从1536压缩到152
            elif backbone == 'efficientnet_b5':
                self.final_channels = 192  # 从2048压缩到192
            else:
                self.final_channels = 128  # 默认值
        elif backbone.startswith('resnet'):
            if backbone == 'resnet50':
                self.final_channels = 32  # 从2048压缩到128
            elif backbone == 'resnet101':
                self.final_channels = 32  # 从2048压缩到160，但考虑到网络深度更多
            else:
                self.final_channels = 128
        else:
            # 其他backbone使用默认通道数
            self.final_channels = 32

        if self.use_gnn:
            self.gnn_final = VectorizedIPGLayerWithStats(
                in_channels=self.final_channels,
                min_connections=1,
                max_connections=16,
                window_size=9,
                top_percent=5
            )

        # 使用Transformer增强，并支持自定义配置
        if self.use_trans:
            # 设置默认的transformer配置
            default_config = {
                'depth': 4,  # transformer的深度
                'num_heads': 4,  # 注意力头数
                'window_height': 8,  # 窗口高度
                'window_width': 8,  # 窗口宽度
                'shift_size': None,  # 移位大小，默认为窗口大小的一半
                'channel_expansion': 2,  # 通道扩展比例
                'mlp_ratio': 4,  # MLP扩展比例
                'drop_rate': dropout_rate,  # dropout率
                'attn_drop_rate': dropout_rate,  # 注意力dropout率
                'downsample_factor': 4  # 默认4倍降采样
            }

            # 使用用户提供的配置覆盖默认配置
            if trans_config is not None:
                for key, value in trans_config.items():
                    default_config[key] = value

            # 打印配置信息
            print(f"Using Transformer with config:")
            for key, value in default_config.items():
                print(f"  {key}: {value}")

            self.trans_final = TransformerEnhancement(
                in_channels=self.final_channels,
                depth=default_config['depth'],
                num_heads=default_config['num_heads'],
                window_height=default_config['window_height'],
                window_width=default_config['window_width'],
                shift_size=default_config['shift_size'],
                channel_expansion=default_config['channel_expansion'],
                mlp_ratio=default_config['mlp_ratio'],
                drop_rate=default_config['drop_rate'],
                attn_drop_rate=default_config['attn_drop_rate'],
                downsample_factor=default_config['downsample_factor']
            )

        self.dropout = nn.Dropout(dropout_rate)

        # 针对不同大小的通道数调整头部网络
        self.fc_hm = nn.Sequential(
            nn.Conv2d(self.final_channels, head_conv, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(head_conv),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Conv2d(head_conv, 2, kernel_size=final_kernel, stride=1,
                      padding=final_kernel // 2, bias=True)
        )

        self.fc_vec = nn.Sequential(
            nn.Conv2d(self.final_channels, head_conv, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(head_conv),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Conv2d(head_conv, 2, kernel_size=final_kernel, stride=1,
                      padding=final_kernel // 2, bias=True)
        )

        self.peak_radius = 5

        # 打印backbone信息
        print(f"Using backbone: {backbone} with final channels: {self.final_channels}")

    def find_peaks_from_dual_heatmap(self, heatmap, threshold=0.1):
        batch_size, _, height, width = heatmap.size()
        device = heatmap.device

        upper_points = torch.zeros(batch_size, 18, 2, device=device)
        lower_points = torch.zeros(batch_size, 18, 2, device=device)

        for b in range(batch_size):
            upper_hm = heatmap[b, 0].clone()
            upper_points[b] = self.extract_multiple_peaks(upper_hm, num_peaks=18, threshold=threshold)

            lower_hm = heatmap[b, 1].clone()
            lower_points[b] = self.extract_multiple_peaks(lower_hm, num_peaks=18, threshold=threshold)

        return upper_points, lower_points

    def extract_multiple_peaks(self, heatmap, num_peaks=18, threshold=0.1):
        device = heatmap.device
        height, width = heatmap.size()
        peaks = torch.zeros(num_peaks, 2, device=device)

        hm_working = heatmap.clone()

        for i in range(num_peaks):
            max_val, _ = torch.max(hm_working.view(-1), dim=0)
            if max_val <= threshold:
                break

            max_idx = torch.argmax(hm_working.view(-1))
            y = torch.div(max_idx, width, rounding_mode='floor')
            x = max_idx % width

            # 确保坐标在合法范围内
            y = torch.clamp(y, 0, height - 1)
            x = torch.clamp(x, 0, width - 1)

            peaks[i, 0] = x.float()
            peaks[i, 1] = y.float()

            y_min = max(0, y - self.peak_radius)
            y_max = min(height, y + self.peak_radius + 1)
            x_min = max(0, x - self.peak_radius)
            x_max = min(width, x + self.peak_radius + 1)

            hm_working[y_min:y_max, x_min:x_max] = 0

        return peaks

    def order_keypoints_by_y_coordinate(self, points):
        batch_size, num_points, _ = points.shape
        ordered_points = torch.zeros_like(points)

        for b in range(batch_size):
            batch_points = points[b]

            # 检查点是否有效
            valid_mask = (batch_points.sum(dim=1) != 0)
            if valid_mask.sum() == 0:
                continue  # 如果没有有效点，跳过

            # 仅对有效点进行排序
            valid_points = batch_points[valid_mask]
            if len(valid_points) > 0:
                _, indices = torch.sort(valid_points[:, 1], dim=0, descending=True)
                ordered_valid = valid_points[indices]

                # 填充回原始大小的张量
                ordered_points[b, :len(ordered_valid)] = ordered_valid

        return ordered_points

    def forward(self, x):
        feature_dict = {}

        # 基于backbone类型处理输入特征
        if self.is_multi_scale:
            # HRNet多尺度特征处理
            p3_combine = self.dec_c4(x[-1], x[-2])
            p2_combine = self.dec_c3(p3_combine, x[-3])
            p1_combine = self.dec_c2(p2_combine, x[-4])

            enhanced_features = p1_combine
        else:
            # 单尺度backbone直接使用特征
            enhanced_features = x

        # 应用GNN增强（如果启用）
        if self.use_gnn:
            enhanced_features = self.gnn_final(enhanced_features)

        # 应用Transformer增强（如果启用）
        if self.use_trans:
            enhanced_features = self.trans_final(enhanced_features)

        enhanced_features = self.dropout(enhanced_features)

        # 预测热图
        dual_hm = self.fc_hm(enhanced_features)
        dual_hm = torch.sigmoid(dual_hm)

        feature_dict['hm'] = dual_hm

        # 记录特征图尺寸供后续使用
        feature_dict['feature_size'] = (dual_hm.size(2), dual_hm.size(3))

        # 提取峰值点
        upper_points, lower_points = self.find_peaks_from_dual_heatmap(dual_hm, threshold=0.1)
        ordered_upper_points = self.order_keypoints_by_y_coordinate(upper_points)
        ordered_lower_points = self.order_keypoints_by_y_coordinate(lower_points)

        feature_dict['peak_points_upper'] = ordered_upper_points
        feature_dict['peak_points_lower'] = ordered_lower_points
        feature_dict['peak_points'] = (ordered_upper_points + ordered_lower_points) / 2

        # 预测向量场
        feature_dict['vec_ind'] = self.fc_vec(enhanced_features)

        return feature_dict