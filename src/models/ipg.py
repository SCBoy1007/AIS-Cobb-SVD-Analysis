import torch
import torch.nn as nn
import torch.nn.functional as F


class DFDetector(nn.Module):
    """改进版DF值检测器 - 更合理的参数设置"""

    def __init__(self, min_connections=1, max_connections=16, threshold=0.1, power=2):
        super(DFDetector, self).__init__()
        self.min_connections = min_connections
        self.max_connections = max_connections
        self.threshold = threshold  # 降低阈值到0.5
        self.power = power  # 使用较低的幂次，默认为2

    def forward(self, x):
        B, C, H, W = x.shape

        # 下采样然后上采样以获取特征图的粗糙版本
        x_down = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)
        x_down_up = F.interpolate(x_down, size=(H, W), mode='bilinear', align_corners=False)

        # 计算绝对差异并在通道维度上求和
        df_values = torch.abs(x - x_down_up).sum(dim=1, keepdim=True)

        # 对当前批次中的DF值进行归一化为[0, 1]范围
        df_min = df_values.view(B, -1).min(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
        df_max = df_values.view(B, -1).max(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
        df_normalized = (df_values - df_min) / (df_max - df_min + 1e-8)

        # 使用较低的幂次，使分布不那么极端
        df_power = torch.pow(df_normalized, self.power)

        # 使用较低的阈值，让更多像素超过阈值
        mask = (df_power > self.threshold).float()

        # 将DF值映射到连接数，使用平滑映射以获得更多变化
        # 映射公式调整：将阈值以上的值更均匀地映射到连接范围
        connection_range = self.max_connections - self.min_connections

        # 使用线性映射而不是基于阈值的映射
        connections_above_min = torch.round(df_power * connection_range)
        connections_above_min = torch.clamp(connections_above_min, min=0)

        # 将mask应用到超过阈值的像素
        # 阈值下的像素只有min_connections，阈值上的像素有更多连接
        connections = self.min_connections + connections_above_min * mask

        return df_values, connections


class VectorizedIPGLayer(nn.Module):
    """带连接数统计的向量化IPG层"""

    def __init__(self, in_channels, min_connections=1, max_connections=16, window_size=3, threshold=0.9):
        super(VectorizedIPGLayer, self).__init__()
        self.in_channels = in_channels
        self.window_size = window_size
        self.min_connections = min_connections
        self.max_connections = max_connections
        self.threshold = threshold

        # 确保num_groups能整除in_channels
        num_groups = 1
        for i in range(min(32, in_channels), 0, -1):
            if in_channels % i == 0:
                num_groups = i
                break

        self.norm = nn.GroupNorm(num_groups, in_channels)
        self.conv_ffn = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * 2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)
        )

        # DF检测器
        self.df_detector = DFDetector(min_connections, max_connections, threshold)

        # 添加统计信息标志
        self.print_stats = True
        self.stats_interval = 1  # 每10次迭代打印一次统计信息
        self.iter_count = 0

    def forward(self, x):
        """修复后的向量化IPG前向传播"""
        B, C, H, W = x.shape
        identity = x
        device = x.device
        self.iter_count += 1

        # 计算DF值和连接数
        _, connection_counts = self.df_detector(x)
        k_max = min(self.max_connections, self.window_size ** 2)

        # 收集连接数统计信息
        if self.print_stats and (self.iter_count % self.stats_interval == 0):
            # 转为整数类型以便统计
            counts = connection_counts.long().view(-1)

            # 统计每个连接数的数量
            stats = {}
            total_pixels = counts.numel()

            for k in range(self.min_connections, self.max_connections + 1):
                num_pixels = (counts == k).sum().item()
                percentage = 100.0 * num_pixels / total_pixels
                stats[k] = (num_pixels, percentage)

            # 打印统计信息
            print("\n======= IPG连接数统计 =======")
            print(f"特征图大小: {H}x{W}, 总像素数: {total_pixels}")
            print(f"窗口大小: {self.window_size}x{self.window_size}, 最大连接数: {self.max_connections}")
            print("连接数 | 像素数 | 百分比")

            for k in range(self.min_connections, self.max_connections + 1):
                if k in stats:
                    num, pct = stats[k]
                    print(f"   {k:2d}  | {num:7d} | {pct:6.2f}%")

            print("================================\n")

        # 使用固定窗口大小
        ksize = self.window_size
        padding = ksize // 2

        # 使用unfold提取局部区域 [B, C*ksize*ksize, H*W]
        patches = F.unfold(x, kernel_size=ksize, padding=padding)
        patches = patches.view(B, C, ksize * ksize, H * W)

        # 准备输出 - 默认直接复制输入，对于k=1的情况
        output = x.clone()

        # 将H*W展平为一维处理
        x_flat = x.view(B, C, H * W)

        # 对每个批次单独处理
        for b in range(B):
            # 获取当前批次的特征和连接数
            curr_features = x_flat[b]  # [C, H*W]
            curr_patches = patches[b]  # [C, ksize*ksize, H*W]
            curr_counts = connection_counts[b].view(-1)  # [H*W]

            # 找出k>1的像素位置，只处理这些像素
            valid_indices = (curr_counts > 1).nonzero()

            # 检查是否有有效位置
            if valid_indices.numel() == 0:
                continue  # 如果没有k>1的像素，直接跳过

            # 展平索引张量为一维
            valid_mask = valid_indices.view(-1)

            # 只提取k>1的像素的特征和连接数
            valid_features = curr_features[:, valid_mask]  # [C, num_valid]
            valid_patches = curr_patches[:, :, valid_mask]  # [C, ksize*ksize, num_valid]
            valid_counts = curr_counts[valid_mask]  # [num_valid]

            # 检查有效像素数量
            num_valid = valid_mask.size(0)
            if num_valid == 0:
                continue

            # 归一化特征用于相似度计算
            valid_features_norm = F.normalize(valid_features, p=2, dim=0)  # [C, num_valid]
            valid_patches_norm = F.normalize(valid_patches, p=2, dim=0)  # [C, ksize*ksize, num_valid]

            # 计算所有有效像素的相似度矩阵
            # 这一步需要循环批处理来避免内存不足
            batch_size = 1000  # 每批处理1000个像素

            for start_idx in range(0, num_valid, batch_size):
                end_idx = min(start_idx + batch_size, num_valid)

                # 获取当前批次的索引
                batch_indices = valid_mask[start_idx:end_idx]

                # 获取当前批次的特征和连接数
                batch_features = valid_features[:, start_idx:end_idx]  # [C, batch_size]
                batch_patches = valid_patches[:, :, start_idx:end_idx]  # [C, ksize*ksize, batch_size]
                batch_counts = valid_counts[start_idx:end_idx]  # [batch_size]

                curr_batch_size = end_idx - start_idx
                if curr_batch_size == 0:
                    continue

                # 计算当前批次所有像素的相似度得分
                batch_sim_list = []
                for i in range(curr_batch_size):
                    # 获取当前像素的特征和patches
                    center_feat = batch_features[:, i:i + 1]  # [C, 1]
                    local_patches = batch_patches[:, :, i]  # [C, ksize*ksize]

                    # 归一化特征
                    center_norm = F.normalize(center_feat, p=2, dim=0)  # [C, 1]
                    patches_norm = F.normalize(local_patches, p=2, dim=0)  # [C, ksize*ksize]

                    # 计算相似度
                    sim = torch.mm(center_norm.t(), patches_norm)  # [1, ksize*ksize]
                    batch_sim_list.append(sim)

                # 为每个像素获取不同数量的top-k
                # 创建结果存储器
                weighted_sums = torch.zeros((curr_batch_size, C), device=device)

                # 逐像素处理
                for i in range(curr_batch_size):
                    # 获取连接数，确保不超过窗口大小
                    k = min(int(batch_counts[i].item()), k_max)

                    # 获取相似度
                    sim = batch_sim_list[i][0]  # [ksize*ksize]

                    # 获取Top-K相似度和索引
                    top_sim, top_idx = torch.topk(sim, k=k)

                    # 计算权重
                    weights = torch.exp(top_sim)
                    weights = weights / weights.sum()

                    # 获取选中的patches并计算加权和
                    selected_patches = batch_patches[:, top_idx, i]  # [C, k]
                    weighted_sum = torch.matmul(selected_patches, weights)  # [C]

                    # 存储结果
                    weighted_sums[i] = weighted_sum

                # 更新输出
                # 计算原始坐标
                h_indices = torch.div(batch_indices, W, rounding_mode='floor')
                w_indices = batch_indices % W

                # 更新输出张量
                output[b, :, h_indices, w_indices] = weighted_sums.t()

        # 应用归一化和残差连接
        norm_x = self.norm(identity)
        enhanced = output + norm_x

        # 应用FFN和另一个残差连接
        final_output = enhanced + self.conv_ffn(enhanced)

        return final_output


class PercentileBasedDFDetector(nn.Module):
    """
    基于百分比的DF检测器 - 确保固定百分比的像素获得多个连接
    """

    def __init__(self, min_connections=1, max_connections=16, top_percent=10):
        super(PercentileBasedDFDetector, self).__init__()
        self.min_connections = min_connections
        self.max_connections = max_connections
        self.top_percent = top_percent  # 前N%的像素获得多个连接

    def forward(self, x):
        B, C, H, W = x.shape

        # 下采样然后上采样以获取特征图的粗糙版本
        x_down = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)
        x_down_up = F.interpolate(x_down, size=(H, W), mode='bilinear', align_corners=False)

        # 计算绝对差异并在通道维度上求和
        df_values = torch.abs(x - x_down_up).sum(dim=1, keepdim=True)

        # 准备输出连接数
        connection_counts = torch.ones_like(df_values, dtype=torch.long) * self.min_connections

        # 对每个样本单独处理
        for b in range(B):
            # 获取当前样本的DF值并展平
            curr_df = df_values[b, 0].view(-1)

            # 计算阈值点（前top_percent%的像素）
            num_pixels = curr_df.numel()
            num_top_pixels = int(num_pixels * self.top_percent / 100)

            if num_top_pixels > 0:
                # 找出前top_percent%的像素
                _, top_indices = torch.topk(curr_df, num_top_pixels)

                # 为这些像素分配连接数
                # 创建二次函数映射：排名靠前的获得更多连接
                connection_range = self.max_connections - self.min_connections

                # 归一化排名到[0,1]范围
                ranks = torch.arange(num_top_pixels, device=curr_df.device, dtype=torch.float32) / num_top_pixels
                ranks = 1.0 - ranks  # 反转排名，使最高DF值得到最多连接

                # 使用二次函数映射替代线性映射: y = x²
                # 将ranks平方以创建二次曲线
                ranks_squared = ranks * ranks

                # 映射到连接数
                additional_connections = torch.round(ranks_squared * connection_range)
                connection_values = self.min_connections + additional_connections

                # 更新连接数
                flat_connections = connection_counts[b, 0].view(-1)
                flat_connections[top_indices] = connection_values.long()
                connection_counts[b, 0] = flat_connections.view(H, W)

        return df_values, connection_counts


class VectorizedIPGLayerWithStats(nn.Module):
    """带连接数统计的向量化IPG层，使用百分比检测器"""

    def __init__(self, in_channels, min_connections=1, max_connections=16, window_size=7, top_percent=10):
        super(VectorizedIPGLayerWithStats, self).__init__()
        self.in_channels = in_channels
        self.window_size = window_size
        self.min_connections = min_connections
        self.max_connections = max_connections
        self.top_percent = top_percent

        # 确保num_groups能整除in_channels
        num_groups = 1
        for i in range(min(32, in_channels), 0, -1):
            if in_channels % i == 0:
                num_groups = i
                break

        self.norm = nn.GroupNorm(num_groups, in_channels)
        self.conv_ffn = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * 2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)
        )

        # 使用百分比DF检测器
        self.df_detector = PercentileBasedDFDetector(
            min_connections,
            max_connections,
            top_percent
        )

        # 添加统计信息标志
        self.print_stats = False
        self.stats_interval = 1  # 每1次迭代打印一次统计信息
        self.iter_count = 0

    def forward(self, x):
        """带统计信息的IPG前向传播"""
        B, C, H, W = x.shape
        identity = x
        device = x.device
        self.iter_count += 1

        # 计算DF值和连接数
        _, connection_counts = self.df_detector(x)
        k_max = min(self.max_connections, self.window_size ** 2)

        # 收集连接数统计信息
        if self.print_stats and (self.iter_count % self.stats_interval == 0):
            # 转为整数类型以便统计
            counts = connection_counts.long().view(-1)

            # 统计每个连接数的数量
            stats = {}
            total_pixels = counts.numel()

            for k in range(self.min_connections, self.max_connections + 1):
                num_pixels = (counts == k).sum().item()
                percentage = 100.0 * num_pixels / total_pixels
                stats[k] = (num_pixels, percentage)

            # 打印统计信息
            print("\n======= IPG连接数统计 =======")
            print(f"特征图大小: {H}x{W}, 总像素数: {total_pixels}")
            print(f"窗口大小: {self.window_size}x{self.window_size}, 最大连接数: {self.max_connections}")
            print("连接数 | 像素数 | 百分比")

            for k in range(self.min_connections, self.max_connections + 1):
                if k in stats:
                    num, pct = stats[k]
                    print(f"   {k:2d}  | {num:7d} | {pct:6.2f}%")

            print("================================\n")

        # 使用固定窗口大小
        ksize = self.window_size
        padding = ksize // 2

        # 使用unfold提取局部区域 [B, C*ksize*ksize, H*W]
        patches = F.unfold(x, kernel_size=ksize, padding=padding)
        patches = patches.view(B, C, ksize * ksize, H * W)

        # 准备输出 - 默认直接复制输入，对于k=1的情况
        output = x.clone()

        # 将H*W展平为一维处理
        x_flat = x.view(B, C, H * W)

        # 对每个批次单独处理
        for b in range(B):
            # 获取当前批次的特征和连接数
            curr_features = x_flat[b]  # [C, H*W]
            curr_patches = patches[b]  # [C, ksize*ksize, H*W]
            curr_counts = connection_counts[b].view(-1)  # [H*W]

            # 找出k>1的像素位置，只处理这些像素
            valid_indices = (curr_counts > 1).nonzero()

            # 检查是否有有效位置
            if valid_indices.numel() == 0:
                continue  # 如果没有k>1的像素，直接跳过

            # 展平索引张量为一维
            valid_mask = valid_indices.view(-1)

            # 只提取k>1的像素的特征和连接数
            valid_features = curr_features[:, valid_mask]  # [C, num_valid]
            valid_patches = curr_patches[:, :, valid_mask]  # [C, ksize*ksize, num_valid]
            valid_counts = curr_counts[valid_mask]  # [num_valid]

            # 检查有效像素数量
            num_valid = valid_mask.size(0)
            if num_valid == 0:
                continue

            # 归一化特征用于相似度计算
            valid_features_norm = F.normalize(valid_features, p=2, dim=0)  # [C, num_valid]
            valid_patches_norm = F.normalize(valid_patches, p=2, dim=0)  # [C, ksize*ksize, num_valid]

            # 计算所有有效像素的相似度矩阵
            # 这一步需要循环批处理来避免内存不足
            batch_size = 10000  # 每批处理1000个像素

            for start_idx in range(0, num_valid, batch_size):
                end_idx = min(start_idx + batch_size, num_valid)

                # 获取当前批次的索引
                batch_indices = valid_mask[start_idx:end_idx]

                # 获取当前批次的特征和连接数
                batch_features = valid_features[:, start_idx:end_idx]  # [C, batch_size]
                batch_patches = valid_patches[:, :, start_idx:end_idx]  # [C, ksize*ksize, batch_size]
                batch_counts = valid_counts[start_idx:end_idx]  # [batch_size]

                curr_batch_size = end_idx - start_idx
                if curr_batch_size == 0:
                    continue

                # 计算当前批次所有像素的相似度得分
                batch_sim_list = []
                for i in range(curr_batch_size):
                    # 获取当前像素的特征和patches
                    center_feat = batch_features[:, i:i + 1]  # [C, 1]
                    local_patches = batch_patches[:, :, i]  # [C, ksize*ksize]

                    # 归一化特征
                    center_norm = F.normalize(center_feat, p=2, dim=0)  # [C, 1]
                    patches_norm = F.normalize(local_patches, p=2, dim=0)  # [C, ksize*ksize]

                    # 计算相似度
                    sim = torch.mm(center_norm.t(), patches_norm)  # [1, ksize*ksize]
                    batch_sim_list.append(sim)

                # 为每个像素获取不同数量的top-k
                # 创建结果存储器
                weighted_sums = torch.zeros((curr_batch_size, C), device=device)

                # 逐像素处理
                for i in range(curr_batch_size):
                    # 获取连接数，确保不超过窗口大小
                    k = min(int(batch_counts[i].item()), k_max)

                    # 获取相似度
                    sim = batch_sim_list[i][0]  # [ksize*ksize]

                    # 获取Top-K相似度和索引
                    top_sim, top_idx = torch.topk(sim, k=k)

                    # 计算权重
                    weights = torch.exp(top_sim)
                    weights = weights / weights.sum()

                    # 获取选中的patches并计算加权和
                    selected_patches = batch_patches[:, top_idx, i]  # [C, k]
                    weighted_sum = torch.matmul(selected_patches, weights)  # [C]

                    # 存储结果
                    weighted_sums[i] = weighted_sum

                # 更新输出
                # 计算原始坐标
                h_indices = torch.div(batch_indices, W, rounding_mode='floor')
                w_indices = batch_indices % W

                # 更新输出张量
                output[b, :, h_indices, w_indices] = weighted_sums.t()

        # 应用归一化和残差连接
        norm_x = self.norm(identity)
        enhanced = output + norm_x

        # 应用FFN和另一个残差连接
        final_output = enhanced + self.conv_ffn(enhanced)

        return final_output



class HybridIPGLayer(nn.Module):
    """混合IPG层：使用卷积近似相似度计算，但保留DF的注意力机制"""

    def __init__(self, in_channels, min_connections=1, max_connections=16, window_size=3, threshold=0.9):
        super(HybridIPGLayer, self).__init__()
        self.in_channels = in_channels
        self.window_size = window_size
        self.min_connections = min_connections
        self.max_connections = max_connections
        self.threshold = threshold

        # 确保num_groups能整除in_channels
        num_groups = 1
        for i in range(min(32, in_channels), 0, -1):
            if in_channels % i == 0:
                num_groups = i
                break

        self.norm = nn.GroupNorm(num_groups, in_channels)
        self.conv_ffn = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * 2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)
        )

        # DF检测器
        self.df_detector = DFDetector(min_connections, max_connections, threshold)

        # 为每种可能的连接数创建一个卷积层
        self.k_convs = nn.ModuleList()
        for k in range(min_connections, max_connections + 1):
            # 对每个连接数使用一个卷积层
            self.k_convs.append(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=window_size,
                    padding=window_size // 2,
                    bias=False
                )
            )

    def forward(self, x):
        """混合IPG前向传播 - 完全无循环"""
        B, C, H, W = x.shape
        identity = x

        # 计算DF值和连接计数
        df_values, connection_counts = self.df_detector(x)
        connection_counts = connection_counts.long()

        # 归一化特征以便相似度计算
        x_norm = F.normalize(x, p=2, dim=1)

        # 准备输出
        output = torch.zeros_like(x)

        # 使用不同的卷积层处理不同连接数的像素
        for k in range(self.min_connections, self.max_connections + 1):
            # 创建掩码，标识连接数为k的像素
            mask = (connection_counts == k).float()

            if mask.sum() == 0:
                continue  # 没有连接数为k的像素，跳过

            # 使用第(k-min_connections)个卷积层处理
            k_output = self.k_convs[k - self.min_connections](x_norm)

            # 只保留连接数为k的像素的输出
            output = output + k_output * mask

        # 应用归一化和残差连接
        norm_x = self.norm(identity)
        enhanced = output + norm_x

        # 应用FFN和另一个残差连接
        final_output = enhanced + self.conv_ffn(enhanced)

        return final_output


class SimpleIPGLayer(nn.Module):
    """
    简化版IPG层 - 完全无循环
    """

    def __init__(self, in_channels, min_connections=1, max_connections=16, window_size=3, threshold=0.9):
        super(SimpleIPGLayer, self).__init__()
        self.in_channels = in_channels
        self.window_size = window_size
        self.threshold = threshold

        # 确保num_groups能整除in_channels
        num_groups = 1
        for i in range(min(32, in_channels), 0, -1):
            if in_channels % i == 0:
                num_groups = i
                break

        self.norm = nn.GroupNorm(num_groups, in_channels)
        self.conv_ffn = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * 2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)
        )

        # DF检测器
        self.df_detector = DFDetector(min_connections, max_connections, threshold)

        # 主要卷积层，用于特征聚合
        self.conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=window_size,
            padding=window_size // 2,
            bias=False
        )

        # 用于高细节区域的卷积层，更大的感受野
        self.detail_conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=window_size + 2,
            padding=(window_size + 2) // 2,
            bias=False
        )

    def forward(self, x):
        """完全无循环的IPG实现"""
        B, C, H, W = x.shape
        identity = x

        # 计算DF值和连接数
        df_values, connection_counts = self.df_detector(x)

        # 归一化输入特征
        x_norm = F.normalize(x, p=2, dim=1)

        # 基础特征聚合 - 使用卷积
        basic_output = self.conv(x_norm)

        # 高细节区域的增强聚合 - 使用更大卷积核
        detail_output = self.detail_conv(x_norm)

        # 创建高细节区域的掩码
        detail_mask = (connection_counts > 1).float()

        # 混合基础聚合和细节聚合
        # 根据DF值决定使用哪种聚合结果
        output = basic_output * (1 - detail_mask) + detail_output * detail_mask

        # 应用注意力权重 - 使用sigmoid(DF)作为权重
        attention = torch.sigmoid(df_values)
        output = x * (1 - attention) + output * attention

        # 应用归一化和残差连接
        norm_x = self.norm(identity)
        enhanced = output + norm_x

        # 应用FFN和另一个残差连接
        final_output = enhanced + self.conv_ffn(enhanced)

        return final_output