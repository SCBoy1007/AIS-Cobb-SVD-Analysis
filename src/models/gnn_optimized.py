import torch
import torch.nn as nn
import torch.nn.functional as F


class DFDetector(nn.Module):
    """
    Detail-Aware Metric (DF) Detector that determines the connection count for each pixel
    based on its detail richness.
    """

    def __init__(self, min_connections=1, max_connections=16):
        """
        Args:
            min_connections: Minimum number of connections for a pixel (including self)
            max_connections: Maximum number of connections for a pixel (including self)
        """
        super(DFDetector, self).__init__()
        self.min_connections = min_connections
        self.max_connections = max_connections

    def forward(self, x):
        """
        Calculate the detail richness of each pixel in the feature map.
        Uses a non-linear mapping to ensure most pixels only connect with themselves.

        Args:
            x: Input feature map [B, C, H, W]

        Returns:
            df_values: Detail richness values [B, 1, H, W]
            connection_counts: Number of connections for each pixel [B, 1, H, W]
        """
        B, C, H, W = x.shape

        # Downsample and then upsample to get a coarse version of the feature map
        x_down = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)
        x_down_up = F.interpolate(x_down, size=(H, W), mode='bilinear', align_corners=False)

        # Calculate the absolute difference and sum across channels
        df_values = torch.abs(x - x_down_up).sum(dim=1, keepdim=True)

        # Normalize the DF values to [0, 1] range for the current batch
        df_min = df_values.view(B, -1).min(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
        df_max = df_values.view(B, -1).max(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
        df_normalized = (df_values - df_min) / (df_max - df_min + 1e-8)

        # Apply power function to make distribution non-linear
        df_power = torch.pow(df_normalized, 4)

        # Threshold to make most pixels have just 1 connection (self)
        threshold = 0.9
        mask = (df_power > threshold).float()

        # Map the powered DF values to connection counts
        connection_range = self.max_connections - self.min_connections
        connections_above_min = torch.round(((df_power - threshold) / (1 - threshold + 1e-8)) * connection_range)
        connections_above_min = torch.clamp(connections_above_min, min=0) * mask

        connection_counts = self.min_connections + connections_above_min

        return df_values, connection_counts


class FastIPGLayer(nn.Module):
    """
    Fast implementation of IPG layer using vectorized operations
    """

    def __init__(self, in_channels, min_connections=1, max_connections=16, window_size=3):
        """
        Args:
            in_channels: Number of input channels
            min_connections: Minimum number of connections for a pixel
            max_connections: Maximum number of connections for a pixel
            window_size: Fixed window size for local aggregation (3 or 5 recommended)
        """
        super(FastIPGLayer, self).__init__()
        self.in_channels = in_channels
        self.window_size = window_size
        self.df_detector = DFDetector(min_connections, max_connections)

        # Ensure num_groups divides in_channels
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

    def forward(self, x):
        """
        Fully vectorized implementation of IPG layer
        """
        B, C, H, W = x.shape

        # Padding for unfold
        padding = self.window_size // 2

        # Get connection counts
        _, connection_counts = self.df_detector(x)

        # Fast implementation using unfold to extract all patches at once
        padded_x = F.pad(x, [padding, padding, padding, padding], mode='reflect')

        # Using unfold to extract patches: [B, C, H*W, window_size*window_size]
        patches = F.unfold(padded_x, kernel_size=self.window_size, padding=0, stride=1)
        patches = patches.view(B, C, self.window_size ** 2, H * W)

        # Transpose to get shape [B, C, H*W, window_size*window_size]
        patches = patches.permute(0, 1, 3, 2)

        # Reshape x for similarity calculation
        x_flat = x.view(B, C, H * W)

        # Normalize features for similarity calculation
        patches_norm = F.normalize(patches, p=2, dim=1)
        x_flat_norm = F.normalize(x_flat, p=2, dim=1)

        # Calculate similarity for all patches at once
        # [B, H*W, window_size*window_size]
        similarity = torch.matmul(x_flat_norm.transpose(1, 2), patches_norm)

        # Create a mask for each pixel based on its connection count
        k_values = connection_counts.view(B, 1, H * W).to(torch.int64)
        k_values = torch.clamp(k_values, max=self.window_size ** 2)

        # Initialize output
        output = torch.zeros_like(x_flat)

        # Process each batch (still need this loop but we process all pixels at once)
        for b in range(B):
            # For each position in the batch
            for pos in range(H * W):
                # Get k for current position
                k = k_values[b, 0, pos]

                # Get top-k similar patches
                sim = similarity[b, pos]
                top_sim, top_idx = torch.topk(sim, k=k)

                # Calculate weights
                weights = torch.exp(top_sim)
                weights = weights / (weights.sum() + 1e-8)

                # Get selected patches for this position
                selected_patches = patches[b, :, pos, top_idx]  # [C, k]

                # Calculate weighted sum
                weighted_sum = torch.matmul(selected_patches, weights)

                # Update output
                output[b, :, pos] = weighted_sum

        # Reshape output back to [B, C, H, W]
        output = output.view(B, C, H, W)

        # Apply normalization with residual connection
        norm_x = self.norm(x)
        enhanced = output + norm_x

        # Apply FFN with another residual connection
        final_output = enhanced + self.conv_ffn(enhanced)

        return final_output


class OptimizedIPGLayer(nn.Module):
    """
    Fully optimized implementation of IPG layer using full batch processing
    """

    def __init__(self, in_channels, min_connections=1, max_connections=16, window_size=3):
        """
        Args:
            in_channels: Number of input channels
            min_connections: Minimum number of connections for a pixel
            max_connections: Maximum number of connections for a pixel
            window_size: Fixed window size for local aggregation (3 or 5 recommended)
        """
        super(OptimizedIPGLayer, self).__init__()
        self.in_channels = in_channels
        self.window_size = window_size
        self.df_detector = DFDetector(min_connections, max_connections)

        # Ensure num_groups divides in_channels
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

    def forward(self, x):
        """
        Fully vectorized and batch-processed implementation of IPG layer
        """
        B, C, H, W = x.shape

        # Calculate DF values and connection counts
        _, connection_counts = self.df_detector(x)

        # Padding for unfold
        padding = self.window_size // 2

        # Extract all local patches at once
        patches = F.unfold(x, kernel_size=self.window_size, padding=padding)
        patches = patches.view(B, C, self.window_size ** 2, H * W)

        # Get center features
        x_flat = x.view(B, C, H * W)

        # Normalize features for similarity calculation (across channel dimension)
        x_flat_norm = F.normalize(x_flat, p=2, dim=1)  # [B, C, H*W]

        # Create a mask for valid similarity values
        # We'll create batched top-k masks

        # Initialize output tensor
        output = torch.zeros_like(x)

        # Process in batches to avoid OOM
        batch_size = 1000  # Process 1000 pixels at a time

        for start_idx in range(0, H * W, batch_size):
            end_idx = min(start_idx + batch_size, H * W)
            current_batch_size = end_idx - start_idx

            # Get patches for current batch [B, C, window_size*window_size, batch_size]
            batch_patches = patches[:, :, :, start_idx:end_idx]

            # Normalize patches [B, C, window_size*window_size, batch_size]
            batch_patches_norm = F.normalize(batch_patches, p=2, dim=1)

            # Get center features for current batch [B, C, batch_size]
            batch_centers = x_flat[:, :, start_idx:end_idx]
            batch_centers_norm = x_flat_norm[:, :, start_idx:end_idx]

            # Compute similarity [B, batch_size, window_size*window_size]
            # For each center feature, compute similarity with all its surrounding patches
            sim_list = []
            for b in range(B):
                # [batch_size, C] @ [C, window_size*window_size, batch_size] -> [batch_size, window_size*window_size]
                sim = torch.bmm(
                    batch_centers_norm[b].transpose(0, 1),  # [batch_size, C]
                    batch_patches_norm[b].view(C, self.window_size ** 2, current_batch_size)
                    # [C, window_size**2, batch_size]
                )
                sim_list.append(sim)

            # Stack similarities [B, batch_size, window_size*window_size]
            similarity = torch.stack(sim_list, dim=0)

            # Get connection counts for current batch [B, 1, batch_size]
            batch_k = connection_counts.view(B, 1, H * W)[:, :, start_idx:end_idx].to(torch.int64)
            batch_k = torch.clamp(batch_k, max=self.window_size ** 2)

            # Process each item in batch
            for b in range(B):
                for i in range(current_batch_size):
                    pos_idx = start_idx + i
                    h, w = pos_idx // W, pos_idx % W

                    # Get connection count for this pixel
                    k = batch_k[b, 0, i]

                    # Get similarity for this pixel
                    sim = similarity[b, i]  # [window_size*window_size]

                    # Get top-k similar patches
                    top_sim, top_idx = torch.topk(sim, k=k)

                    # Calculate weights
                    weights = torch.exp(top_sim)
                    weights = weights / (weights.sum() + 1e-8)

                    # Get selected patches
                    selected_patches = batch_patches[b, :, :, i]  # [C, window_size*window_size]
                    selected_patches = selected_patches[:, top_idx]  # [C, k]

                    # Calculate weighted sum
                    weighted_sum = torch.matmul(selected_patches, weights)  # [C]

                    # Update output
                    output[b, :, h, w] = weighted_sum

        # Apply normalization with residual connection
        norm_x = self.norm(x)
        enhanced = output + norm_x

        # Apply FFN with another residual connection
        final_output = enhanced + self.conv_ffn(enhanced)

        return final_output


class FullVectorizedIPGLayer(nn.Module):
    """
    最彻底的向量化实现，完全避免Python循环
    """

    def __init__(self, in_channels, min_connections=1, max_connections=16, window_size=3):
        super(FullVectorizedIPGLayer, self).__init__()
        self.in_channels = in_channels
        self.window_size = window_size
        self.min_connections = min_connections
        self.max_connections = max_connections

        # 使用标准卷积代替GNN的相似性计算
        self.attention_conv = nn.Conv2d(in_channels, in_channels, kernel_size=window_size,
                                        padding=window_size // 2, groups=1, bias=False)

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

        # 添加一个额外的1x1卷积层，作为特征调整器
        self.feature_adapter = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        identity = x

        # 标准化输入特征
        x_norm = F.normalize(x, p=2, dim=1)

        # 使用卷积作为局部注意力机制
        attention_weights = self.attention_conv(x_norm)
        attention_weights = torch.sigmoid(attention_weights)

        # 应用注意力权重
        enhanced = x * attention_weights

        # 使用1x1卷积作为特征调整
        enhanced = self.feature_adapter(enhanced)

        # 应用归一化
        norm_x = self.norm(identity)
        enhanced = enhanced + norm_x

        # 应用FFN与残差连接
        final_output = enhanced + self.conv_ffn(enhanced)

        return final_output