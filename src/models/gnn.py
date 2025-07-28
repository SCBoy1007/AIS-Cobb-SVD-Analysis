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

        Args:
            x: Input feature map [B, C, H, W]

        Returns:
            df_values: Detail richness values [B, 1, H, W]
            connection_counts: Number of connections for each pixel [B, 1, H, W]
        """
        B, C, H, W = x.shape

        # Downsample and then upsample to get a coarse version of the feature map
        # Using scale factor of 0.5 (downsample by 2) as mentioned in the paper
        x_down = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)
        x_down_up = F.interpolate(x_down, size=(H, W), mode='bilinear', align_corners=False)

        # Calculate the absolute difference and sum across channels
        df_values = torch.abs(x - x_down_up).sum(dim=1, keepdim=True)

        # Normalize the DF values to [0, 1] range for the current batch
        df_min = df_values.view(B, -1).min(dim=1, keepdim=True)[0].unsqueeze(-1).unsqueeze(-1)
        df_max = df_values.view(B, -1).max(dim=1, keepdim=True)[0].unsqueeze(-1).unsqueeze(-1)
        df_normalized = (df_values - df_min) / (df_max - df_min + 1e-8)

        # Map the normalized DF values to connection counts
        # Scale from min_connections to max_connections
        connection_range = self.max_connections - self.min_connections
        connection_counts = self.min_connections + torch.round(df_normalized * connection_range)

        return df_values, connection_counts


class IPGLayer(nn.Module):
    """
    Image Processing GNN Layer that applies graph-based aggregation
    based on dynamic connection counts.
    """

    def __init__(self, in_channels, min_connections=1, max_connections=16, local_window_ratio=0.25):
        """
        Args:
            in_channels: Number of input channels
            min_connections: Minimum number of connections for a pixel (including self)
            max_connections: Maximum number of connections for a pixel (including self)
            local_window_ratio: Local window size as a ratio of the shortest feature map dimension
        """
        super(IPGLayer, self).__init__()
        self.in_channels = in_channels
        self.df_detector = DFDetector(min_connections, max_connections)
        self.local_window_ratio = local_window_ratio

        # Add LayerNorm and a simple convolutional FFN for post-processing
        self.layer_norm = nn.LayerNorm([in_channels, 1, 1])
        self.conv_ffn = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * 2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)
        )

    def forward(self, x):
        """
        Apply IPG graph aggregation to the input feature map.

        Args:
            x: Input feature map [B, C, H, W]

        Returns:
            Enhanced feature map [B, C, H, W]
        """
        B, C, H, W = x.shape

        # Calculate DF values and connection counts
        _, connection_counts = self.df_detector(x)

        # Determine local window size
        shortest_dim = min(H, W)
        local_window_size = max(3, int(shortest_dim * self.local_window_ratio))
        # Make sure it's odd for centering
        if local_window_size % 2 == 0:
            local_window_size += 1

        half_window = local_window_size // 2

        # Reshape input for easier processing
        x_reshaped = x.permute(0, 2, 3, 1).reshape(B, H * W, C)  # [B, H*W, C]

        # Create position indices for all pixels
        y_indices, x_indices = torch.meshgrid(torch.arange(H, device=x.device),
                                              torch.arange(W, device=x.device),
                                              indexing='ij')
        position_indices = torch.stack([y_indices, x_indices], dim=-1).reshape(-1, 2)  # [H*W, 2]

        # Output tensor to store aggregated features
        output = torch.zeros_like(x_reshaped)

        # Process each pixel
        for pixel_idx in range(H * W):
            # Get the position of the current pixel
            y, x = position_indices[pixel_idx]

            # Determine the local window boundaries with padding for edge pixels
            y_min = max(0, y - half_window)
            y_max = min(H - 1, y + half_window)
            x_min = max(0, x - half_window)
            x_max = min(W - 1, x + half_window)

            # Calculate local indices within the window
            local_y, local_x = torch.meshgrid(
                torch.arange(y_min, y_max + 1, device=x.device),
                torch.arange(x_min, x_max + 1, device=x.device),
                indexing='ij'
            )
            local_indices = local_y * W + local_x
            local_indices = local_indices.reshape(-1)

            # Get features of the pixel and its local neighbors
            pixel_feature = x_reshaped[:, pixel_idx:pixel_idx + 1, :]  # [B, 1, C]
            local_features = x_reshaped[:, local_indices, :]  # [B, num_local, C]

            # Calculate cosine similarity
            pixel_feature_norm = F.normalize(pixel_feature, p=2, dim=-1)
            local_features_norm = F.normalize(local_features, p=2, dim=-1)
            similarity = torch.bmm(pixel_feature_norm, local_features_norm.transpose(1, 2))  # [B, 1, num_local]

            # Get the number of connections for this pixel
            k = connection_counts[:, 0, y, x].long()  # [B]

            # For each batch, select top-k similar nodes
            weights = torch.zeros(B, 1, local_indices.size(0), device=x.device)

            for b in range(B):
                k_b = min(k[b].item(), local_indices.size(0))
                # Get top-k similarities and their indices
                top_sim, top_idx = torch.topk(similarity[b, 0], k=k_b, dim=0)
                # Apply exponential to convert similarities to weights
                top_weights = torch.exp(top_sim)
                # Normalize weights
                top_weights = top_weights / torch.sum(top_weights)
                # Place weights at the right indices
                weights[b, 0, top_idx] = top_weights

            # Aggregate features using the weights
            aggregated = torch.bmm(weights, local_features)  # [B, 1, C]
            output[:, pixel_idx:pixel_idx + 1, :] = aggregated

        # Reshape back to original format
        output = output.reshape(B, H, W, C).permute(0, 3, 1, 2)  # [B, C, H, W]

        # Apply layer normalization (properly reshaped)
        x_norm = self.layer_norm(x.reshape(B, C, -1).transpose(1, 2))
        x_norm = x_norm.transpose(1, 2).reshape(B, C, H, W)

        # Apply graph aggregation with residual connection
        enhanced = output + x_norm

        # Apply FFN with another residual connection
        final_output = enhanced + self.conv_ffn(enhanced)

        return final_output


class KNNLayer(nn.Module):
    """
    K-Nearest Neighbor Graph Layer for comparison with IPG.
    Uses a fixed k for all pixels regardless of detail richness.
    """

    def __init__(self, in_channels, k=8, local_window_ratio=0.25):
        """
        Args:
            in_channels: Number of input channels
            k: Fixed number of neighbors for each pixel (including self)
            local_window_ratio: Local window size as a ratio of the shortest feature map dimension
        """
        super(KNNLayer, self).__init__()
        self.in_channels = in_channels
        self.k = k
        self.local_window_ratio = local_window_ratio

        # Add LayerNorm and a simple convolutional FFN for post-processing
        self.layer_norm = nn.LayerNorm([in_channels, 1, 1])
        self.conv_ffn = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * 2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)
        )

    def forward(self, x):
        """
        Apply KNN graph aggregation to the input feature map.

        Args:
            x: Input feature map [B, C, H, W]

        Returns:
            Enhanced feature map [B, C, H, W]
        """
        B, C, H, W = x.shape

        # Determine local window size
        shortest_dim = min(H, W)
        local_window_size = max(3, int(shortest_dim * self.local_window_ratio))
        # Make sure it's odd for centering
        if local_window_size % 2 == 0:
            local_window_size += 1

        half_window = local_window_size // 2

        # Reshape input for easier processing
        x_reshaped = x.permute(0, 2, 3, 1).reshape(B, H * W, C)  # [B, H*W, C]

        # Create position indices for all pixels
        y_indices, x_indices = torch.meshgrid(torch.arange(H, device=x.device),
                                              torch.arange(W, device=x.device),
                                              indexing='ij')
        position_indices = torch.stack([y_indices, x_indices], dim=-1).reshape(-1, 2)  # [H*W, 2]

        # Output tensor to store aggregated features
        output = torch.zeros_like(x_reshaped)

        # Process each pixel
        for pixel_idx in range(H * W):
            # Get the position of the current pixel
            y, x = position_indices[pixel_idx]

            # Determine the local window boundaries with padding for edge pixels
            y_min = max(0, y - half_window)
            y_max = min(H - 1, y + half_window)
            x_min = max(0, x - half_window)
            x_max = min(W - 1, x + half_window)

            # Calculate local indices within the window
            local_y, local_x = torch.meshgrid(
                torch.arange(y_min, y_max + 1, device=x.device),
                torch.arange(x_min, x_max + 1, device=x.device),
                indexing='ij'
            )
            local_indices = local_y * W + local_x
            local_indices = local_indices.reshape(-1)

            # Get features of the pixel and its local neighbors
            pixel_feature = x_reshaped[:, pixel_idx:pixel_idx + 1, :]  # [B, 1, C]
            local_features = x_reshaped[:, local_indices, :]  # [B, num_local, C]

            # Calculate cosine similarity
            pixel_feature_norm = F.normalize(pixel_feature, p=2, dim=-1)
            local_features_norm = F.normalize(local_features, p=2, dim=-1)
            similarity = torch.bmm(pixel_feature_norm, local_features_norm.transpose(1, 2))  # [B, 1, num_local]

            # For each batch, select top-k similar nodes (k is fixed for all pixels)
            weights = torch.zeros(B, 1, local_indices.size(0), device=x.device)

            for b in range(B):
                k_b = min(self.k, local_indices.size(0))
                # Get top-k similarities and their indices
                top_sim, top_idx = torch.topk(similarity[b, 0], k=k_b, dim=0)
                # Apply exponential to convert similarities to weights
                top_weights = torch.exp(top_sim)
                # Normalize weights
                top_weights = top_weights / torch.sum(top_weights)
                # Place weights at the right indices
                weights[b, 0, top_idx] = top_weights

            # Aggregate features using the weights
            aggregated = torch.bmm(weights, local_features)  # [B, 1, C]
            output[:, pixel_idx:pixel_idx + 1, :] = aggregated

        # Reshape back to original format
        output = output.reshape(B, H, W, C).permute(0, 3, 1, 2)  # [B, C, H, W]

        # Apply layer normalization (properly reshaped)
        x_norm = self.layer_norm(x.reshape(B, C, -1).transpose(1, 2))
        x_norm = x_norm.transpose(1, 2).reshape(B, C, H, W)

        # Apply graph aggregation with residual connection
        enhanced = output + x_norm

        # Apply FFN with another residual connection
        final_output = enhanced + self.conv_ffn(enhanced)

        return final_output


# More efficient implementation for practical use
class EfficientIPGLayer(nn.Module):
    """
    More efficient implementation of IPG layer using unfold operation
    instead of explicit loop over all pixels.
    """

    def __init__(self, in_channels, min_connections=1, max_connections=16, local_window_ratio=0.25):
        """
        Args:
            in_channels: Number of input channels
            min_connections: Minimum number of connections for a pixel (including self)
            max_connections: Maximum number of connections for a pixel (including self)
            local_window_ratio: Local window size as a ratio of the shortest feature map dimension
        """
        super(EfficientIPGLayer, self).__init__()
        self.in_channels = in_channels
        self.df_detector = DFDetector(min_connections, max_connections)
        self.local_window_ratio = local_window_ratio

        # Add GroupNorm and a simple convolutional FFN for post-processing
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
        Apply IPG graph aggregation to the input feature map using unfold for efficiency.

        Args:
            x: Input feature map [B, C, H, W]

        Returns:
            Enhanced feature map [B, C, H, W]
        """
        B, C, H, W = x.shape

        # Calculate DF values and connection counts
        _, connection_counts = self.df_detector(x)

        # Determine local window size
        shortest_dim = min(H, W)
        local_window_size = max(3, int(shortest_dim * self.local_window_ratio))
        # Make sure it's odd for centering
        if local_window_size % 2 == 0:
            local_window_size += 1

        # Use unfold to extract local neighborhoods for all pixels at once
        # This is much more efficient than the loop in the basic implementation
        padded_x = F.pad(x, [local_window_size // 2] * 4, mode='reflect')
        patches = F.unfold(padded_x, kernel_size=local_window_size, stride=1)
        # patches: [B, C*local_window_size*local_window_size, H*W]

        patches = patches.reshape(B, C, local_window_size ** 2, H * W)
        patches = patches.permute(0, 3, 1, 2)  # [B, H*W, C, local_window_size**2]

        # Reshape input for easier processing
        x_flat = x.view(B, C, H * W).permute(0, 2, 1)  # [B, H*W, C]

        # Normalize features for cosine similarity
        x_norm = F.normalize(x_flat, p=2, dim=-1)
        patches_norm = F.normalize(patches, p=2, dim=2)

        # Calculate similarities between each pixel and its local neighborhood
        # Reshape for batch matrix multiplication
        x_norm = x_norm.unsqueeze(-1)  # [B, H*W, C, 1]
        similarities = torch.matmul(patches_norm, x_norm).squeeze(-1)  # [B, H*W, local_window_size**2]

        # Initialize output tensor
        output = torch.zeros_like(x_flat)

        # Process each pixel position
        for pos in range(H * W):
            for b in range(B):
                # Get number of connections for this pixel
                k = min(connection_counts[b, 0, pos // W, pos % W].long().item(), local_window_size ** 2)

                # Get top-k similarities
                top_sim, top_idx = torch.topk(similarities[b, pos], k=k, dim=0)

                # Calculate weights
                weights = torch.exp(top_sim)
                weights = weights / weights.sum()

                # Extract selected neighbors
                selected_features = patches[b, pos, :, top_idx]  # [C, k]

                # Weighted sum
                output[b, pos] = torch.matmul(selected_features, weights)

        # Reshape back
        output = output.permute(0, 2, 1).reshape(B, C, H, W)

        # Apply normalization with residual connection
        norm_x = self.norm(x)
        enhanced = output + norm_x

        # Apply FFN with another residual connection
        final_output = enhanced + self.conv_ffn(enhanced)

        return final_output