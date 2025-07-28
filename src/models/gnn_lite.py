import torch
import torch.nn as nn
import torch.nn.functional as F


class DFDetector(nn.Module):
    """
    Simplified Detail-Aware Metric (DF) Detector
    """

    def __init__(self, min_connections=1, max_connections=8):
        super(DFDetector, self).__init__()
        self.min_connections = min_connections
        self.max_connections = max_connections
        self.conv = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        # Initialize as Laplacian filter for edge detection
        self.conv.weight.data.fill_(0)
        self.conv.weight.data[0, 0, 1, 1] = 4
        self.conv.weight.data[0, 0, 0, 1] = -1
        self.conv.weight.data[0, 0, 1, 0] = -1
        self.conv.weight.data[0, 0, 2, 1] = -1
        self.conv.weight.data[0, 0, 1, 2] = -1
        self.conv.weight.requires_grad = False

    def forward(self, x):
        """
        Calculate the detail richness using a simple edge detection filter
        """
        # Average across channels to get grayscale-like representation
        avg_x = torch.mean(x, dim=1, keepdim=True)

        # Apply Laplacian filter for edge detection (approximates detail areas)
        df_values = torch.abs(self.conv(avg_x))

        # Normalize the DF values per batch
        B = x.shape[0]
        df_min = df_values.view(B, -1).min(dim=1, keepdim=True)[0].unsqueeze(-1).unsqueeze(-1)
        df_max = df_values.view(B, -1).max(dim=1, keepdim=True)[0].unsqueeze(-1).unsqueeze(-1)
        df_normalized = (df_values - df_min) / (df_max - df_min + 1e-8)

        # Map to connection counts
        range_connections = self.max_connections - self.min_connections
        connection_counts = self.min_connections + torch.round(df_normalized * range_connections)

        return df_values, connection_counts


class LightweightIPGLayer(nn.Module):
    """
    Memory-efficient implementation of IPG layer for large feature maps
    """

    def __init__(self, in_channels, min_connections=1, max_connections=8):
        super(LightweightIPGLayer, self).__init__()
        self.in_channels = in_channels
        self.df_detector = DFDetector(min_connections, max_connections)

        # Ensure num_groups properly divides in_channels
        num_groups = 1
        for i in range(min(16, in_channels), 0, -1):
            if in_channels % i == 0:
                num_groups = i
                break

        self.norm = nn.GroupNorm(num_groups, in_channels)
        self.conv_ffn = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * 2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)
        )

        # Fixed kernel sizes based on resolution
        self.kernel_sizes = {
            (48, 16): 3,  # Branch 4 (smallest)
            (96, 32): 3,  # Branch 3
            (192, 64): 3,  # Branch 2
            (384, 128): 3  # Branch 1 (largest)
        }

    def forward(self, x):
        """
        Memory-efficient implementation using fixed 3×3 windows for all scales
        """
        B, C, H, W = x.shape

        # Check if we need to reduce feature map size for very large inputs
        if H * W > 10000:  # Arbitrary threshold
            # Process downsampled version for large feature maps
            scale_factor = min(1.0, np.sqrt(10000 / (H * W)))
            new_H, new_W = int(H * scale_factor), int(W * scale_factor)
            x_small = F.interpolate(x, size=(new_H, new_W), mode='bilinear', align_corners=False)

            # Process the smaller feature map
            result_small = self._process_feature_map(x_small)

            # Upsample result back to original size
            result = F.interpolate(result_small, size=(H, W), mode='bilinear', align_corners=False)
            return result
        else:
            # Process original size for smaller feature maps
            return self._process_feature_map(x)

    def _process_feature_map(self, x):
        """
        Core processing function with fixed 3×3 window
        """
        B, C, H, W = x.shape

        # Calculate DF values and connection counts
        _, connection_counts = self.df_detector(x)

        # Use fixed 3×3 kernel for all scales to minimize memory usage
        kernel_size = self.kernel_sizes.get((H, W), 3)
        padding = kernel_size // 2

        # Get unfolded patches - this is memory efficient for 3×3 windows
        padded_x = F.pad(x, [padding, padding, padding, padding], mode='reflect')
        patches = F.unfold(padded_x, kernel_size=kernel_size)
        patches = patches.view(B, C, kernel_size ** 2, H * W)

        # Output tensor
        output = torch.zeros_like(x.view(B, C, H * W))

        # Process in chunks to save memory
        chunk_size = min(1024, H * W)  # Adjusted for memory constraints
        num_chunks = (H * W + chunk_size - 1) // chunk_size

        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, H * W)

            if start_idx >= end_idx:
                continue

            # Extract chunks
            chunk_patches = patches[:, :, :, start_idx:end_idx]  # [B, C, kernel_size², chunk_size]

            # Calculate positions for current chunk
            y_positions = (torch.arange(start_idx, end_idx, device=x.device) // W).long()
            x_positions = (torch.arange(start_idx, end_idx, device=x.device) % W).long()

            # Process each batch separately to save memory
            for b in range(B):
                # Get center features for this chunk (reshaped for cosine similarity)
                chunk_center = x[b, :, y_positions, x_positions]  # [C, chunk_size]
                chunk_center_norm = F.normalize(chunk_center, p=2, dim=0)  # Normalize for cosine similarity

                # Calculate similarities for each position in the chunk
                for i in range(end_idx - start_idx):
                    pos_idx = start_idx + i

                    # Get current position patches
                    pos_patches = chunk_patches[b, :, :, i]  # [C, kernel_size²]

                    # Normalize patches for cosine similarity
                    pos_patches_norm = F.normalize(pos_patches, p=2, dim=0)

                    # Get center feature for this position
                    center_feat = chunk_center_norm[:, i:i + 1]  # [C, 1]

                    # Calculate cosine similarities
                    similarities = torch.matmul(center_feat.t(), pos_patches_norm)  # [1, kernel_size²]

                    # Get connection count for this position
                    k = min(int(connection_counts[b, 0, y_positions[i], x_positions[i]].item()), kernel_size ** 2)

                    # Get top-k similar patches
                    top_sim, top_idx = torch.topk(similarities[0], k=k)

                    # Calculate weights
                    weights = torch.exp(top_sim)
                    weights = weights / weights.sum()

                    # Weight and sum selected patches
                    selected_patches = pos_patches[:, top_idx]  # [C, k]
                    weighted_sum = torch.matmul(selected_patches, weights)  # [C]

                    # Update output
                    output[b, :, pos_idx] = weighted_sum

        # Reshape output
        output = output.view(B, C, H, W)

        # Apply norm and FFN with residual connections
        x_norm = self.norm(x)
        enhanced = output + x_norm
        final_output = enhanced + self.conv_ffn(enhanced)

        return final_output


class SimpleIPGLayer(nn.Module):
    """
    Extremely simplified version of IPG for very limited GPU memory
    Uses regular convolution with attention weights based on DF values
    """

    def __init__(self, in_channels, min_attention=0.5, max_attention=1.0):
        super(SimpleIPGLayer, self).__init__()
        self.in_channels = in_channels

        # Edge detection for detail awareness
        self.df_detector = DFDetector(1, 8)  # We'll just use the DF values, not counts

        # Regular convolution for feature aggregation
        self.conv3x3 = nn.Conv2d(in_channels, in_channels, kernel_size=3,
                                 padding=1, groups=1, bias=False)

        # Ensure num_groups divides in_channels
        num_groups = 1
        for i in range(min(16, in_channels), 0, -1):
            if in_channels % i == 0:
                num_groups = i
                break

        self.norm = nn.GroupNorm(num_groups, in_channels)
        self.ffn = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * 2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)
        )

        self.min_attention = min_attention
        self.max_attention = max_attention

    def forward(self, x):
        """
        Extremely simplified GNN-inspired processing
        """
        # Get detail richness map
        df_values, _ = self.df_detector(x)

        # Normalize DF values
        B = x.shape[0]
        df_min = df_values.view(B, -1).min(dim=1, keepdim=True)[0].unsqueeze(-1).unsqueeze(-1)
        df_max = df_values.view(B, -1).max(dim=1, keepdim=True)[0].unsqueeze(-1).unsqueeze(-1)
        df_norm = (df_values - df_min) / (df_max - df_min + 1e-8)

        # Create attention weights based on detail richness
        attention_range = self.max_attention - self.min_attention
        attention_weights = self.min_attention + (df_norm * attention_range)

        # Apply regular convolution
        conv_features = self.conv3x3(x)

        # Apply attention weights
        weighted_features = conv_features * attention_weights

        # Residual connection
        enhanced = x + weighted_features

        # Normalization
        norm_enhanced = self.norm(enhanced)

        # FFN with residual
        output = norm_enhanced + self.ffn(norm_enhanced)

        return output