import os
import torch
import cv2
import numpy as np
from scipy.io import loadmat
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import mean_absolute_error, median_absolute_error
import random
from collections import defaultdict

from .models import vltenet
from .dataset import rearrange_pts, pafdata
from .utils import transform


def parse_args():
    parser = argparse.ArgumentParser(description='Test model performance on dataset')
    parser.add_argument('--model_path', type=str,
                        default=r"I:\GitProjects\VTF-18V\checkpoints_hrnet18_dual_endplate_hm1_vec0.05_cons0.0_enhanced\model_las0_vec.pth",
                        help='Path to the trained model')
    parser.add_argument('--data_dir', type=str,
                        default="I:/GitProjects/VTF-18V/dataset/data",
                        help='Directory containing the test images')
    parser.add_argument('--labels_dir', type=str,
                        default="I:/GitProjects/VTF-18V/dataset/labels",
                        help='Directory containing label files (.mat)')
    parser.add_argument('--test_list', type=str,
                        default="I:/GitProjects/VTF-18V/dataset/splits/fold0/test.txt",
                        help='File containing list of test images')
    parser.add_argument('--input_h', type=int, default=1536, help='Input height')
    parser.add_argument('--input_w', type=int, default=512, help='Input width')
    parser.add_argument('--down_ratio', type=int, default=4, help='Downsampling ratio')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save test results')
    parser.add_argument('--backbone', type=str, default='hrnet18',
                        help='Backbone: hrnet18 or hrnet32')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize predictions')
    parser.add_argument('--bootstrap_iterations', type=int, default=1000,
                        help='Number of bootstrap iterations for confidence intervals')
    parser.add_argument('--conf_interval', type=float, default=0.95,
                        help='Confidence interval (default: 0.95 for 95% CI)')
    return parser.parse_args()


def load_model(model_path, backbone='hrnet18'):
    print(f"Loading model from {model_path}...")

    # 检查模型路径是否包含transformer
    use_transformer = 'transformer' in model_path

    if use_transformer:
        print("Initializing model with transformer enhancement...")
        # 从模型路径中提取transformer参数
        # 例如 transformer_16x16_ds4 表示 window_height=16, window_width=16, downsample_factor=4
        transformer_info = model_path.split('transformer_')[1].split('_')[
            0] if 'transformer_' in model_path else '16x16'
        ds_info = model_path.split('ds')[1].split('\\')[0] if 'ds' in model_path else '4'

        window_h, window_w = map(int, transformer_info.split('x')) if 'x' in transformer_info else (16, 16)
        downsample_factor = int(ds_info) if ds_info.isdigit() else 4

        print(f"Transformer config: window size {window_h}x{window_w}, downsample factor {downsample_factor}")

        transformer_config = {
            'window_height': window_h,
            'window_width': window_w,
            'downsample_factor': downsample_factor
        }

        model = vltenet.Vltenet(
            pretrained=False,
            final_kernel=1,
            backbone=backbone,
            use_transformer=True,
            transformer_config=transformer_config
        )
    else:
        model = vltenet.Vltenet(pretrained=False, final_kernel=1, backbone=backbone)

    # 加载权重
    checkpoint = torch.load(model_path, map_location='cpu')
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)

    if torch.cuda.is_available():
        model = model.cuda()
        print("Model loaded on CUDA device")
    else:
        print("Model loaded on CPU device")

    model.eval()
    return model


def get_test_images(test_list_path):
    with open(test_list_path, 'r') as f:
        test_images = [line.strip() for line in f.readlines()]
    return test_images


def extract_keypoints_from_heatmap(heatmap, num_peaks=18, threshold=0.05, nms_radius=3):
    """
    从热图提取关键点并按照从下到上的顺序排列

    Args:
        heatmap: 单通道热图 [H, W]
        num_peaks: 要提取的峰值数量
        threshold: 检测置信度阈值 (降低为0.05以增加敏感度)
        nms_radius: 非极大值抑制的半径 (从5降低为3以检测更密集的点)

    Returns:
        ordered_points: 从下到上排序的点（y坐标降序）
        scores: 每个点的置信度得分
    """
    height, width = heatmap.shape

    # 应用高斯模糊减少噪声，使用较小核以保留更多细节
    hm = cv2.GaussianBlur(heatmap, (3, 3), 0)

    # 寻找局部最大值
    points = []
    scores = []

    # 为NMS创建热图的工作副本
    hm_working = hm.copy()

    for i in range(num_peaks):
        # 找到全局最大值
        max_val = np.max(hm_working)
        if max_val <= threshold:
            break  # 没有更多高于阈值的峰值

        # 获取峰值位置
        max_idx = np.argmax(hm_working)
        y = max_idx // width
        x = max_idx % width

        # 存储峰值位置和得分
        points.append([x, y])
        scores.append(max_val)

        # 应用NMS - 将检测到的峰值周围区域置零，使用更小的半径
        y_min = max(0, y - nms_radius)
        y_max = min(height, y + nms_radius + 1)
        x_min = max(0, x - nms_radius)
        x_max = min(width, x + nms_radius + 1)

        hm_working[y_min:y_max, x_min:x_max] = 0

    points = np.array(points)
    scores = np.array(scores)

    if len(points) == 0:
        return np.empty((0, 2)), np.empty(0)

    # 按y坐标排序（从下到上）
    sort_indices = np.argsort(points[:, 1])[::-1]  # 降序
    ordered_points = points[sort_indices]
    ordered_scores = scores[sort_indices]

    return ordered_points, ordered_scores


def calculate_angle(vector):
    return np.degrees(np.arctan2(vector[1], vector[0]))


def normalize_vector(vector):
    norm = np.linalg.norm(vector)
    if norm < 1e-6:
        return np.zeros_like(vector)
    return vector / norm


def calculate_angle_error(pred_vector, gt_vector):
    pred_vector = normalize_vector(pred_vector)
    gt_vector = normalize_vector(gt_vector)

    dot_product = np.clip(np.dot(pred_vector, gt_vector), -1.0, 1.0)
    angle_error = np.degrees(np.arccos(dot_product))

    return angle_error


def evaluate_model(model, data_dir, labels_dir, test_images, input_h, input_w, down_ratio):
    position_errors_upper = []
    position_errors_lower = []
    angle_errors_upper = []
    angle_errors_lower = []
    detection_rates = []

    transforms = transform.Compose([
        transform.ConvertImgFloat(),
        transform.Resize(h=input_h, w=input_w)
    ])

    results = []

    for img_name in tqdm(test_images, desc="Evaluating"):
        img_path = os.path.join(data_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Cannot load image {img_path}")
            continue

        original_img = img.copy()

        label_path = os.path.join(labels_dir, os.path.splitext(img_name)[0] + '.mat')
        try:
            pts = loadmat(label_path)['p2']
            pts = rearrange_pts(pts)
        except Exception as e:
            print(f"Warning: Cannot load label {label_path}: {e}")
            continue

        img_transformed, pts_transformed = transforms(img.copy(), pts.copy())
        img_tensor = np.transpose((np.clip(img_transformed, 0, 255) - 128) / 255., (2, 0, 1))
        img_tensor = torch.from_numpy(img_tensor).float().unsqueeze(0)

        gt_centers_upper = []
        gt_centers_lower = []
        gt_vectors_upper = []
        gt_vectors_lower = []

        for k in range(18):
            pts_group = pts_transformed[4 * k:4 * k + 4, :]

            upper_pts = pts_group[:2]
            upper_center = np.mean(upper_pts, axis=0)
            gt_centers_upper.append(upper_center)

            upper_vec = normalize_vector(upper_pts[1] - upper_pts[0])
            gt_vectors_upper.append(upper_vec)

            lower_pts = pts_group[2:4]
            lower_center = np.mean(lower_pts, axis=0)
            gt_centers_lower.append(lower_center)

            lower_vec = normalize_vector(lower_pts[1] - lower_pts[0])
            gt_vectors_lower.append(lower_vec)

        gt_centers_upper = np.array(gt_centers_upper)
        gt_centers_lower = np.array(gt_centers_lower)
        gt_vectors_upper = np.array(gt_vectors_upper)
        gt_vectors_lower = np.array(gt_vectors_lower)

        if torch.cuda.is_available():
            img_tensor = img_tensor.cuda()

        with torch.no_grad():
            outputs = model(img_tensor)

        heatmaps = outputs['hm'][0].cpu().numpy()

        pred_centers_upper, upper_scores = extract_keypoints_from_heatmap(
            heatmaps[0],
            num_peaks=18,
            threshold=0.1
        )

        pred_centers_lower, lower_scores = extract_keypoints_from_heatmap(
            heatmaps[1],
            num_peaks=18,
            threshold=0.1
        )

        pred_centers_upper = pred_centers_upper * down_ratio
        pred_centers_lower = pred_centers_lower * down_ratio

        pred_vectors_upper = []
        pred_vectors_lower = []

        vector_map = outputs['vec_ind'][0].cpu().numpy().transpose(1, 2, 0)

        for point in pred_centers_upper:
            x, y = int(point[0] / down_ratio), int(point[1] / down_ratio)
            if 0 <= y < vector_map.shape[0] and 0 <= x < vector_map.shape[1]:
                vec = vector_map[y, x, :]
                pred_vectors_upper.append(normalize_vector(vec))
            else:
                pred_vectors_upper.append(np.array([1.0, 0.0]))

        for point in pred_centers_lower:
            x, y = int(point[0] / down_ratio), int(point[1] / down_ratio)
            if 0 <= y < vector_map.shape[0] and 0 <= x < vector_map.shape[1]:
                vec = vector_map[y, x, :]
                pred_vectors_lower.append(normalize_vector(vec))
            else:
                pred_vectors_lower.append(np.array([1.0, 0.0]))

        pred_vectors_upper = np.array(pred_vectors_upper)
        pred_vectors_lower = np.array(pred_vectors_lower)

        pred_count_upper = min(len(pred_centers_upper), len(gt_centers_upper))
        pred_count_lower = min(len(pred_centers_lower), len(gt_centers_lower))

        for i in range(pred_count_upper):
            error = np.linalg.norm(pred_centers_upper[i] - gt_centers_upper[i])
            position_errors_upper.append(error)

        for i in range(pred_count_lower):
            error = np.linalg.norm(pred_centers_lower[i] - gt_centers_lower[i])
            position_errors_lower.append(error)

        for i in range(pred_count_upper):
            if len(pred_vectors_upper) > i:
                error = calculate_angle_error(pred_vectors_upper[i], gt_vectors_upper[i])
                angle_errors_upper.append(error)

        for i in range(pred_count_lower):
            if len(pred_vectors_lower) > i:
                error = calculate_angle_error(pred_vectors_lower[i], gt_vectors_lower[i])
                angle_errors_lower.append(error)

        detection_rate = (pred_count_upper + pred_count_lower) / (len(gt_centers_upper) + len(gt_centers_lower))
        detection_rates.append(detection_rate)

        result = {
            'img_name': img_name,
            'pred_count_upper': pred_count_upper,
            'pred_count_lower': pred_count_lower,
            'gt_count_upper': len(gt_centers_upper),
            'gt_count_lower': len(gt_centers_lower),
            'detection_rate': detection_rate,
            'upper_position_errors': [np.linalg.norm(pred_centers_upper[i] - gt_centers_upper[i]) for i in
                                      range(pred_count_upper)],
            'lower_position_errors': [np.linalg.norm(pred_centers_lower[i] - gt_centers_lower[i]) for i in
                                      range(pred_count_lower)],
            'upper_angle_errors': [calculate_angle_error(pred_vectors_upper[i], gt_vectors_upper[i]) for i in
                                   range(pred_count_upper) if len(pred_vectors_upper) > i],
            'lower_angle_errors': [calculate_angle_error(pred_vectors_lower[i], gt_vectors_lower[i]) for i in
                                   range(pred_count_lower) if len(pred_vectors_lower) > i],
        }
        results.append(result)

    position_errors_all = position_errors_upper + position_errors_lower
    angle_errors_all = angle_errors_upper + angle_errors_lower

    stats = {
        'num_test_images': len(test_images),
        'num_evaluated_images': len(results),

        'position_error_mean_all': np.mean(position_errors_all) if position_errors_all else 0,
        'position_error_median_all': np.median(position_errors_all) if position_errors_all else 0,
        'position_error_mean_upper': np.mean(position_errors_upper) if position_errors_upper else 0,
        'position_error_median_upper': np.median(position_errors_upper) if position_errors_upper else 0,
        'position_error_mean_lower': np.mean(position_errors_lower) if position_errors_lower else 0,
        'position_error_median_lower': np.median(position_errors_lower) if position_errors_lower else 0,

        'angle_error_mean_all': np.mean(angle_errors_all) if angle_errors_all else 0,
        'angle_error_median_all': np.median(angle_errors_all) if angle_errors_all else 0,
        'angle_error_mean_upper': np.mean(angle_errors_upper) if angle_errors_upper else 0,
        'angle_error_median_upper': np.median(angle_errors_upper) if angle_errors_upper else 0,
        'angle_error_mean_lower': np.mean(angle_errors_lower) if angle_errors_lower else 0,
        'angle_error_median_lower': np.median(angle_errors_lower) if angle_errors_lower else 0,

        'detection_rate_mean': np.mean(detection_rates) if detection_rates else 0,
    }

    return stats, results, {
        'position_errors_upper': position_errors_upper,
        'position_errors_lower': position_errors_lower,
        'angle_errors_upper': angle_errors_upper,
        'angle_errors_lower': angle_errors_lower,
        'detection_rates': detection_rates
    }


def bootstrap_confidence_intervals(results, errors, n_iterations=1000, conf_level=0.95):
    """
    Calculate bootstrap confidence intervals for metrics

    Args:
        results: List of result dictionaries from evaluate_model
        errors: Dictionary containing lists of errors
        n_iterations: Number of bootstrap iterations
        conf_level: Confidence level (e.g., 0.95 for 95% CI)

    Returns:
        Dictionary containing confidence intervals for each metric
    """
    print(f"\nCalculating {conf_level * 100:.0f}% confidence intervals using {n_iterations} bootstrap iterations...")

    # Calculate alpha for the confidence interval
    alpha = (1 - conf_level) / 2
    lower_percentile = alpha * 100
    upper_percentile = (1 - alpha) * 100

    # Metrics to bootstrap
    metrics = [
        'position_error_mean_all', 'position_error_median_all',
        'position_error_mean_upper', 'position_error_median_upper',
        'position_error_mean_lower', 'position_error_median_lower',
        'angle_error_mean_all', 'angle_error_median_all',
        'angle_error_mean_upper', 'angle_error_median_upper',
        'angle_error_mean_lower', 'angle_error_median_lower',
        'detection_rate_mean'
    ]

    # Initialize storage for bootstrap samples
    bootstrap_samples = {metric: [] for metric in metrics}

    # Store individual errors per image for resampling
    image_errors = []
    for result in results:
        img_errors = {
            'upper_position_errors': result['upper_position_errors'],
            'lower_position_errors': result['lower_position_errors'],
            'upper_angle_errors': result['upper_angle_errors'],
            'lower_angle_errors': result['lower_angle_errors'],
            'detection_rate': result['detection_rate']
        }
        image_errors.append(img_errors)

    # Perform bootstrap iterations
    for i in tqdm(range(n_iterations), desc="Bootstrap Resampling"):
        # Resample with replacement
        bootstrap_indices = np.random.choice(len(image_errors), size=len(image_errors), replace=True)
        bootstrap_errors = [image_errors[idx] for idx in bootstrap_indices]

        # Collect all errors for this bootstrap iteration
        bootstrap_position_errors_upper = []
        bootstrap_position_errors_lower = []
        bootstrap_angle_errors_upper = []
        bootstrap_angle_errors_lower = []
        bootstrap_detection_rates = []

        for err in bootstrap_errors:
            bootstrap_position_errors_upper.extend(err['upper_position_errors'])
            bootstrap_position_errors_lower.extend(err['lower_position_errors'])
            bootstrap_angle_errors_upper.extend(err['upper_angle_errors'])
            bootstrap_angle_errors_lower.extend(err['lower_angle_errors'])
            bootstrap_detection_rates.append(err['detection_rate'])

        bootstrap_position_errors_all = bootstrap_position_errors_upper + bootstrap_position_errors_lower
        bootstrap_angle_errors_all = bootstrap_angle_errors_upper + bootstrap_angle_errors_lower

        # Calculate metrics for this bootstrap sample
        bootstrap_stats = {
            'position_error_mean_all': np.mean(bootstrap_position_errors_all) if bootstrap_position_errors_all else 0,
            'position_error_median_all': np.median(
                bootstrap_position_errors_all) if bootstrap_position_errors_all else 0,
            'position_error_mean_upper': np.mean(
                bootstrap_position_errors_upper) if bootstrap_position_errors_upper else 0,
            'position_error_median_upper': np.median(
                bootstrap_position_errors_upper) if bootstrap_position_errors_upper else 0,
            'position_error_mean_lower': np.mean(
                bootstrap_position_errors_lower) if bootstrap_position_errors_lower else 0,
            'position_error_median_lower': np.median(
                bootstrap_position_errors_lower) if bootstrap_position_errors_lower else 0,

            'angle_error_mean_all': np.mean(bootstrap_angle_errors_all) if bootstrap_angle_errors_all else 0,
            'angle_error_median_all': np.median(bootstrap_angle_errors_all) if bootstrap_angle_errors_all else 0,
            'angle_error_mean_upper': np.mean(bootstrap_angle_errors_upper) if bootstrap_angle_errors_upper else 0,
            'angle_error_median_upper': np.median(bootstrap_angle_errors_upper) if bootstrap_angle_errors_upper else 0,
            'angle_error_mean_lower': np.mean(bootstrap_angle_errors_lower) if bootstrap_angle_errors_lower else 0,
            'angle_error_median_lower': np.median(bootstrap_angle_errors_lower) if bootstrap_angle_errors_lower else 0,

            'detection_rate_mean': np.mean(bootstrap_detection_rates) if bootstrap_detection_rates else 0,
        }

        # Store the values
        for metric in metrics:
            bootstrap_samples[metric].append(bootstrap_stats[metric])

    # Calculate confidence intervals
    confidence_intervals = {}
    for metric in metrics:
        lower = np.percentile(bootstrap_samples[metric], lower_percentile)
        upper = np.percentile(bootstrap_samples[metric], upper_percentile)
        confidence_intervals[metric] = (lower, upper)

    return confidence_intervals


def visualize_results_with_ci(stats, errors, confidence_intervals, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(12, 6))

    plt.subplot(2, 2, 1)
    plt.hist(errors['position_errors_upper'], bins=20, alpha=0.7, color='blue', label='Upper endplate')
    plt.xlabel('Position Error (pixels)')
    plt.ylabel('Frequency')
    plt.title('Upper Endplate Position Errors')
    plt.axvline(stats['position_error_mean_upper'], color='r', linestyle='--',
                label=f'Mean: {stats["position_error_mean_upper"]:.2f} [{confidence_intervals["position_error_mean_upper"][0]:.2f}, {confidence_intervals["position_error_mean_upper"][1]:.2f}]')
    plt.axvline(stats['position_error_median_upper'], color='g', linestyle='--',
                label=f'Median: {stats["position_error_median_upper"]:.2f} [{confidence_intervals["position_error_median_upper"][0]:.2f}, {confidence_intervals["position_error_median_upper"][1]:.2f}]')
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)

    plt.subplot(2, 2, 2)
    plt.hist(errors['position_errors_lower'], bins=20, alpha=0.7, color='orange', label='Lower endplate')
    plt.xlabel('Position Error (pixels)')
    plt.ylabel('Frequency')
    plt.title('Lower Endplate Position Errors')
    plt.axvline(stats['position_error_mean_lower'], color='r', linestyle='--',
                label=f'Mean: {stats["position_error_mean_lower"]:.2f} [{confidence_intervals["position_error_mean_lower"][0]:.2f}, {confidence_intervals["position_error_mean_lower"][1]:.2f}]')
    plt.axvline(stats['position_error_median_lower'], color='g', linestyle='--',
                label=f'Median: {stats["position_error_median_lower"]:.2f} [{confidence_intervals["position_error_median_lower"][0]:.2f}, {confidence_intervals["position_error_median_lower"][1]:.2f}]')
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)

    plt.subplot(2, 2, 3)
    plt.hist(errors['angle_errors_upper'], bins=20, alpha=0.7, color='blue', label='Upper endplate')
    plt.xlabel('Angle Error (degrees)')
    plt.ylabel('Frequency')
    plt.title('Upper Endplate Angle Errors')
    plt.axvline(stats['angle_error_mean_upper'], color='r', linestyle='--',
                label=f'Mean: {stats["angle_error_mean_upper"]:.2f}° [{confidence_intervals["angle_error_mean_upper"][0]:.2f}, {confidence_intervals["angle_error_mean_upper"][1]:.2f}]°')
    plt.axvline(stats['angle_error_median_upper'], color='g', linestyle='--',
                label=f'Median: {stats["angle_error_median_upper"]:.2f}° [{confidence_intervals["angle_error_median_upper"][0]:.2f}, {confidence_intervals["angle_error_median_upper"][1]:.2f}]°')
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)

    plt.subplot(2, 2, 4)
    plt.hist(errors['angle_errors_lower'], bins=20, alpha=0.7, color='orange', label='Lower endplate')
    plt.xlabel('Angle Error (degrees)')
    plt.ylabel('Frequency')
    plt.title('Lower Endplate Angle Errors')
    plt.axvline(stats['angle_error_mean_lower'], color='r', linestyle='--',
                label=f'Mean: {stats["angle_error_mean_lower"]:.2f}° [{confidence_intervals["angle_error_mean_lower"][0]:.2f}, {confidence_intervals["angle_error_mean_lower"][1]:.2f}]°')
    plt.axvline(stats['angle_error_median_lower'], color='g', linestyle='--',
                label=f'Median: {stats["angle_error_median_lower"]:.2f}° [{confidence_intervals["angle_error_median_lower"][0]:.2f}, {confidence_intervals["angle_error_median_lower"][1]:.2f}]°')
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'error_distributions_with_ci.png'), dpi=200)
    plt.close()

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.hist(errors['position_errors_upper'], bins=20, alpha=0.6, color='blue', label='Upper endplate')
    plt.hist(errors['position_errors_lower'], bins=20, alpha=0.6, color='orange', label='Lower endplate')
    plt.xlabel('Position Error (pixels)')
    plt.ylabel('Frequency')
    plt.title('Position Errors')
    plt.legend()
    plt.grid(alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.hist(errors['angle_errors_upper'], bins=20, alpha=0.6, color='blue', label='Upper endplate')
    plt.hist(errors['angle_errors_lower'], bins=20, alpha=0.6, color='orange', label='Lower endplate')
    plt.xlabel('Angle Error (degrees)')
    plt.ylabel('Frequency')
    plt.title('Angle Errors')
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'combined_error_distributions.png'), dpi=200)
    plt.close()

    # Add visualization for bootstrap distributions
    visualize_bootstrap_distributions(stats, confidence_intervals, output_dir)

    print(f"Visualizations saved to {output_dir}")


def visualize_bootstrap_distributions(stats, confidence_intervals, output_dir):
    """
    Visualize the bootstrap distributions and confidence intervals
    """
    metrics_to_plot = [
        ('position_error_mean_all', 'Overall Position Error (Mean)'),
        ('position_error_mean_upper', 'Upper Endplate Position Error (Mean)'),
        ('position_error_mean_lower', 'Lower Endplate Position Error (Mean)'),
        ('angle_error_mean_all', 'Overall Angle Error (Mean)'),
        ('angle_error_mean_upper', 'Upper Endplate Angle Error (Mean)'),
        ('angle_error_mean_lower', 'Lower Endplate Angle Error (Mean)'),
        ('detection_rate_mean', 'Detection Rate')
    ]

    rows = len(metrics_to_plot) // 2 + len(metrics_to_plot) % 2
    fig, axes = plt.subplots(rows, 2, figsize=(14, 3 * rows))
    axes = axes.flatten()

    for i, (metric, title) in enumerate(metrics_to_plot):
        if i < len(axes):
            # Create artificial bootstrap distribution for visualization
            mean = stats[metric]
            ci_lower, ci_upper = confidence_intervals[metric]
            std = (ci_upper - ci_lower) / 3.92  # Approximation for 95% CI width

            # Generate normal distribution for visual representation
            x = np.linspace(ci_lower - 0.5 * (ci_upper - ci_lower),
                            ci_upper + 0.5 * (ci_upper - ci_lower), 1000)
            y = np.exp(-0.5 * ((x - mean) / std) ** 2) / (std * np.sqrt(2 * np.pi))

            axes[i].plot(x, y, 'k-', linewidth=1.5)
            axes[i].axvline(mean, color='r', linestyle='-', linewidth=2, label=f'Mean: {mean:.2f}')
            axes[i].axvline(ci_lower, color='g', linestyle='--', linewidth=1.5,
                            label=f'95% CI: [{ci_lower:.2f}, {ci_upper:.2f}]')
            axes[i].axvline(ci_upper, color='g', linestyle='--', linewidth=1.5)
            axes[i].fill_between(x, 0, y, where=(x >= ci_lower) & (x <= ci_upper),
                                 alpha=0.3, color='green')
            axes[i].set_title(title)
            axes[i].legend(fontsize=8)
            axes[i].grid(alpha=0.3)

            # Remove y-axis as it's just a visual representation
            axes[i].set_yticklabels([])

    # Hide any unused axes
    for i in range(len(metrics_to_plot), len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'bootstrap_distributions.png'), dpi=200)
    plt.close()


def main():
    args = parse_args()

    if args.output_dir is None:
        model_dir = os.path.dirname(args.model_path)
        model_name = os.path.basename(args.model_path).split('.')[0]
        args.output_dir = os.path.join(model_dir, f'test_results_{model_name}')

    os.makedirs(args.output_dir, exist_ok=True)

    model = load_model(args.model_path, args.backbone)

    test_images = get_test_images(args.test_list)
    print(f"Found {len(test_images)} test images")

    stats, results, errors = evaluate_model(
        model,
        args.data_dir,
        args.labels_dir,
        test_images,
        args.input_h,
        args.input_w,
        args.down_ratio
    )

    # Calculate bootstrap confidence intervals
    confidence_intervals = bootstrap_confidence_intervals(
        results,
        errors,
        n_iterations=args.bootstrap_iterations,
        conf_level=args.conf_interval
    )

    # Calculate uncertainties as half of CI width
    uncertainties = {}
    for metric in confidence_intervals.keys():
        lower, upper = confidence_intervals[metric]
        uncertainties[metric] = (upper - lower) / 2

    print("\n===== Model Performance with Uncertainties =====")
    print(f"Number of test images: {stats['num_test_images']}")
    print(f"Number of evaluated images: {stats['num_evaluated_images']}")
    print("\n--- Position Errors (in pixels) ---")
    print(
        f"Overall - Mean: {stats['position_error_mean_all']:.2f}±{uncertainties['position_error_mean_all']:.2f}, Median: {stats['position_error_median_all']:.2f}±{uncertainties['position_error_median_all']:.2f}")
    print(
        f"Upper Endplate - Mean: {stats['position_error_mean_upper']:.2f}±{uncertainties['position_error_mean_upper']:.2f}, Median: {stats['position_error_median_upper']:.2f}±{uncertainties['position_error_median_upper']:.2f}")
    print(
        f"Lower Endplate - Mean: {stats['position_error_mean_lower']:.2f}±{uncertainties['position_error_mean_lower']:.2f}, Median: {stats['position_error_median_lower']:.2f}±{uncertainties['position_error_median_lower']:.2f}")

    print("\n--- Angle Errors (in degrees) ---")
    print(
        f"Overall - Mean: {stats['angle_error_mean_all']:.2f}±{uncertainties['angle_error_mean_all']:.2f}, Median: {stats['angle_error_median_all']:.2f}±{uncertainties['angle_error_median_all']:.2f}")
    print(
        f"Upper Endplate - Mean: {stats['angle_error_mean_upper']:.2f}±{uncertainties['angle_error_mean_upper']:.2f}, Median: {stats['angle_error_median_upper']:.2f}±{uncertainties['angle_error_median_upper']:.2f}")
    print(
        f"Lower Endplate - Mean: {stats['angle_error_mean_lower']:.2f}±{uncertainties['angle_error_mean_lower']:.2f}, Median: {stats['angle_error_median_lower']:.2f}±{uncertainties['angle_error_median_lower']:.2f}")

    print(
        f"\nAverage Detection Rate: {stats['detection_rate_mean'] * 100:.2f}%±{uncertainties['detection_rate_mean'] * 100:.2f}%")

    # Calculate uncertainties for the text file as well
    uncertainties = {}
    for metric in confidence_intervals.keys():
        lower, upper = confidence_intervals[metric]
        uncertainties[metric] = (upper - lower) / 2

    with open(os.path.join(args.output_dir, 'stats_with_uncertainties.txt'), 'w') as f:
        f.write("===== Model Performance with Uncertainties =====\n")
        f.write(f"Model: {args.model_path}\n")
        f.write(f"Backbone: {args.backbone}\n")
        f.write(f"Test List: {args.test_list}\n")
        f.write(f"Bootstrap Iterations: {args.bootstrap_iterations}\n")
        f.write(f"Confidence Level: {args.conf_interval * 100:.0f}%\n\n")

        f.write(f"Number of test images: {stats['num_test_images']}\n")
        f.write(f"Number of evaluated images: {stats['num_evaluated_images']}\n\n")

        f.write("--- Position Errors (in pixels) ---\n")
        f.write(
            f"Overall - Mean: {stats['position_error_mean_all']:.4f}±{uncertainties['position_error_mean_all']:.4f}, Median: {stats['position_error_median_all']:.4f}±{uncertainties['position_error_median_all']:.4f}\n")
        f.write(
            f"Upper Endplate - Mean: {stats['position_error_mean_upper']:.4f}±{uncertainties['position_error_mean_upper']:.4f}, Median: {stats['position_error_median_upper']:.4f}±{uncertainties['position_error_median_upper']:.4f}\n")
        f.write(
            f"Lower Endplate - Mean: {stats['position_error_mean_lower']:.4f}±{uncertainties['position_error_mean_lower']:.4f}, Median: {stats['position_error_median_lower']:.4f}±{uncertainties['position_error_median_lower']:.4f}\n\n")

        f.write("--- Angle Errors (in degrees) ---\n")
        f.write(
            f"Overall - Mean: {stats['angle_error_mean_all']:.4f}±{uncertainties['angle_error_mean_all']:.4f}, Median: {stats['angle_error_median_all']:.4f}±{uncertainties['angle_error_median_all']:.4f}\n")
        f.write(
            f"Upper Endplate - Mean: {stats['angle_error_mean_upper']:.4f}±{uncertainties['angle_error_mean_upper']:.4f}, Median: {stats['angle_error_median_upper']:.4f}±{uncertainties['angle_error_median_upper']:.4f}\n")
        f.write(
            f"Lower Endplate - Mean: {stats['angle_error_mean_lower']:.4f}±{uncertainties['angle_error_mean_lower']:.4f}, Median: {stats['angle_error_median_lower']:.4f}±{uncertainties['angle_error_median_lower']:.4f}\n\n")

        f.write(
            f"Average Detection Rate: {stats['detection_rate_mean'] * 100:.2f}%±{uncertainties['detection_rate_mean'] * 100:.2f}%\n")

        # Also write the raw confidence intervals for reference
        f.write("\n\n===== Raw 95% Confidence Intervals =====\n")
        for metric, (lower, upper) in confidence_intervals.items():
            f.write(f"{metric}: [{lower:.4f}, {upper:.4f}]\n")

    # Only create basic visualizations if requested
    if args.visualize:
        visualize_results_with_ci(stats, errors, confidence_intervals, args.output_dir)
    else:
        print("\nSkipping visualizations as --visualize flag was not set")

    print(f"\nTest results with confidence intervals saved to {args.output_dir}")


if __name__ == "__main__":
    main()