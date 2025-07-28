import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from loss import RegL1Loss, AngleConstraintLoss

plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'DejaVu Sans'


def visualize_epoch(model, epoch, save_dir, device='cuda'):
    vis_dir = os.path.join(save_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)

    image_path = "I:/GitProjects/VTF-18V/dataset/data/01198.png"

    model.eval()

    reg_l1_loss = RegL1Loss()
    constraint_calculator = AngleConstraintLoss()

    with torch.no_grad():
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not load image {image_path}")
            return

        orig_h, orig_w = img.shape[:2]

        img_float = img.astype(np.float32) / 255.0
        img_resized = cv2.resize(img_float, (512, 1536))
        img_normalized = (img_resized - 0.5) * 2.0
        img_transposed = np.transpose(img_normalized, (2, 0, 1))

        img_tensor = torch.from_numpy(img_transposed).float().unsqueeze(0).to(device)
        outputs = model(img_tensor)

        if 'hm' in outputs:
            heatmaps = outputs['hm']
        else:
            print("Error: Model output doesn't contain 'hm' key")
            return

        if 'vec_ind' in outputs:
            vectors = outputs['vec_ind']
        else:
            print("Error: Model output doesn't contain 'vec_ind' key")
            return

        if 'peak_points_upper' in outputs:
            centers_upper = outputs['peak_points_upper']
        else:
            print("Error: Model output doesn't contain 'peak_points_upper' key")
            return

        if 'peak_points_lower' in outputs:
            centers_lower = outputs['peak_points_lower']
        else:
            print("Error: Model output doesn't contain 'peak_points_lower' key")
            return

        h, w = heatmaps.size(2), heatmaps.size(3)
        indices = torch.zeros((1, 36), dtype=torch.int64, device=device)

        for c in range(18):
            y, x = centers_upper[0, c, 1].long(), centers_upper[0, c, 0].long()
            y = torch.clamp(y, 0, h - 1)
            x = torch.clamp(x, 0, w - 1)
            indices[0, c] = y * w + x

        for c in range(18):
            y, x = centers_lower[0, c, 1].long(), centers_lower[0, c, 0].long()
            y = torch.clamp(y, 0, h - 1)
            x = torch.clamp(x, 0, w - 1)
            indices[0, c + 18] = y * w + x

        all_directions = reg_l1_loss._tranpose_and_gather_feat(vectors, indices)
        directions_upper = all_directions[0, :18]
        directions_lower = all_directions[0, 18:]

        avg_centers = (centers_upper + centers_lower) / 2
        constraint_loss = constraint_calculator(
            avg_centers,
            directions_upper.unsqueeze(0),
            directions_lower.unsqueeze(0)
        )

        visualize_predictions(
            img_resized,
            centers_upper[0].cpu().numpy(),
            centers_lower[0].cpu().numpy(),
            directions_upper.cpu().numpy(),
            directions_lower.cpu().numpy(),
            heatmaps[0].cpu().numpy(),
            constraint_loss.item(),
            os.path.join(vis_dir, f"epoch_{epoch:03d}")
        )

        visualize_heatmaps(
            heatmaps[0].cpu().numpy(),
            os.path.join(vis_dir, f"epoch_{epoch:03d}_heatmaps")
        )


def visualize_predictions(input_img, centers_upper, centers_lower, directions_upper, directions_lower,
                          heatmaps, constraint_loss, save_path):
    h, w = heatmaps.shape[1:]
    downsampled_img = cv2.resize(input_img, (w, h))

    fig = plt.figure(figsize=(18, 14))

    ax1 = fig.add_subplot(2, 2, 1)
    ax1.imshow(cv2.cvtColor(downsampled_img, cv2.COLOR_BGR2RGB))
    ax1.set_title(f"Upper Endplate Keypoints ({w}x{h})")

    for i in range(centers_upper.shape[0]):
        x, y = centers_upper[i]
        dx, dy = directions_upper[i]

        ax1.scatter(x, y, c=f'C{i%10}', s=20, label=f'Point {i}')

        arrow_length = 5
        ax1.arrow(x, y, dx * arrow_length, dy * arrow_length,
                  head_width=1, head_length=1, fc=f'C{i%10}', ec=f'C{i%10}', alpha=0.7)

    for i in range(centers_upper.shape[0] - 1):
        x1, y1 = centers_upper[i]
        x2, y2 = centers_upper[i + 1]
        ax1.plot([x1, x2], [y1, y2], 'k--', alpha=0.5)

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.imshow(cv2.cvtColor(downsampled_img, cv2.COLOR_BGR2RGB))
    ax2.set_title(f"Lower Endplate Keypoints ({w}x{h})")

    for i in range(centers_lower.shape[0]):
        x, y = centers_lower[i]
        dx, dy = directions_lower[i]

        ax2.scatter(x, y, c=f'C{i%10}', s=20, label=f'Point {i}')

        arrow_length = 5
        ax2.arrow(x, y, dx * arrow_length, dy * arrow_length,
                  head_width=1, head_length=1, fc=f'C{i%10}', ec=f'C{i%10}', alpha=0.7)

    for i in range(centers_lower.shape[0] - 1):
        x1, y1 = centers_lower[i]
        x2, y2 = centers_lower[i + 1]
        ax2.plot([x1, x2], [y1, y2], 'k--', alpha=0.5)

    ax3 = fig.add_subplot(2, 2, 3)
    combined_heatmap = np.max(heatmaps, axis=0)
    ax3.imshow(combined_heatmap, cmap='jet')
    ax3.set_title(f"Combined Heatmap (Upper & Lower Endplates)")

    for i in range(centers_upper.shape[0]):
        x, y = centers_upper[i]
        ax3.scatter(x, y, c='white', s=15, edgecolor='black')
        ax3.text(x + 1, y + 1, f"U{i}", color='white', fontsize=8)

    for i in range(centers_lower.shape[0]):
        x, y = centers_lower[i]
        ax3.scatter(x, y, c='yellow', s=15, edgecolor='black')
        ax3.text(x + 1, y + 1, f"L{i}", color='yellow', fontsize=8)

    ax4 = fig.add_subplot(2, 2, 4)

    angles_upper = np.degrees(np.arctan2(directions_upper[:, 1], directions_upper[:, 0]))
    angles_lower = np.degrees(np.arctan2(directions_lower[:, 1], directions_lower[:, 0]))
    angles_avg = (angles_upper + angles_lower) / 2

    centers_avg = (centers_upper + centers_lower) / 2
    center_vecs = centers_avg[1:] - centers_avg[:-1]
    center_angles = np.degrees(np.arctan2(center_vecs[:, 1], center_vecs[:, 0]))

    beta_angles = []
    for i in range(len(center_angles)):
        if i < len(center_angles) - 1:
            beta = (center_angles[i] + center_angles[i + 1]) / 2 + 90
        else:
            beta = center_angles[i] + 90
        beta_angles.append(beta)

    beta_angles.append(beta_angles[-1] if beta_angles else 0)

    ax4.plot(angles_upper, 'r-o', label='Upper Endplate Angles')
    ax4.plot(angles_lower, 'g-o', label='Lower Endplate Angles')
    ax4.plot(angles_avg, 'm-o', label='Average Angles')
    ax4.plot(beta_angles, 'b-o', label='Expected Angles')
    ax4.set_xlabel('Point Index')
    ax4.set_ylabel('Angle (degrees)')
    ax4.set_title(f'Direction Angles (Constraint Loss: {constraint_loss:.4f})')
    ax4.legend()
    ax4.grid(True)

    plt.tight_layout()
    plt.savefig(f'{save_path}.png', dpi=200)
    plt.close()


def visualize_heatmaps(heatmaps, save_path):
    h, w = heatmaps.shape[1:]

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    axes[0, 0].imshow(heatmaps[0], cmap='jet')
    axes[0, 0].set_title("Upper Endplate Heatmap")

    upper_max_idx = np.argmax(heatmaps[0])
    upper_y, upper_x = upper_max_idx // w, upper_max_idx % w
    axes[0, 0].scatter(upper_x, upper_y, c='white', s=30, edgecolor='black')

    axes[0, 1].imshow(heatmaps[1], cmap='jet')
    axes[0, 1].set_title("Lower Endplate Heatmap")

    lower_max_idx = np.argmax(heatmaps[1])
    lower_y, lower_x = lower_max_idx // w, lower_max_idx % w
    axes[0, 1].scatter(lower_x, lower_y, c='white', s=30, edgecolor='black')

    combined = np.max(heatmaps, axis=0)
    axes[1, 0].imshow(combined, cmap='jet')
    axes[1, 0].set_title("Combined Heatmap (Max)")

    average = np.mean(heatmaps, axis=0)
    axes[1, 1].imshow(average, cmap='jet')
    axes[1, 1].set_title("Combined Heatmap (Average)")

    plt.tight_layout()
    plt.savefig(f'{save_path}.png', dpi=200)
    plt.close()