import torch
import torch.utils.data as data
import numpy as np
import os
import math
import cv2
from scipy.io import loadmat
from .utils import transform


def rearrange_pts(pts):
    boxes = []
    for k in range(0, len(pts), 4):
        pts_4 = pts[k:k + 4, :]
        x_inds = np.argsort(pts_4[:, 0])
        pt_l = np.asarray(pts_4[x_inds[:2], :])
        pt_r = np.asarray(pts_4[x_inds[2:], :])
        y_inds_l = np.argsort(pt_l[:, 1])
        y_inds_r = np.argsort(pt_r[:, 1])
        tl = pt_l[y_inds_l[0], :]
        bl = pt_l[y_inds_l[1], :]
        tr = pt_r[y_inds_r[0], :]
        br = pt_r[y_inds_r[1], :]
        boxes.append(tl)
        boxes.append(tr)
        boxes.append(bl)
        boxes.append(br)
    return np.asarray(boxes, np.float32)


class pafdata(data.Dataset):
    def __init__(self, data_dir, img_ids, phase, input_h=1536, input_w=512, down_ratio=4):
        self.img_ids = img_ids
        self.data_dir = data_dir
        self.phase = phase
        self.input_h = input_h
        self.input_w = input_w
        self.down_ratio = down_ratio
        self.num_classes = 18

        self.heatmap_height = self.input_h // self.down_ratio
        self.heatmap_width = self.input_w // self.down_ratio
        self.sigma = 10

        self.Radius = 5

        guassian_mask = torch.zeros(2 * self.Radius, 2 * self.Radius, dtype=torch.float)
        for i in range(2 * self.Radius):
            for j in range(2 * self.Radius):
                distance = np.linalg.norm([i - self.Radius, j - self.Radius])
                if distance < self.Radius:
                    guassian_mask[i][j] = math.exp(-0.5 * math.pow(distance, 2) / \
                                                   math.pow(self.Radius / 3, 2))
        self.guassian_mask = guassian_mask

    def generate_ground_truth(self, image, pts_2, image_h, image_w, img_name, cen_pts_upper, cen_pts_lower, ori_pts):
        heatmap = np.zeros((2, image_h, image_w), dtype=np.float32)

        ind = np.zeros((36), dtype=np.int64)
        vec_ind = np.zeros((36, 2), dtype=np.float32)
        reg_mask = np.zeros((36), dtype=np.uint8)

        for k in range(18):
            pts = pts_2[4 * k:4 * k + 4, :]

            upper_pts = pts[:2]
            upper_x, upper_y = np.mean(upper_pts, axis=0)
            upper_ct = np.asarray([upper_x, upper_y], dtype=np.float32)
            upper_ct_int = upper_ct.astype(np.int32)

            lower_pts = pts[2:]
            lower_x, lower_y = np.mean(lower_pts, axis=0)
            lower_ct = np.asarray([lower_x, lower_y], dtype=np.float32)
            lower_ct_int = lower_ct.astype(np.int32)

            margin_x_left = max(0, upper_ct_int[0] - self.Radius)
            margin_x_right = min(image_w, upper_ct_int[0] + self.Radius)
            margin_y_bottom = max(0, upper_ct_int[1] - self.Radius)
            margin_y_top = min(image_h, upper_ct_int[1] + self.Radius)

            y_range = margin_y_top - margin_y_bottom
            x_range = margin_x_right - margin_x_left

            if y_range > 0 and x_range > 0:
                current = heatmap[0, margin_y_bottom:margin_y_top, margin_x_left:margin_x_right]
                heatmap[0, margin_y_bottom:margin_y_top, margin_x_left:margin_x_right] = \
                    np.maximum(current, self.guassian_mask[0:y_range, 0:x_range])

            margin_x_left = max(0, lower_ct_int[0] - self.Radius)
            margin_x_right = min(image_w, lower_ct_int[0] + self.Radius)
            margin_y_bottom = max(0, lower_ct_int[1] - self.Radius)
            margin_y_top = min(image_h, lower_ct_int[1] + self.Radius)

            y_range = margin_y_top - margin_y_bottom
            x_range = margin_x_right - margin_x_left

            if y_range > 0 and x_range > 0:
                current = heatmap[1, margin_y_bottom:margin_y_top, margin_x_left:margin_x_right]
                heatmap[1, margin_y_bottom:margin_y_top, margin_x_left:margin_x_right] = \
                    np.maximum(current, self.guassian_mask[0:y_range, 0:x_range])

            upper_left = pts[0]
            upper_right = pts[1]
            upper_dis = np.sqrt((upper_right[0] - upper_left[0]) ** 2 + (upper_right[1] - upper_left[1]) ** 2)
            if upper_dis > 0:
                upper_vec = (upper_right - upper_left) / upper_dis
            else:
                upper_vec = np.array([0, 0], dtype=np.float32)

            lower_left = pts[2]
            lower_right = pts[3]
            lower_dis = np.sqrt((lower_right[0] - lower_left[0]) ** 2 + (lower_right[1] - lower_left[1]) ** 2)
            if lower_dis > 0:
                lower_vec = (lower_right - lower_left) / lower_dis
            else:
                lower_vec = np.array([0, 0], dtype=np.float32)

            ind[k] = upper_ct_int[1] * image_w + upper_ct_int[0]
            vec_ind[k] = upper_vec
            reg_mask[k] = 1

            ind[k + 18] = lower_ct_int[1] * image_w + lower_ct_int[0]
            vec_ind[k + 18] = lower_vec
            reg_mask[k + 18] = 1

        result = {
            'input': torch.from_numpy(image).float(),
            'img_name': img_name,
            'hm': torch.from_numpy(heatmap).float(),
            'ind': torch.from_numpy(ind),
            'vec_ind': torch.from_numpy(vec_ind).float(),
            'cen_pts_upper': np.array(cen_pts_upper, dtype=np.int32),
            'cen_pts_lower': np.array(cen_pts_lower, dtype=np.int32),
            'ori_pts': ori_pts,
            'reg_mask': torch.from_numpy(reg_mask),
        }
        return result

    def load_annotation(self, img_name):
        mat_name = os.path.splitext(img_name)[0] + '.mat'
        mat_path = os.path.join(self.data_dir, 'labels', mat_name)
        try:
            pts = loadmat(mat_path)['p2']
            return rearrange_pts(pts)
        except Exception as e:
            raise FileNotFoundError(f'Cannot load annotation file: {mat_path}, error: {str(e)}')

    def __getitem__(self, index):
        img_name = self.img_ids[index]

        img_path = os.path.join(self.data_dir, 'data', img_name)
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f'Cannot load image: {img_path}')

        pts = self.load_annotation(img_name)
        cen_pts_upper = []
        cen_pts_lower = []

        # 在dataset.py中的__getitem__方法中：
        if self.phase == 'train':
            datatran = transform.Compose([
                transform.ConvertImgFloat(),
                transform.RandomGammaCorrection(gamma_range=(0.8, 1.2), prob=0.3),  # 调整曝光
                transform.PhotometricDistort(),  # 基本光度变化
                transform.CLAHE(prob=0.4),  # X射线图像增强
                transform.RandomRotation(max_angle=3.0, prob=0.3),  # 小角度旋转
                transform.Expand(max_scale=1.4, prob=0.3),  # 随机扩展
                transform.RandomMirror_w(prob=0.5),  # 水平翻转
                transform.RandomGaussianBlur(prob=0.2),  # 偶尔模糊
                transform.Resize(h=self.input_h, w=self.input_w)  # 最后调整大小
            ])
        else:
            datatran = transform.Compose([
                transform.ConvertImgFloat(),
                transform.Resize(h=self.input_h, w=self.input_w)
            ])

        img_new, pts_new = datatran(img.copy(), pts.copy())
        img_new = np.clip(img_new, a_min=0., a_max=255.)
        img_new = np.transpose((img_new - 128) / 255., (2, 0, 1))
        pts_new = rearrange_pts(pts_new)

        for k in range(18):
            pts_group = pts_new[4 * k:4 * k + 4, :]

            upper_pts = pts_group[:2]
            upper_cen = np.mean(upper_pts, axis=0)
            cen_pts_upper.append(upper_cen)

            lower_pts = pts_group[2:]
            lower_cen = np.mean(lower_pts, axis=0)
            cen_pts_lower.append(lower_cen)

        pts_new = transform.rescale_pts(pts_new, down_ratio=self.down_ratio)

        dict_data = self.generate_ground_truth(
            image=img_new,
            pts_2=pts_new,
            image_h=self.input_h // self.down_ratio,
            image_w=self.input_w // self.down_ratio,
            img_name=img_name,
            cen_pts_upper=cen_pts_upper,
            cen_pts_lower=cen_pts_lower,
            ori_pts=pts
        )

        return dict_data

    def __len__(self):
        return len(self.img_ids)