import numpy as np
from numpy import random
import cv2
import math


def rescale_pts(pts, down_ratio):
    """将坐标点按照下采样比例缩放"""
    return np.asarray(pts, np.float32) / float(down_ratio)


def outlier_rejection(pts, num_vertebrae=18):
    """
    去除异常椎体点

    Args:
        pts: 形状为[N, 12]的点集，N为椎体数量
        num_vertebrae: 期望的椎体数量

    Returns:
        形状为[num_vertebrae, 12]的过滤后点集
    """
    remained_pts = []
    for i, p in enumerate(pts):
        # 当前椎体中心x坐标
        cur_ver_center_x = p[0]
        # 当前椎体宽度
        cur_ver_width = abs(p[8] - p[2])

        # 处理第一个椎体
        if i == 0:
            if len(pts) > 2:
                next_ver_center_x = pts[i + 1][0]
                next_next_ver_center_x = pts[i + 2][0]

                # 如果当前椎体与后两个椎体的距离都大于一半椎体宽度，可能是异常值
                if abs(next_ver_center_x - cur_ver_center_x) > (cur_ver_width / 2) and abs(
                        next_next_ver_center_x - cur_ver_center_x) > (cur_ver_width / 2):
                    pass
                else:
                    remained_pts.append(p)
            else:
                remained_pts.append(p)

        # 处理最后一个椎体
        elif i == len(pts) - 1:
            if len(pts) > 2:
                pre_ver_center_x = pts[i - 1][0]
                pre_pre_ver_center_x = pts[i - 2][0]

                # 如果当前椎体与前两个椎体的距离都大于一半椎体宽度，可能是异常值
                if abs(pre_ver_center_x - cur_ver_center_x) > (cur_ver_width / 2) and abs(
                        pre_pre_ver_center_x - cur_ver_center_x) > (cur_ver_width / 2):
                    pass
                else:
                    remained_pts.append(p)
            else:
                remained_pts.append(p)

        # 处理中间椎体
        else:
            pre_ver_center_x = pts[i - 1][0]
            next_ver_center_x = pts[i + 1][0]

            # 如果当前椎体与前后椎体的距离都大于一半椎体宽度，可能是异常值
            if abs(pre_ver_center_x - cur_ver_center_x) > (cur_ver_width / 2) and abs(
                    next_ver_center_x - cur_ver_center_x) > (cur_ver_width / 2):
                pass
            else:
                remained_pts.append(p)

    # 如果剩余椎体数量少于期望数量，通过复制最后几个椎体补齐
    if len(remained_pts) < num_vertebrae:
        missing_number = num_vertebrae - len(remained_pts)
        print(f'[WARNING] number of vertebra less than {num_vertebrae}! missing numbers is {missing_number}')

        # 更智能的填充策略：如果有足够的椎体，则复制最后missing_number个椎体
        # 否则，重复复制最后一个椎体直到达到所需数量
        if len(remained_pts) > missing_number:
            remained_pts.extend(remained_pts[-missing_number:])
        else:
            for _ in range(missing_number):
                remained_pts.append(remained_pts[-1].copy())

    remained_pts = np.array(remained_pts)
    return remained_pts


def resize_img(src, dst_wh, interpolation=cv2.INTER_CUBIC):
    """
    调整图像大小并保持宽高比，在必要时进行填充

    Args:
        src: 输入图像
        dst_wh: 目标尺寸(宽度,高度)
        interpolation: 插值方法

    Returns:
        调整大小后的图像和变换记录(左偏移,上偏移,缩放比例)
    """
    # 检查输入图像维度
    if len(src.shape) == 3:
        sh, sw = src.shape[:2]
        channels = 3
    else:
        sh, sw = src.shape
        src = src[:, :, np.newaxis]
        channels = 1

    dw, dh = dst_wh

    # 计算缩放比例，保持宽高比
    ratio_src = sw / sh
    ratio_dst = dw / dh

    if ratio_src >= ratio_dst:
        resize_ratio = dw / sw
        nw = dw
        nh = int(sh * resize_ratio)
    else:
        resize_ratio = dh / sh
        nw = int(sw * resize_ratio)
        nh = dh

    # 调整大小
    if channels == 1:
        resized_img = cv2.resize(src[:, :, 0], (nw, nh), interpolation=interpolation)
        resized_img = resized_img[:, :, np.newaxis]
    else:
        resized_img = cv2.resize(src, (nw, nh), interpolation=interpolation)

    # 创建目标尺寸画布并居中粘贴
    result = np.zeros((dh, dw, channels), dtype=src.dtype)
    left = (dw - nw) // 2
    top = (dh - nh) // 2

    result[top:top + nh, left:left + nw, :] = resized_img

    # 如果原始图像是单通道，则转换结果为三通道
    if channels == 1:
        tmp = np.zeros((dh, dw, 3), dtype=src.dtype)
        tmp[:, :, 0] = result[:, :, 0]
        tmp[:, :, 1] = result[:, :, 0]
        tmp[:, :, 2] = result[:, :, 0]
        result = tmp

    resize_record = (left, top, resize_ratio)
    return result, resize_record


def resize_pt(xy, resize_record):
    """
    根据图像缩放记录调整坐标点

    Args:
        xy: 坐标点数组
        resize_record: 缩放记录(left, top, ratio)

    Returns:
        调整后的坐标点
    """
    left, top, ratio = resize_record
    xy_new = xy.copy()  # 创建副本，避免修改输入
    xy_new[:, 0] = xy[:, 0] * ratio + left
    xy_new[:, 1] = xy[:, 1] * ratio + top
    return xy_new


def fix_vertebrae_area(img, pts, padding_top=20, padding_bottom=30):
    """
    裁剪图像，只保留包含椎体的区域（加一些内边距）

    Args:
        img: 输入图像
        pts: 椎体坐标点
        padding_top: 顶部额外保留像素
        padding_bottom: 底部额外保留像素

    Returns:
        裁剪后的图像
    """
    y = pts[:, 1]
    ymin = int(np.min(y)) - padding_top
    ymax = int(np.max(y)) + padding_bottom

    if ymin < 0:
        ymin = 0
    if ymax > img.shape[0]:
        ymax = img.shape[0]

    # 清除上部无用区域
    img[0:ymin, :, :] = 0
    # 裁剪图像
    crop = img[:ymax, :, :]
    return crop


class Compose(object):
    """组合多个变换"""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, pts):
        for t in self.transforms:
            img, pts = t(img, pts)
        return img, pts


class ConvertImgFloat(object):
    """将图像和点转换为float32类型"""

    def __call__(self, img, pts):
        return img.astype(np.float32), pts.astype(np.float32)


class RandomContrast(object):
    """随机调整对比度"""

    def __init__(self, lower=0.5, upper=1.5, prob=0.5):
        self.lower = lower
        self.upper = upper
        self.prob = prob
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, img, pts):
        if random.random() < self.prob:
            alpha = random.uniform(self.lower, self.upper)
            img = img * alpha
        return img, pts


class RandomBrightness(object):
    """随机调整亮度"""

    def __init__(self, delta=32, prob=0.5):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta
        self.prob = prob

    def __call__(self, img, pts):
        if random.random() < self.prob:
            delta = random.uniform(-self.delta, self.delta)
            img = img + delta
        return img, pts


class RandomLightingNoise(object):
    """随机调整RGB通道顺序"""

    def __init__(self, prob=0.5):
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))
        self.prob = prob

    def __call__(self, img, pts):
        if random.random() < self.prob:
            swap = self.perms[random.randint(len(self.perms))]
            shuffle = SwapChannels(swap)
            img = shuffle(img)
        return img, pts


class SwapChannels(object):
    """交换图像的RGB通道"""

    def __init__(self, swaps):
        self.swaps = swaps

    def __call__(self, image):
        image = image[:, :, self.swaps]
        return image


class PhotometricDistort(object):
    """光度失真：组合多种亮度、对比度和颜色扭曲"""

    def __init__(self, contrast_prob=0.5, brightness_prob=0.5, lighting_prob=0.5):
        self.pd = RandomContrast(prob=contrast_prob)
        self.rb = RandomBrightness(prob=brightness_prob)
        self.rln = RandomLightingNoise(prob=lighting_prob)

    def __call__(self, img, pts):
        img, pts = self.rb(img, pts)
        if random.random() < 0.5:
            distort = self.pd
        else:
            distort = self.pd
        img, pts = distort(img, pts)
        img, pts = self.rln(img, pts)
        return img, pts


class Expand(object):
    """随机扩展图像，给图像添加边距"""

    def __init__(self, max_scale=1.5, mean=(0.5, 0.5, 0.5), prob=0.5):
        self.mean = mean
        self.max_scale = max_scale
        self.prob = prob

    def __call__(self, img, pts):
        if random.random() >= self.prob:
            return img, pts

        h, w, c = img.shape
        ratio = random.uniform(1, self.max_scale)
        y1 = random.uniform(0, h * ratio - h)
        x1 = random.uniform(0, w * ratio - w)

        # 确保所有点在扩展后图像内部
        if np.max(pts[:, 0]) + int(x1) > w - 1 or np.max(pts[:, 1]) + int(y1) > h - 1:
            return img, pts
        else:
            expand_img = np.zeros(shape=(int(h * ratio), int(w * ratio), c), dtype=img.dtype)
            expand_img[:, :, :] = self.mean
            expand_img[int(y1):int(y1 + h), int(x1):int(x1 + w)] = img
            pts[:, 0] += int(x1)
            pts[:, 1] += int(y1)
            return expand_img, pts


class RandomMirror_w(object):
    """随机水平翻转图像"""

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, img, pts):
        h, w, _ = img.shape
        if random.random() < self.prob:
            img = img[:, ::-1, :]
            pts[:, 0] = w - pts[:, 0]
        return img, pts


class RandomMirror_h(object):
    """随机垂直翻转图像（脊柱图像通常不建议使用）"""

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, img, pts):
        h, _, _ = img.shape
        if random.random() < self.prob:
            img = img[::-1, :, :]
            pts[:, 1] = h - pts[:, 1]
        return img, pts


class RandomRotation(object):
    """随机旋转图像（小角度）"""

    def __init__(self, max_angle=5.0, prob=0.5):
        self.max_angle = max_angle
        self.prob = prob

    def __call__(self, img, pts):
        if random.random() >= self.prob:
            return img, pts

        h, w, _ = img.shape
        center = (w // 2, h // 2)
        angle = random.uniform(-self.max_angle, self.max_angle)

        # 旋转矩阵
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        # 旋转点坐标
        pts_new = pts.copy()
        for i in range(len(pts)):
            x, y = pts[i]
            # 应用旋转变换
            new_x = M[0, 0] * x + M[0, 1] * y + M[0, 2]
            new_y = M[1, 0] * x + M[1, 1] * y + M[1, 2]
            pts_new[i] = [new_x, new_y]

        return rotated_img, pts_new


class RandomGaussianBlur(object):
    """随机高斯模糊"""

    def __init__(self, kernel_size=5, sigma_range=(0.1, 1.5), prob=0.3):
        self.kernel_size = kernel_size
        self.sigma_min, self.sigma_max = sigma_range
        self.prob = prob

    def __call__(self, img, pts):
        if random.random() >= self.prob:
            return img, pts

        sigma = random.uniform(self.sigma_min, self.sigma_max)
        blurred_img = cv2.GaussianBlur(img, (self.kernel_size, self.kernel_size), sigma)
        return blurred_img, pts


class RandomGammaCorrection(object):
    """随机伽马校正，调整图像明暗度"""

    def __init__(self, gamma_range=(0.8, 1.2), prob=0.3):
        self.gamma_min, self.gamma_max = gamma_range
        self.prob = prob

    def __call__(self, img, pts):
        if random.random() >= self.prob:
            return img, pts

        gamma = random.uniform(self.gamma_min, self.gamma_max)
        # 标准化到0-1范围
        img_norm = img / 255.0
        # 应用伽马校正
        img_gamma = np.power(img_norm, gamma) * 255.0
        # 确保值在合法范围
        img_gamma = np.clip(img_gamma, 0, 255)

        return img_gamma, pts


class Resize(object):
    """调整图像大小到指定尺寸"""

    def __init__(self, h, w, interpolation=cv2.INTER_CUBIC):
        self.dsize = (w, h)
        self.interpolation = interpolation

    def __call__(self, img, pts):
        img, resize_record = resize_img(img, self.dsize, self.interpolation)
        pts = resize_pt(pts, resize_record)
        return img, np.asarray(pts)


class CropVertebraeArea(object):
    """裁剪只包含椎体的区域"""

    def __init__(self, padding_top=20, padding_bottom=30):
        self.padding_top = padding_top
        self.padding_bottom = padding_bottom

    def __call__(self, img, pts):
        crop_img = fix_vertebrae_area(img, pts, self.padding_top, self.padding_bottom)
        return crop_img, pts


class CLAHE(object):
    """应用对比度受限自适应直方图均衡化"""

    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8), prob=0.5):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.prob = prob

    def __call__(self, img, pts):
        if random.random() >= self.prob:
            return img, pts

        # 转换为灰度图像
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        else:
            gray = img.astype(np.uint8)

        # 应用CLAHE
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        enhanced = clahe.apply(gray)

        # 转换回BGR
        if len(img.shape) == 3:
            enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
            # 混合原始色彩和增强后的亮度
            hsv = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2HSV)
            hsv[..., 2] = enhanced
            result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).astype(img.dtype)
        else:
            result = enhanced.astype(img.dtype)

        return result, pts