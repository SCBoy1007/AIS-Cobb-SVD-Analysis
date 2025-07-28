import torch
import math
import numpy as np
from torch.optim.lr_scheduler import _LRScheduler


class FlexibleScheduler(_LRScheduler):
    """
    灵活的学习率调度器，支持多种调度策略的组合

    Args:
        optimizer: 优化器
        warmup_epochs: 预热阶段的epoch数
        warmup_factor: 预热开始时的学习率系数 (相对于base_lr)
        plateau_epochs: 高原期的epoch数 (保持最大学习率)
        decay_type: 衰减类型，支持 'cosine', 'linear', 'step', 'exp', 'poly'
        decay_factor: 用于step和exp衰减的衰减系数
        decay_epochs: step衰减的步长
        min_lr_factor: 最小学习率系数 (相对于base_lr)
        cycle: 是否在衰减后使用循环策略
        total_epochs: 总的训练epoch数
        last_epoch: 上一个epoch的索引
    """

    def __init__(self, optimizer, warmup_epochs=5, warmup_factor=0.1,
                 plateau_epochs=40, decay_type='cosine', decay_factor=0.1,
                 decay_epochs=30, min_lr_factor=0.05, cycle=False,
                 total_epochs=100, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.warmup_factor = warmup_factor
        self.plateau_epochs = plateau_epochs
        self.decay_type = decay_type
        self.decay_factor = decay_factor
        self.decay_epochs = decay_epochs
        self.min_lr_factor = min_lr_factor
        self.cycle = cycle
        self.total_epochs = total_epochs

        # 保存基础学习率
        self.base_lrs = None
        super(FlexibleScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            print("警告: 请在optimizer.step()之后调用scheduler.step()")

        if self.base_lrs is None:
            self.base_lrs = [group['lr'] / self.warmup_factor for group in self.optimizer.param_groups]

        # 预热阶段
        if self.last_epoch < self.warmup_epochs:
            factor = self.warmup_factor + (1.0 - self.warmup_factor) * (self.last_epoch / self.warmup_epochs)
            return [base_lr * factor for base_lr in self.base_lrs]

        # 高原期 - 保持最大学习率
        if self.last_epoch < self.warmup_epochs + self.plateau_epochs:
            return self.base_lrs

        # 计算衰减阶段的进度
        decay_epochs = self.total_epochs - self.warmup_epochs - self.plateau_epochs
        decay_progress = (self.last_epoch - self.warmup_epochs - self.plateau_epochs) / decay_epochs

        # 如果启用了循环，计算周期进度
        if self.cycle:
            decay_progress = decay_progress % 1.0

        # 根据不同衰减类型计算衰减因子
        if self.decay_type == 'cosine':
            # 余弦衰减
            factor = self.min_lr_factor + 0.5 * (1.0 - self.min_lr_factor) * (1 + math.cos(math.pi * decay_progress))

        elif self.decay_type == 'linear':
            # 线性衰减
            factor = 1.0 - (1.0 - self.min_lr_factor) * decay_progress

        elif self.decay_type == 'step':
            # 阶梯式衰减
            step_index = int(decay_progress * decay_epochs / self.decay_epochs)
            factor = self.decay_factor ** step_index
            factor = max(factor, self.min_lr_factor)

        elif self.decay_type == 'exp':
            # 指数衰减
            factor = self.decay_factor ** decay_progress
            factor = max(factor, self.min_lr_factor)

        elif self.decay_type == 'poly':
            # 多项式衰减
            factor = self.min_lr_factor + (1.0 - self.min_lr_factor) * (1 - decay_progress) ** 2

        else:
            raise ValueError(f"不支持的衰减类型: {self.decay_type}")

        return [base_lr * factor for base_lr in self.base_lrs]


class WarmupCosineAnnealingLR(_LRScheduler):
    """
    带有预热的余弦退火学习率调度器，是FlexibleScheduler的简化版本

    Args:
        optimizer: 优化器
        warmup_epochs: 预热阶段的epoch数
        warmup_start_factor: 预热开始时的学习率系数
        total_epochs: 总的训练epoch数
        min_lr_factor: 最小学习率系数
        last_epoch: 上一个epoch的索引
    """

    def __init__(self, optimizer, warmup_epochs=5, warmup_start_factor=0.1,
                 total_epochs=100, min_lr_factor=0.05, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.warmup_start_factor = warmup_start_factor
        self.total_epochs = total_epochs
        self.min_lr_factor = min_lr_factor
        super(WarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # 预热阶段
            alpha = self.last_epoch / self.warmup_epochs
            factor = self.warmup_start_factor + (1.0 - self.warmup_start_factor) * alpha
        else:
            # 余弦退火阶段
            progress = (self.last_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            progress = min(1.0, progress)
            factor = self.min_lr_factor + 0.5 * (1.0 - self.min_lr_factor) * (1 + math.cos(math.pi * progress))

        return [base_lr * factor for base_lr in self.base_lrs]


class LinearWarmupCosineAnnealingLR(_LRScheduler):
    """
    带有线性预热的余弦退火学习率调度器

    Args:
        optimizer: 优化器
        warmup_epochs: 预热阶段的epoch数
        max_epochs: 总的训练epoch数
        warmup_start_lr: 预热开始的学习率
        eta_min: 最小学习率
        last_epoch: 上一个epoch的索引
    """

    def __init__(self, optimizer, warmup_epochs=5, max_epochs=100,
                 warmup_start_lr=0, eta_min=0, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min
        super(LinearWarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # 线性预热
            return [self.warmup_start_lr + (base_lr - self.warmup_start_lr) *
                    (self.last_epoch / self.warmup_epochs) for base_lr in self.base_lrs]
        else:
            # 余弦退火
            progress = (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            progress = min(1.0, progress)
            return [self.eta_min + (base_lr - self.eta_min) *
                    0.5 * (1 + math.cos(math.pi * progress)) for base_lr in self.base_lrs]


# 从CSV学习率数据创建定制的学习率表
def create_lr_schedule_from_csv(csv_file, min_lr=None, max_lr=None, smoothing=0):
    """
    从CSV文件创建学习率调度表

    Args:
        csv_file: CSV文件路径，包含step和学习率
        min_lr: 最小学习率 (可选，用于规范化)
        max_lr: 最大学习率 (可选，用于规范化)
        smoothing: 平滑窗口大小

    Returns:
        学习率列表
    """
    import pandas as pd

    df = pd.read_csv(csv_file)
    lr_values = df['Value'].values

    # 应用平滑处理
    if smoothing > 0:
        window = np.ones(smoothing) / smoothing
        lr_values = np.convolve(lr_values, window, mode='same')

    # 规范化学习率 (如果提供了min_lr和max_lr)
    if min_lr is not None and max_lr is not None:
        lr_min, lr_max = lr_values.min(), lr_values.max()
        lr_values = min_lr + (max_lr - min_lr) * (lr_values - lr_min) / (lr_max - lr_min)

    return lr_values.tolist()


class CSVScheduler(_LRScheduler):
    """
    基于预定义学习率表的调度器

    Args:
        optimizer: 优化器
        lr_list: 学习率列表
        repeat: 当epoch超过列表长度时是否重复
        last_epoch: 上一个epoch的索引
    """

    def __init__(self, optimizer, lr_list, repeat=False, last_epoch=-1):
        self.lr_list = lr_list
        self.repeat = repeat
        super(CSVScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        idx = self.last_epoch
        if self.repeat:
            idx = idx % len(self.lr_list)
        else:
            idx = min(idx, len(self.lr_list) - 1)

        return [self.lr_list[idx] for _ in self.optimizer.param_groups]