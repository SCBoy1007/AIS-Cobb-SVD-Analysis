import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import os
import argparse
import numpy as np
from sklearn.model_selection import KFold
from tqdm import tqdm
from torch.utils.tensorboard.writer import SummaryWriter
import math
import time

from .loss import LossAll
from . import dataset
from .models import vltenet
from .visualize import visualize_epoch


def count_parameters(model):
    """计算模型的参数数量（以百万为单位）"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6


def count_model_parts(model):
    """分别计算模型骨干网络和解码器的参数数量"""
    backbone_params = sum(p.numel() for p in model.module.base_network.parameters() if p.requires_grad) / 1e6
    decoder_params = sum(p.numel() for p in model.module.Decodeing.parameters() if p.requires_grad) / 1e6
    return backbone_params, decoder_params


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default=[0, 1], nargs='+', type=int,
                        dest='gpu', help='the gpu used')
    parser.add_argument('--pretrained', default=None, type=str,
                        dest='pretrained', help='the path of pretrained model')
    parser.add_argument('--data_dir', type=str,
                        dest='data_dir', help='the path of data file')
    parser.add_argument('--cross_dir', type=str,
                        dest='cross_dir', help='the path of 5-fold cross-validation file')
    parser.add_argument('--input_h', default=1536, type=int,
                        dest='input_h', help='input_h')
    parser.add_argument('--input_w', default=512, type=int,
                        dest='input_w', help='input_w')
    parser.add_argument('--down_ratio', type=int, default=4, help='down ratio')
    parser.add_argument('--epoch', type=int, default=100, help='epoch')
    parser.add_argument('--batch_size', type=int, default=4, help='batch_size')
    parser.add_argument('--save_path', type=str, default='', help='weights to be resumed')
    parser.add_argument('--phase', type=str, default='train', help='data directory')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
    parser.add_argument('--backbone', type=str, default='hrnet18',
                        help='backbone: hrnet18, hrnet32, resnet50, resnet101, densenet121, efficientnet_b0, etc.')
    parser.add_argument('--lambda_hm', type=float, default=1, help='weight for heatmap loss')
    parser.add_argument('--lambda_vec', type=float, default=0.05, help='weight for vector loss')
    parser.add_argument('--lambda_constraint', type=float, default=0.05, help='weight for angle constraint loss')

    # 学习率相关参数
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='initial learning rate')
    parser.add_argument('--min_learning_rate', type=float, default=1e-5,
                        help='minimum learning rate, default is learning_rate * 0.05')
    parser.add_argument('--warmup_epochs', type=int, default=None,
                        help='warmup epochs, default is 5 for hrnet and 10 for resnet')
    parser.add_argument('--decay_start', type=int, default=40,
                        help='epoch to start learning rate decay, default is 40 for hrnet and 30 for resnet')
    parser.add_argument('--decay_end', type=int, default=90,
                        help='epoch to end learning rate decay (reaches min_lr), default is total epochs')

    # Transformer相关参数（如果需要）
    parser.add_argument('--use_transformer', action='store_true', help='use transformer enhancement')
    parser.add_argument('--window_height', type=int, default=16, help='window height for transformer')
    parser.add_argument('--window_width', type=int, default=16, help='window width for transformer')
    parser.add_argument('--downsample_factor', type=int, default=4, help='downsample factor for transformer')
    parser.add_argument('--trans_depth', type=int, default=4, help='transformer depth (number of layers)')
    parser.add_argument('--trans_heads', type=int, default=4, help='transformer heads (number of attention heads)')

    parser.add_argument('--debug', action='store_true', help='enable debug output')
    return parser.parse_args()


def construct_model(dropout_rate=0.1, backbone='hrnet18', use_transformer=False,
                    window_height=16, window_width=16, downsample_factor=4):
    # 构建Transformer配置（如果需要）
    if use_transformer:
        transformer_config = {
            'depth': 4,
            'num_heads': 4,
            'window_height': window_height,
            'window_width': window_width,
            'downsample_factor': downsample_factor
        }
    else:
        transformer_config = None

    # 创建模型
    model = vltenet.Vltenet(
        pretrained=True,
        final_kernel=1,
        dropout_rate=dropout_rate,
        backbone=backbone,
        use_transformer=use_transformer,
        transformer_config=transformer_config
    )
    model = nn.DataParallel(model).cuda()
    return model


def save_model(path, epoch, model):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if isinstance(model, torch.nn.DataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    data = {'epoch': epoch, 'state_dict': state_dict}
    torch.save(data, path)
    print(f"Model saved to {path}")


def validate_indices(indices, feature_size, debug=False):
    """
    验证索引是否在有效范围内，并修复超出范围的索引

    Args:
        indices: 索引张量
        feature_size: 特征图的尺寸 (H*W)
        debug: 是否打印调试信息

    Returns:
        修复后的索引张量
    """
    if indices is None:
        return None

    max_index = indices.max().item()
    if debug:
        print(f"索引范围检查: 最大索引={max_index}, 特征尺寸={feature_size}")

    if max_index >= feature_size:
        if debug:
            print(f"警告: 索引超出范围! max_index={max_index}, feature_size={feature_size}")
            over_indices = torch.nonzero(indices >= feature_size).cpu().numpy()
            print(f"超出范围的索引位置: {over_indices[:10] if len(over_indices) > 10 else over_indices}")

        # 将越界索引限制在有效范围内
        valid_indices = torch.clamp(indices, 0, feature_size - 1)
        if debug:
            print(f"已修复索引，新的最大索引: {valid_indices.max().item()}")
        return valid_indices

    return indices


class CustomScheduler:
    def __init__(self, optimizer, max_lr, min_lr, warmup_epochs, decay_start, decay_end, total_epochs):
        self.optimizer = optimizer
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_epochs = warmup_epochs
        self.decay_start = decay_start
        self.decay_end = decay_end
        self.total_epochs = total_epochs
        self.current_epoch = 0

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = max_lr * 0.1

    def step(self):
        self.current_epoch += 1
        lr = self.get_lr(self.current_epoch)

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        return lr

    def get_lr(self, epoch):
        # 预热阶段：从初始学习率*0.1线性增加到max_lr
        if epoch <= self.warmup_epochs:
            return self.max_lr * 0.1 + (self.max_lr - self.max_lr * 0.1) * (epoch / self.warmup_epochs)

        # 高原期：保持最大学习率
        elif epoch <= self.decay_start:
            return self.max_lr

        # 衰减期：使用余弦退火
        elif epoch <= self.decay_end:
            # 计算在衰减期内的进度
            progress = (epoch - self.decay_start) / (self.decay_end - self.decay_start)
            progress = min(1.0, progress)  # 确保不超过1.0
            # 余弦退火
            return self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + math.cos(math.pi * progress))

        # 稳定期：保持最小学习率
        else:
            return self.min_lr


def train_val(model, args, numi, X_train, X_test):
    print(f"[Info] Training set size: {len(X_train)}, Validation set size: {len(X_test)}")
    print(
        f"[Info] Using fixed loss weights: hm={args.lambda_hm}, vec={args.lambda_vec}, constraint={args.lambda_constraint}")
    print(f"[Info] Using dual-heatmap model for upper and lower endplates")
    print(f"[Info] Using backbone: {args.backbone}")

    log_dir = os.path.join(args.save_path, 'tensorboard_logs')
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    train_dataset = dataset.pafdata(args.data_dir, X_train, 'train',
                                    input_h=args.input_h, input_w=args.input_w,
                                    down_ratio=args.down_ratio)

    val_dataset = dataset.pafdata(args.data_dir, X_test, 'val',
                                  input_h=args.input_h, input_w=args.input_w,
                                  down_ratio=args.down_ratio)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0)

    criterion = LossAll(
        lambda_hm=args.lambda_hm,
        lambda_vec=args.lambda_vec,
        lambda_constraint=args.lambda_constraint
    )

    # 设置学习率和学习率调度器参数
    learning_rate = args.learning_rate
    min_learning_rate = args.min_learning_rate if args.min_learning_rate is not None else learning_rate * 0.05

    # 设置默认的预热epochs和衰减起点/结束点
    if args.warmup_epochs is None:
        warmup_epochs = 10 if args.backbone.startswith('resnet') else 5
    else:
        warmup_epochs = args.warmup_epochs

    if args.decay_start is None:
        decay_start = 30 if args.backbone.startswith('resnet') else 40
    else:
        decay_start = args.decay_start

    if args.decay_end is None:
        decay_end = args.epoch  # 默认到最后一个epoch结束
    else:
        decay_end = args.decay_end

    print(f"[Info] Learning rate config: max_lr={learning_rate}, min_lr={min_learning_rate}")
    print(f"[Info] Scheduler config: warmup={warmup_epochs} epochs, decay_start={decay_start}, decay_end={decay_end}")

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate * 0.1, weight_decay=5e-4)

    scheduler = CustomScheduler(
        optimizer,
        max_lr=learning_rate,
        min_lr=min_learning_rate,
        warmup_epochs=warmup_epochs,
        decay_start=decay_start,
        decay_end=decay_end,
        total_epochs=args.epoch
    )

    vis_loss_train = []
    vis_loss_val = []
    vis_loss_val_vec = []
    vis_loss_val_hm = []
    vis_loss_val_con = []

    best_val_loss = float('inf')

    for epoch in range(1, args.epoch + 1):
        model.train()
        run_train_loss = run_val_loss = 0
        run_train_losshm = run_train_lossvec = run_train_lossconstraint = 0
        run_val_losshm = run_val_lossvec = run_val_lossconstraint = 0

        train_progress_bar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]", dynamic_ncols=True, leave=True)

        for i, data_dict in enumerate(train_progress_bar):
            input_var = data_dict['input'].cuda()
            heatmap_var = data_dict['hm'].cuda()
            vec_ind_var = data_dict['vec_ind'].cuda()
            ind_var = data_dict['ind'].cuda()
            reg_mask_var = data_dict['reg_mask'].cuda()

            if args.debug and i == 0:
                print(f"Debug - 输入形状: input={input_var.shape}, hm={heatmap_var.shape}")
                print(
                    f"Debug - 向量形状: vec_ind={vec_ind_var.shape}, ind={ind_var.shape}, reg_mask={reg_mask_var.shape}")

            with torch.enable_grad():
                dec_dict = model(input_var)

                if args.debug and i == 0:
                    print(f"Debug - 输出形状: hm={dec_dict['hm'].shape}, vec_ind={dec_dict['vec_ind'].shape}")
                    if 'feature_size' in dec_dict:
                        print(f"Debug - 特征尺寸: {dec_dict['feature_size']}")
                    print(f"Debug - 峰值点形状: peak_points_upper={dec_dict['peak_points_upper'].shape}")

                # 验证并修复索引
                feature_size = dec_dict['vec_ind'].size(2) * dec_dict['vec_ind'].size(3)
                ind_var = validate_indices(ind_var, feature_size, args.debug and i == 0)

                gt_batch = {
                    'hm': heatmap_var,
                    'ind': ind_var,
                    'vec_ind': vec_ind_var,
                    'reg_mask': reg_mask_var,
                    'cen_pts_upper': data_dict['cen_pts_upper'] if 'cen_pts_upper' in data_dict else None,
                    'cen_pts_lower': data_dict['cen_pts_lower'] if 'cen_pts_lower' in data_dict else None
                }

                loss_dec, loss_hm, loss_vec, loss_constraint = criterion(dec_dict, gt_batch)
                optimizer.zero_grad()
                loss_dec.backward()
                optimizer.step()

            run_train_loss += loss_dec.item()
            run_train_losshm += loss_hm.item()
            run_train_lossvec += loss_vec.item()
            run_train_lossconstraint += loss_constraint.item()

            train_progress_bar.set_postfix({
                'loss': f"{loss_dec.item():.4f}",
                'vec': f"{loss_vec.item():.4f}",
                'hm': f"{loss_hm.item():.4f}",
                'con': f"{loss_constraint.item():.4f}",
                'lr': f"{optimizer.param_groups[0]['lr']:.8f}"
            })

        train_progress_bar.close()

        model.eval()
        val_progress_bar = tqdm(val_loader, desc=f"Epoch {epoch} [Val]", dynamic_ncols=True, leave=True)

        with torch.no_grad():
            for j, data_dict in enumerate(val_progress_bar):
                input_var = data_dict['input'].cuda()
                heatmap_var = data_dict['hm'].cuda()
                vec_ind_var = data_dict['vec_ind'].cuda()
                ind_var = data_dict['ind'].cuda()
                reg_mask_var = data_dict['reg_mask'].cuda()

                dec_dict = model(input_var)

                # 验证并修复索引
                feature_size = dec_dict['vec_ind'].size(2) * dec_dict['vec_ind'].size(3)
                ind_var = validate_indices(ind_var, feature_size, args.debug and j == 0)

                gt_batch = {
                    'hm': heatmap_var,
                    'ind': ind_var,
                    'vec_ind': vec_ind_var,
                    'reg_mask': reg_mask_var,
                    'cen_pts_upper': data_dict['cen_pts_upper'] if 'cen_pts_upper' in data_dict else None,
                    'cen_pts_lower': data_dict['cen_pts_lower'] if 'cen_pts_lower' in data_dict else None
                }

                loss_dec, loss_hm, loss_vec, loss_constraint = criterion(dec_dict, gt_batch)

                run_val_loss += loss_dec.item()
                run_val_losshm += loss_hm.item()
                run_val_lossvec += loss_vec.item()
                run_val_lossconstraint += loss_constraint.item()

                val_progress_bar.set_postfix({
                    'loss': f"{loss_dec.item():.4f}",
                    'vec': f"{loss_vec.item():.4f}",
                    'hm': f"{loss_hm.item():.4f}",
                    'con': f"{loss_constraint.item():.4f}"
                })

        val_progress_bar.close()

        try:
            visualize_epoch(model, epoch, args.save_path, device='cuda')
        except Exception as e:
            print(f"Visualization error: {e}")

        scheduler.step()

        train_loss = run_train_loss / len(train_loader)
        val_loss = run_val_loss / len(val_loader)
        val_vec_loss = run_val_lossvec / len(val_loader)
        val_hm_loss = run_val_losshm / len(val_loader)
        val_con_loss = run_val_lossconstraint / len(val_loader)
        train_vec_loss = run_train_lossvec / len(train_loader)
        train_hm_loss = run_train_losshm / len(train_loader)
        train_con_loss = run_train_lossconstraint / len(train_loader)

        vis_loss_train.append(train_loss)
        vis_loss_val.append(val_loss)
        vis_loss_val_vec.append(val_vec_loss)
        vis_loss_val_hm.append(val_hm_loss)
        vis_loss_val_con.append(val_con_loss)

        writer.add_scalar('Loss/train/total', train_loss, epoch)
        writer.add_scalar('Loss/train/vector', train_vec_loss, epoch)
        writer.add_scalar('Loss/train/heatmap', train_hm_loss, epoch)
        writer.add_scalar('Loss/train/constraint', train_con_loss, epoch)

        writer.add_scalar('Loss/val/total', val_loss, epoch)
        writer.add_scalar('Loss/val/vector', val_vec_loss, epoch)
        writer.add_scalar('Loss/val/heatmap', val_hm_loss, epoch)
        writer.add_scalar('Loss/val/constraint', val_con_loss, epoch)

        writer.add_scalar('Learning_rate', optimizer.param_groups[0]['lr'], epoch)

        print(f"Epoch: {epoch}, LR: {optimizer.param_groups[0]['lr']:.8f}")
        print(
            f"Train - Total: {train_loss:.8f}, HM: {train_hm_loss:.8f}, "
            f"Vec: {train_vec_loss:.8f}, "
            f"Constraint: {train_con_loss:.8f}"
        )
        print(
            f"Val   - Total: {val_loss:.8f}, HM: {val_hm_loss:.8f}, "
            f"Vec: {val_vec_loss:.8f}, "
            f"Constraint: {val_con_loss:.8f}"
        )

        if epoch == 1:
            writer.add_text('Hyperparameters/backbone', args.backbone, epoch)
            writer.add_text('Hyperparameters/lambda_hm', str(args.lambda_hm), epoch)
            writer.add_text('Hyperparameters/lambda_vec', str(args.lambda_vec), epoch)
            writer.add_text('Hyperparameters/lambda_constraint', str(args.lambda_constraint), epoch)
            writer.add_text('Hyperparameters/batch_size', str(args.batch_size), epoch)
            writer.add_text('Hyperparameters/learning_rate', str(learning_rate), epoch)
            writer.add_text('Hyperparameters/min_learning_rate', str(min_learning_rate), epoch)
            writer.add_text('Hyperparameters/warmup_epochs', str(warmup_epochs), epoch)
            writer.add_text('Hyperparameters/decay_start', str(decay_start), epoch)
            writer.add_text('Hyperparameters/decay_end', str(decay_end), epoch)
            writer.add_text('Hyperparameters/dropout', str(args.dropout), epoch)

        if epoch % 25 == 0:
            save_model(os.path.join(args.save_path, f'model_{numi}_{epoch}.pth'), epoch, model)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(os.path.join(args.save_path, f'model_las{numi}_best.pth'), epoch, model)
            print(f"New best model saved with validation loss: {val_loss:.8f}")

        if len(vis_loss_val_hm) > 1 and val_hm_loss < min(vis_loss_val_hm[:-1]):
            save_model(os.path.join(args.save_path, f'model_las{numi}_hm.pth'), epoch, model)

        if len(vis_loss_val_vec) > 1 and val_vec_loss < min(vis_loss_val_vec[:-1]):
            save_model(os.path.join(args.save_path, f'model_las{numi}_vec.pth'), epoch, model)

        if len(vis_loss_val_con) > 1 and val_con_loss < min(vis_loss_val_con[:-1]):
            save_model(os.path.join(args.save_path, f'model_las{numi}_con.pth'), epoch, model)

        time.sleep(0.1)

    writer.close()


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    torch.cuda.empty_cache()

    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    args = parse()

    args.data_dir = "/mnt/i/GitProjects/VTF-18V/dataset"
    args.cross_dir = "/mnt/i/GitProjects/VTF-18V/dataset/splits"

    # 构建包含退火信息的保存路径
    decay_info = f"decay{args.decay_start}-{args.decay_end}" if args.decay_end is not None else f"decay{args.decay_start}"
    loss_config = f"{args.backbone}_dual_endplate_hm{args.lambda_hm}_vec{args.lambda_vec}_cons{args.lambda_constraint}"

    # 添加Transformer信息（如果启用）
    if args.use_transformer:
        transformer_info = f"_transformer_{args.window_height}x{args.window_width}_ds{args.downsample_factor}"
    else:
        transformer_info = ""

    args.save_path = f"/mnt/i/GitProjects/VTF-18V/checkpoints_{loss_config}_{decay_info}{transformer_info}_test"

    os.makedirs(args.save_path, exist_ok=True)
    print(f"Model checkpoints will be saved to: {args.save_path}")
    print(f"Using backbone: {args.backbone}")
    print(f"Using dual-endplate model for upper and lower vertebral endplates")
    print(f"Using fixed loss weights: hm={args.lambda_hm}, vec={args.lambda_vec}, constraint={args.lambda_constraint}")

    # 如果使用CUDA_LAUNCH_BLOCKING=1可以获得更详细的错误信息
    if args.debug:
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        print("Debug模式已启用，CUDA_LAUNCH_BLOCKING=1")

    for i in range(5):
        print(f"\n开始训练 Fold {i}")

        with open(os.path.join(args.cross_dir, f'fold{i}/train.txt'), 'r') as f:
            train_ids = [line.strip() for line in f.readlines()]
        with open(os.path.join(args.cross_dir, f'fold{i}/val.txt'), 'r') as f:
            val_ids = [line.strip() for line in f.readlines()]

        model = construct_model(
            dropout_rate=args.dropout,
            backbone=args.backbone,
            use_transformer=args.use_transformer,
            window_height=args.window_height,
            window_width=args.window_width,
            downsample_factor=args.downsample_factor
        )

        # 计算并显示模型参数数量
        total_params = count_parameters(model)
        backbone_params, decoder_params = count_model_parts(model)

        print(f"\n模型参数统计信息:")
        print(f"总参数量: {total_params:.2f}M")
        print(f"骨干网络参数: {backbone_params:.2f}M ({backbone_params / total_params * 100:.1f}%)")
        print(f"解码器参数: {decoder_params:.2f}M ({decoder_params / total_params * 100:.1f}%)")
        print(f"输入尺寸: {args.input_h}x{args.input_w}, 下采样率: {args.down_ratio}")
        print(f"特征图尺寸: {args.input_h // args.down_ratio}x{args.input_w // args.down_ratio}")

        # 打印一些FLOPS和内存估计
        batch_memory = 4 * 3 * args.input_h * args.input_w * 4 / 1024 / 1024  # 假设单精度浮点数(4字节)
        print(f"估计批量为{args.batch_size}的输入内存占用: {batch_memory:.2f}MB\n")

        train_val(model, args, i, train_ids, val_ids)
