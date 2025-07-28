import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class RegL1Loss(nn.Module):
    def __init__(self):
        super(RegL1Loss, self).__init__()

    def _gather_feat(self, feat, ind, mask=None):
        dim = feat.size(2)
        ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
        feat = feat.gather(1, ind)
        if mask is not None:
            mask = mask.unsqueeze(2).expand_as(feat)
            feat = feat[mask]
            feat = feat.view(-1, dim)
        return feat

    def _tranpose_and_gather_feat(self, feat, ind):
        feat = feat.permute(0, 2, 3, 1).contiguous()
        feat = feat.view(feat.size(0), -1, feat.size(3))
        feat = self._gather_feat(feat, ind)
        return feat

    def forward(self, output, ind, target, reg_mask):
        pred = self._tranpose_and_gather_feat(output, ind)
        loss = F.l1_loss(pred, target, reduction='sum')
        loss = loss / (reg_mask.sum() + 1e-4)
        return loss


class MseWight(nn.Module):
    def __init__(self, power=50):
        super(MseWight, self).__init__()
        self.power = power

    def forward(self, pred, gt):
        if pred.size(1) != gt.size(1):
            raise ValueError(f"预测和目标的通道数不匹配: pred={pred.size()}, gt={gt.size()}")

        criterion = nn.MSELoss(reduction='none')
        loss = criterion(pred, gt)
        ratio = torch.pow(self.power, gt)
        loss = torch.mul(loss, ratio)
        loss = torch.mean(loss)
        return loss


class AngleConstraintLoss(nn.Module):
    def __init__(self, epsilon=1e-6):
        super(AngleConstraintLoss, self).__init__()
        self.thresholds = torch.tensor([
            2.0919, 1.5026, 1.6009, 2.1762, 2.3260,
            2.1743, 2.0768, 1.9951, 2.0089, 1.9652, 2.1529,
            2.5862, 2.6576, 2.5778, 2.7211, 2.5900
        ])
        self.epsilon = epsilon

    def normalize_angle(self, angle):
        return torch.atan2(torch.sin(angle), torch.cos(angle))

    def forward(self, centers, directions_upper, directions_lower):
        device = centers.device
        self.thresholds = self.thresholds.to(device)

        center_vecs = centers[:, 1:, :] - centers[:, :-1, :]

        betas = torch.atan2(center_vecs[:, :, 1], center_vecs[:, :, 0])

        thetas_upper = torch.atan2(directions_upper[:, :, 1], directions_upper[:, :, 0])
        thetas_lower = torch.atan2(directions_lower[:, :, 1], directions_lower[:, :, 0])
        thetas = (thetas_upper + thetas_lower) / 2

        total_loss = 0
        for i in range(16):
            if i < 15:
                beta_avg = (betas[:, i:i + 2].mean(dim=1) + torch.pi / 2)
            else:
                beta_avg = (betas[:, i] + torch.pi / 2)

            theta_i = thetas[:, i + 1]
            diff = self.normalize_angle(beta_avg - theta_i)
            threshold = self.thresholds[i] * torch.pi / 180

            loss_i = torch.abs(diff)
            mask = (loss_i > threshold)
            loss_i = loss_i * mask.float()

            total_loss = total_loss + loss_i.mean()

        return total_loss / 16


class LossAll(torch.nn.Module):
    def __init__(self, lambda_hm=1.0, lambda_vec=1.0, lambda_constraint=0.5):
        super(LossAll, self).__init__()
        self.L_hm = MseWight(power=50)
        self.vec = RegL1Loss()
        self.angle_constraint = AngleConstraintLoss()

        self.lambda_hm = lambda_hm
        self.lambda_vec = lambda_vec
        self.lambda_constraint = lambda_constraint

    def forward(self, pr_decs, gt_batch):
        hm_loss = self.L_hm(pr_decs['hm'], gt_batch['hm'])

        vec_loss = self.vec(pr_decs['vec_ind'], gt_batch['ind'], gt_batch['vec_ind'], gt_batch['reg_mask'])

        constraint_loss = torch.tensor(0.0, device=pr_decs['hm'].device)
        if 'peak_points' in pr_decs and self.lambda_constraint > 0:
            all_directions = self.vec._tranpose_and_gather_feat(pr_decs['vec_ind'], gt_batch['ind'])

            upper_directions = all_directions[:, :18]
            lower_directions = all_directions[:, 18:]

            constraint_loss = self.angle_constraint(
                pr_decs['peak_points'],
                upper_directions,
                lower_directions
            )

        loss_dec = (self.lambda_hm * hm_loss +
                    self.lambda_vec * vec_loss +
                    self.lambda_constraint * constraint_loss)

        return loss_dec, hm_loss, vec_loss, constraint_loss


class UncertaintyLossAll(torch.nn.Module):
    def __init__(self, init_lambda_hm=1.0, init_lambda_vec=0.05, init_lambda_constraint=0.05):
        super(UncertaintyLossAll, self).__init__()

        self.L_hm = MseWight(power=50)
        self.vec = RegL1Loss()
        self.angle_constraint = AngleConstraintLoss()

        self.log_var_hm = nn.Parameter(torch.tensor(0.0, requires_grad=True))
        self.log_var_vec = nn.Parameter(torch.tensor(0.0, requires_grad=True))
        self.log_var_constraint = nn.Parameter(torch.tensor(0.0, requires_grad=True))

        self.target_weights = [init_lambda_hm, init_lambda_vec, init_lambda_constraint]

    def get_current_weights(self):
        raw_weights = torch.stack([
            torch.exp(-self.log_var_hm),
            torch.exp(-self.log_var_vec),
            torch.exp(-self.log_var_constraint)
        ])
        normalized_weights = F.softmax(raw_weights, dim=0) * 1

        return normalized_weights[0].item(), normalized_weights[1].item(), normalized_weights[2].item()

    def forward(self, pr_decs, gt_batch):
        hm_loss = self.L_hm(pr_decs['hm'], gt_batch['hm'])
        vec_loss = self.vec(pr_decs['vec_ind'], gt_batch['ind'], gt_batch['vec_ind'], gt_batch['reg_mask'])

        constraint_loss = torch.tensor(0.0, device=pr_decs['hm'].device)
        if 'peak_points' in pr_decs:
            all_directions = self.vec._tranpose_and_gather_feat(pr_decs['vec_ind'], gt_batch['ind'])
            upper_directions = all_directions[:, :18]
            lower_directions = all_directions[:, 18:]
            constraint_loss = self.angle_constraint(
                pr_decs['peak_points'],
                upper_directions,
                lower_directions
            )

        raw_weights = torch.stack([
            torch.exp(-self.log_var_hm),
            torch.exp(-self.log_var_vec),
            torch.exp(-self.log_var_constraint)
        ])
        weights = F.softmax(raw_weights, dim=0) * 1

        weighted_hm_loss = weights[0] * hm_loss + 0.5 * self.log_var_hm
        weighted_vec_loss = weights[1] * vec_loss + 0.5 * self.log_var_vec
        weighted_constraint_loss = weights[2] * constraint_loss + 0.5 * self.log_var_constraint

        loss_dec = weighted_hm_loss + weighted_vec_loss + weighted_constraint_loss

        return loss_dec, hm_loss, vec_loss, constraint_loss