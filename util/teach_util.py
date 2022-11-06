# -*- coding: UTF-8 -*-
"""
@Author: zhHuang
@Date: 2022/9/5
"""
import torch
import torch.nn.functional as F


def _hidden_loss_fn(ft, fs, loss_type: str):
    device = ft[0].device
    if loss_type == "l1":
        loss_fn = F.l1_loss
    elif loss_type == "l2":
        loss_fn = F.mse_loss
    elif loss_type == "ka":
        loss_fn = KA
    else:
        raise ValueError("loss_type must be in ['l1', 'l2', 'ka']")

    loss = torch.tensor(0, dtype=torch.float, device=device)
    for (f1, f2) in zip(ft, fs):
        loss += loss_fn(f1, f2, reduction="sum")

    return loss


def KA(X, Y, reduction):
    X_ = X.view(X.size(0), -1)
    Y_ = Y.view(Y.size(0), -1)
    assert X_.shape[0] == Y_.shape[
        0], f'X_ and Y_ must have the same shape on dim 0, but got {X_.shape[0]} for X_ and {Y_.shape[0]} for Y_.'
    X_vec = X_ @ X_.T
    Y_vec = Y_ @ Y_.T
    ret = (X_vec * Y_vec).sum() / ((X_vec**2).sum() * (Y_vec**2).sum())**0.5
    return ret


def kd_loss(z, sldj, f_t, f_s, kd_ratio, loss_fn, kd_loss_type):
    """知识蒸馏时的训练的loss为中间层知识蒸馏loss加上原始loss"""
    hidden_loss = _hidden_loss_fn(f_t, f_s, kd_loss_type) / z.shape[0]
    output_loss = loss_fn(z, sldj)
    total_loss = hidden_loss * kd_ratio + output_loss
    return hidden_loss, output_loss, total_loss
