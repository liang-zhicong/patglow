import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.glow.act_norm import ActNorm
from models.glow.coupling import Coupling
from models.glow.inv_conv import InvConv


class Glow_kd(nn.Module):
    """Glow Model

    Based on the paper:
    "Glow: Generative Flow with Invertible 1x1 Convolutions"
    by Diederik P. Kingma, Prafulla Dhariwal
    (https://arxiv.org/abs/1807.03039).

    Args:
        num_channels (int): Number of channels in middle convolution of each
            step of flow.
        num_levels (int): Number of levels in the entire model.
        num_steps (int): Number of steps of flow for each level.
    """
    def __init__(self, is_teacher, num_channels, num_levels, num_steps, another_steps, hint_opt="all"):
        # assert hint_opt in ["all", "mid"], "hint_opt must be in ['all', 'mid', 'single','double']!"
        super(Glow_kd, self).__init__()

        # Use bounds to rescale images before converting to logits, not learned
        self.register_buffer('bounds', torch.tensor([0.9], dtype=torch.float32)) #bounds=0.9

        # 教师网络hint layers的索引。如果student_step=0的话，则保存所有层的特征图，学生网络将此参数设置为0
        if hint_opt == "all":
            self.hint_idx = np.around(np.linspace(0, num_steps-1, another_steps)) if is_teacher else np.array(range(0, num_steps))
        elif hint_opt == "mid":
            self.hint_idx = np.array([num_steps // 2])
        else:
            step_temp = num_steps if is_teacher else another_steps
            t_idx = np.delete(np.around(np.linspace(0, step_temp-1, int(hint_opt)+2)), [0, -1])
            self.hint_idx = t_idx if is_teacher else \
                np.delete(np.around(np.linspace(0, num_steps+t_idx[-1]-another_steps, int(hint_opt)+1)), 0)

        self.flows = _Glow(in_channels=4 * 3,  # RGB image after squeeze
                           mid_channels=num_channels,
                           num_levels=num_levels,
                           num_steps=num_steps,
                           hint_idx=self.hint_idx)

    def forward(self, x, reverse=False):
        if reverse:
            sldj = torch.zeros(x.size(0), device=x.device)
        else:
            # Expect inputs in [0, 1]
            if x.min() < 0 or x.max() > 1:
                raise ValueError('Expected x in [0, 1], got min/max {}/{}'
                                 .format(x.min(), x.max()))
            # De-quantize and convert to logits
            x, sldj = self._pre_process(x)

        x = squeeze(x)
        if reverse:
            x, sldj = self.flows(x, sldj, reverse)
            x = squeeze(x, reverse=True)
            return x, sldj
        else:
            x, sldj, hint_list = self.flows(x, sldj, reverse)
            x = squeeze(x, reverse=True)
            return x, sldj, hint_list

    def _pre_process(self, x):
        """Dequantize the input image `x` and convert to logits.

        See Also:
            - Dequantization: https://arxiv.org/abs/1511.01844, Section 3.1
            - Modeling logits: https://arxiv.org/abs/1605.08803, Section 4.1

        Args:
            x (torch.Tensor): Input image.

        Returns:
            y (torch.Tensor): Dequantized logits of `x`.
        """
        y = (x * 255. + torch.rand_like(x)) / 256.
        y = (2 * y - 1) * self.bounds
        y = (y + 1) / 2#scale to [0.05,0.95]
        y = y.log() - (1. - y).log()#ln(y/(1-y))

        # Save log-determinant of Jacobian of initial transform
        ldj = F.softplus(y) + F.softplus(-y) \
            - F.softplus((1. - self.bounds).log() - self.bounds.log())
        sldj = ldj.flatten(1).sum(-1)

        return y, sldj

    def pre_process(self,x):
        z = x + torch.normal(mean=torch.zeros_like(x),
                             std=torch.ones_like(x)*(1. / 256.))
        logdet = torch.zeros_like(x[:,0,0,0])
        logdet += float(-np.log(256.)*x.size(2)*x.size(3))
        return z,logdet

class _Glow(nn.Module):
    """Recursive constructor for a Glow model. Each call creates a single level.

    Args:
        in_channels (int): Number of channels in the input.
        mid_channels (int): Number of channels in hidden layers of each step.
        num_levels (int): Number of levels to construct. Counter for recursion.
        num_steps (int): Number of steps of flow for each level.
    """
    def __init__(self, in_channels, mid_channels, num_levels, num_steps, hint_idx):
        super(_Glow, self).__init__()
        self.steps = nn.ModuleList([_FlowStep(in_channels=in_channels,
                                              mid_channels=mid_channels)
                                    for _ in range(num_steps)])

        self.hint_idx = hint_idx

        if num_levels > 1:
            self.next = _Glow(in_channels=2 * in_channels,
                              mid_channels=mid_channels,
                              num_levels=num_levels - 1,
                              num_steps=num_steps,
                              hint_idx=hint_idx)
        else:
            self.next = None

    def forward(self, x, sldj, reverse=False):
        hint_feature = []
        if not reverse:
            for idx, step in enumerate(self.steps):
                x, sldj = step(x, sldj, reverse)
                if idx in self.hint_idx:
                    hint_feature.append(x)
        if self.next is not None:
            x = squeeze(x)
            x, x_split = x.chunk(2, dim=1)
            if reverse:
                x, sldj = self.next(x, sldj, reverse)
            else:
                x, sldj, f = self.next(x, sldj, reverse)
                hint_feature.extend(f)
            x = torch.cat((x, x_split), dim=1)
            x = squeeze(x, reverse=True)

        if reverse:
            for step in reversed(self.steps):
                x, sldj = step(x, sldj, reverse)
            return x, sldj

        return x, sldj, hint_feature


class _FlowStep(nn.Module):
    def __init__(self, in_channels, mid_channels):
        super(_FlowStep, self).__init__()

        # Activation normalization, invertible 1x1 convolution, affine coupling
        self.norm = ActNorm(in_channels, return_ldj=True)
        self.conv = InvConv(in_channels)
        self.coup = Coupling(in_channels // 2, mid_channels)

    def forward(self, x, sldj=None, reverse=False):
        if reverse:
            x, sldj = self.coup(x, sldj, reverse)
            x, sldj = self.conv(x, sldj, reverse)
            x, sldj = self.norm(x, sldj, reverse)
        else:
            x, sldj = self.norm(x, sldj, reverse)
            x, sldj = self.conv(x, sldj, reverse)
            x, sldj = self.coup(x, sldj, reverse)

        return x, sldj


def squeeze(x, reverse=False):
    """Trade spatial extent for channels. In forward direction, convert each
    1x4x4 volume of input into a 4x1x1 volume of output.

    Args:
        x (torch.Tensor): Input to squeeze or unsqueeze.
        reverse (bool): Reverse the operation, i.e., unsqueeze.

    Returns:
        x (torch.Tensor): Squeezed or unsqueezed tensor.
    """
    b, c, h, w = x.size()
    if reverse:
        x = x.view(b, c // 4, 2, 2, h, w)
        x = x.permute(0, 1, 4, 2, 5, 3).contiguous()#change dimension
        x = x.view(b, c // 4, h * 2, w * 2)
    else:
        x = x.view(b, c, h // 2, 2, w // 2, 2)
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
        x = x.view(b, c * 2 * 2, h // 2, w // 2)
    return x
