#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import random
import sys
from datetime import datetime
import open3d as o3d

import numpy as np
import torch
from PIL import Image


float_dev = torch.tensor([0], device="cuda", dtype=torch.float32)
int_dev = torch.tensor([0], device="cuda", dtype=torch.int32)
bool_dev = torch.tensor([0], device="cuda", dtype=torch.bool)


def devF(tensor: torch.Tensor):
    return tensor.type_as(float_dev)#转成float_dev类型（包括转到cuda和float32)


def devI(tensor: torch.Tensor):
    return tensor.type_as(int_dev)


def devB(tensor: torch.Tensor):
    return tensor.type_as(bool_dev)


def inverse_sigmoid(x):
    return torch.log(x / (1 - x))


def PILtoTorch(pil_image, resolution, method=Image.BILINEAR):
    resized_image_PIL = pil_image.resize(resolution, method) #将image变成resolution，method使用双线性插值
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0#颜色变到0-1之间
    if len(resized_image.shape) == 3:#为RGB图像
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)


def NPtoTorch(np_image, resolution):
    resized_image = torch.from_numpy(np_image)


def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper


def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty


def strip_symmetric(sym):
    return strip_lowerdiag(sym)


def build_rotation(r):
    norm = torch.sqrt(
        r[:, 0] * r[:, 0] + r[:, 1] * r[:, 1] + r[:, 2] * r[:, 2] + r[:, 3] * r[:, 3]
    )

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device="cuda")

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - r * z)
    R[:, 0, 2] = 2 * (x * z + r * y)
    R[:, 1, 0] = 2 * (x * y + r * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - r * x)
    R[:, 2, 0] = 2 * (x * z - r * y)
    R[:, 2, 1] = 2 * (y * z + r * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:, 0, 0] = s[:, 0]
    L[:, 1, 1] = s[:, 1]
    L[:, 2, 2] = s[:, 2]

    L = R @ L
    return L


def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
    L = build_scaling_rotation(scaling_modifier * scaling, rotation)
    actual_covariance = L @ L.transpose(1, 2)
    symm = strip_symmetric(actual_covariance)
    return symm


def safe_state(silent):
    old_f = sys.stdout#将当前标准输出保留到old_f中（后面修改sys.stdout的输出格式，即系统的输出格式)

    class F:
        def __init__(self, silent):
            self.silent = silent

        def write(self, x):
            if not self.silent:
                if x.endswith("\n"):
                    old_f.write(
                        x.replace(
                            "\n",
                            " [{}]\n".format(
                                str(datetime.now().strftime("%d/%m %H:%M:%S"))
                            ),
                        )
                    )
                else:
                    old_f.write(x)

        def flush(self):
            old_f.flush()

    sys.stdout = F(silent)

    random.seed(2024)
    np.random.seed(2024)
    torch.manual_seed(2024)
    torch.cuda.set_device(torch.device("cuda:0"))


def quaternion_from_axis_angle(axis, angle):
    axis = axis / (torch.norm(axis, p=2, dim=-1, keepdim=True) + 1e-8)
    half_angle = angle / 2
    real_part = torch.cos(half_angle).type_as(axis)
    complex_part = axis * torch.sin(half_angle).type_as(axis)
    quaternion = torch.cat([real_part, complex_part], dim=1)
    return quaternion


def is_valid_tensor(x: torch.Tensor):
    value_state = not (torch.isnan(x).any().item() or torch.isinf(x).any().item())
    grad_state = True
    if x.grad is not None:
        grad_state = not (
            torch.isnan(x.grad).any().item() or torch.isinf(x.grad).any().item()
        )
    return value_state and grad_state


def save_tensor_to_ply(save_path, xyz, voxel_size=-1):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.cpu().numpy())
    if voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    o3d.io.write_point_cloud(save_path, pcd)
