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

import numpy as np
from PIL import Image

from scene.cameras import Camera
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal

WARNED = False


def loadCam(args, id, cam_info, resolution_scale, detct_res=None):
    orig_w, orig_h = cam_info.image.size
    preload = args.preload#false
    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w / (resolution_scale * args.resolution)), round(
            orig_h / (resolution_scale * args.resolution)
        )
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print(
                        "[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1"
                    )
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    resized_image_rgb = PILtoTorch(cam_info.image, resolution)#获得resize后的rgb图像
    resized_image_depth = PILtoTorch(cam_info.depth, resolution, Image.NEAREST)#获得resize后的depth图像
    gt_image = resized_image_rgb[:3, ...]#gt_image和resized_image_rgb都为[3,480,640]
    gt_depth = resized_image_depth
    loaded_mask = None
    if resized_image_rgb.shape[1] == 4:
        loaded_mask = resized_image_rgb[3:4, ...]

    #?语义信息，加载语义
    if args.use_semantics:
        resized_image_semantics = PILtoTorch(cam_info.semantics, resolution, Image.NEAREST)
        # resized_image_id = PILtoTorch(cam_info.semantics_id, resolution, Image.NEAREST)
        gt_semantics = resized_image_semantics
        # gt_semantics_id = resized_image_id
    else:
        gt_semantics = None
        # gt_semantics_id = None
    # !待写
    #2024-11-20 加入物体的真值的图像
    if args.use_object:
        gt_object = PILtoTorch(cam_info.object_img,resolution,Image.NEAREST)
        pass

    else:
        gt_object = None


    return Camera(
        colmap_id=cam_info.uid,
        R=cam_info.R,
        T=cam_info.T,
        FoVx=cam_info.FovX,
        FoVy=cam_info.FovY,
        image=gt_image,
        depth=gt_depth,
        gt_alpha_mask=loaded_mask,
        image_name=cam_info.image_name,
        uid=id,
        data_device=args.data_device,
        pose_gt=cam_info.pose_gt,
        cx=cam_info.cx / resolution_scale,
        cy=cam_info.cy / resolution_scale,
        timestamp=cam_info.timestamp,
        preload=preload,
        depth_scale=cam_info.depth_scale,
        #?语义信息
        semantics=gt_semantics if args.use_semantics else None,
        # semantics_id=gt_semantics_id if args.use_semantics else None,
        dect_res=detct_res, #2024-11-19 加入物体检测结果
        object_img=gt_object #2024-11-20 加入物体的真值的图像
    )


def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list


def camera_to_JSON(id, camera: Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        "id": id,
        "img_name": camera.image_name,
        "width": camera.width,
        "height": camera.height,
        "position": pos.tolist(),
        "rotation": serializable_array_2d,
        "fy": fov2focal(camera.FovY, camera.height),
        "fx": fov2focal(camera.FovX, camera.width),
    }
    return camera_entry
