import numpy as np
import math
import torch

from scene.cameras import Camera
from SLAM.utils import devF, devI

from diff_gaussian_rasterization_depth import (
    GaussianRasterizationSettings as GaussianRasterizationSettings_depth,
)
from diff_gaussian_rasterization_depth import (
    GaussianRasterizer as GaussianRasterizer_depth,
)

from utils.general_utils import (
    build_covariance_from_scaling_rotation,
    inverse_sigmoid
)


class Renderer:
    def setup_functions(self):
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, args):
        self.raster_settings = None
        self.rasterizer = None
        self.bg_color = devF(torch.tensor([0, 0, 0]))
        self.renderer_opaque_threshold = args.renderer_opaque_threshold#0.6
        self.renderer_normal_threshold = np.cos(
            np.deg2rad(args.renderer_normal_threshold)#计算余弦值，用于判断采用哪种深度策略
        )
        self.scaling_modifier = 1.0
        self.renderer_depth_threshold = args.renderer_depth_threshold#1.0
        self.max_sh_degree = args.max_sh_degree#3
        self.color_sigma = args.color_sigma#3.0
        if args.active_sh_degree < 0:
            self.active_sh_degree = self.max_sh_degree
        else:
            self.active_sh_degree = args.active_sh_degree
        self.setup_functions()

    def get_scaling(self, scaling):
        return self.scaling_activation(scaling)

    def get_rotation(self, rotaion):
        return self.rotation_activation(rotaion)

    def get_covariance(self, scaling, rotaion, scaling_modifier=1):
        return self.covariance_activation(scaling, scaling_modifier, rotaion)

    #2024-11-21 对物体进行渲染
    def render_obj(
            self,
            viewpoint_camera: Camera,
            gaussian_data,
            tile_mask=None,
    ):
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
        self.raster_settings = GaussianRasterizationSettings_depth(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=self.bg_color,
            scale_modifier=self.scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=self.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            opaque_threshold=self.renderer_opaque_threshold,  # ?新加
            depth_threshold=self.renderer_depth_threshold,  # ?新加
            normal_threshold=self.renderer_normal_threshold,  # ?新加的
            color_sigma=self.color_sigma,  # !是什么
            prefiltered=False,
            debug=False,
            cx=viewpoint_camera.cx,  # ?新加的
            cy=viewpoint_camera.cy,  # ?新加的
            T_threshold=0.0001,  # ?新加的
        )
        self.rasterizer = GaussianRasterizer_depth(
            raster_settings=self.raster_settings
        )
        # 把stable和unstable存起来
        means3D = gaussian_data["xyz"]
        opacity = gaussian_data["opacity"]
        scales = gaussian_data["scales"]
        rotations = gaussian_data["rotations"]
        shs = None
        normal = None
        obj_color = gaussian_data["obj_color"]
        colors_precomp = obj_color
        cov3D_precomp = None

        # 给cuda分块
        if tile_mask is None:
            tile_mask = devI(
                torch.ones(
                    (viewpoint_camera.image_height + 15) // 16,
                    (viewpoint_camera.image_width + 15) // 16,
                    dtype=torch.int32,
                )
            )

        render_results = self.rasterizer(
            means3D=means3D,
            opacities=opacity,
            shs=shs,
            colors_precomp=colors_precomp,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp,
            normal_w=None,
            tile_mask=tile_mask,
        )

        rendered_obj = render_results[0]

        results = {
            "render_obj": rendered_obj,
        }

        return results

    def render(
        self,
        viewpoint_camera: Camera,
        gaussian_data,
        tile_mask=None,
    ):
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
        self.raster_settings = GaussianRasterizationSettings_depth(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=self.bg_color,
            scale_modifier=self.scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=self.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            opaque_threshold=self.renderer_opaque_threshold,#?新加
            depth_threshold=self.renderer_depth_threshold,#?新加
            normal_threshold=self.renderer_normal_threshold,#?新加的
            color_sigma=self.color_sigma,#!是什么
            prefiltered=False,
            debug=False,
            cx=viewpoint_camera.cx,#?新加的
            cy=viewpoint_camera.cy,#?新加的
            T_threshold=0.0001,#?新加的
        )
        self.rasterizer = GaussianRasterizer_depth(
            raster_settings=self.raster_settings
        )
        #把stable和unstable存起来
        means3D = gaussian_data["xyz"]
        opacity = gaussian_data["opacity"]
        scales = gaussian_data["scales"]
        rotations = gaussian_data["rotations"]
        shs = gaussian_data["shs"]
        normal = gaussian_data["normal"]
        instance = gaussian_data["instance"]

        cov3D_precomp = None
        colors_precomp = None
        #给cuda分块
        if tile_mask is None:
            tile_mask = devI(
                torch.ones(
                    (viewpoint_camera.image_height + 15) // 16,
                    (viewpoint_camera.image_width + 15) // 16,
                    dtype=torch.int32,
                )
            )
        
        render_results = self.rasterizer(
            means3D=means3D,
            opacities=opacity,
            shs=shs,
            colors_precomp=colors_precomp,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp,
            normal_w=normal,
            tile_mask=tile_mask,
        )

        rendered_image = render_results[0]
        rendered_depth = render_results[1]
        color_index_map = render_results[2]
        depth_index_map = render_results[3]#!depth_index_map是啥
        color_hit_weight = render_results[4]
        depth_hit_weight = render_results[5]
        T_map = render_results[6]
        n_touched = None
        n_touched = render_results[7]
        # rendered_instance = render_results[8]

        #从给定的index，选出被渲染的normal
        render_normal = devF(torch.zeros_like(rendered_image))
        render_normal[:, depth_index_map[0] > -1] = normal[
            depth_index_map[depth_index_map > -1].long()
        ].permute(1, 0)
        
        results = {
            "render": rendered_image,
            "depth": rendered_depth,
            "normal": render_normal,
            "color_index_map": color_index_map,
            "depth_index_map": depth_index_map,
            "color_hit_weight": color_hit_weight,
            "depth_hit_weight": depth_hit_weight,
            "T_map": T_map,
        }
        # ?渲染语义颜色
        if gaussian_data["semantics_color"] is not None and gaussian_data["semantics_color"].numel() > 1:
            semantics_color = gaussian_data["semantics_color"]
            colors_precomp = semantics_color
            shs=None
            render_results = self.rasterizer(
                means3D=means3D,
                opacities=opacity,
                shs=shs,
                colors_precomp=colors_precomp,
                scales=scales,
                rotations=rotations,
                cov3D_precomp=cov3D_precomp,
                normal_w=normal,
                tile_mask=tile_mask,
                other_color = instance
            )
            rendered_semantic_seg = render_results[0]
            results["semantic_seg"] = rendered_semantic_seg
        else:
            results["semantic_seg"] = None
        ###2024-12-11 使用实例分割结果构建新的loss
        if gaussian_data["instance"] is not None and gaussian_data["instance"].numel() > 1:
            instance = gaussian_data["instance"]
            colors_precomp = instance
            shs=None
            render_results = self.rasterizer(
            means3D=means3D,
            opacities=opacity,
            shs=shs,
            colors_precomp=colors_precomp,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp,
            normal_w=normal,
            tile_mask=tile_mask,
        )
            rendered_instance = render_results[0]
            results["instance"] = rendered_instance
        else:
            results["instance"] = None

        ###2024-12-17 消除floating
        if n_touched is not None:
            results["n_touched"] = n_touched

        return results

