import os
import shutil
import time
import random
import copy
import imgviz

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.multiprocessing as mp
from PIL import Image
from joblib.externals.cloudpickle import instance
# from sklearn.preprocessing.tests.test_data import scales
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from collections import deque
from scene.cameras import Camera
from SLAM.gaussian_pointcloud import *
from SLAM.render import Renderer
from SLAM.utils import merge_ply, rot_compare, trans_compare, bbox_filter
from utils.loss_utils import l1_loss, l2_loss, ssim
from cuda_utils._C import accumulate_gaussian_error
from utils.monitor import Recorder

#加入物体参数
from SLAM.multiprocess.quadrics import ObjectsInitialization, Occlusions_Check, MatchObject, from_Quadircs_to_Mode, \
    Update_Map, detections_filter,plot_ellipse_and_bboxes, plot_ellipse_2d,plot_ellipse_2d_net, get_gt_obj,show_render,Object_Optimize,Save_Keyframe_in_Object,\
    Object_Optimize_only,remove_outlier, rot_to_quat, bboxes_iou, plot_ellipse_3d_net
###2025-01-02 gui可视化
from gui.gui_utils import GaussianPacket
from gui.multiprocessing_utils import clone_obj
frame_window =[]

MODE = 1 # 0：cuda渲染后优化，1：bbox优化 2：ell优化 3：对物体分别优化
DEBUG= True#?控制展示物体的图像信息
ADD_OBJECT = True #?是否加入物体


SAVE_obj_ply = False #是否保存物体的点云数据
Method = 2 #0 不使用#1使用类似颜色的 #2 直接使用mask
USE_PURNE = False #是否使用purne

#设置建图的各个参数
class Mapping(object):
    def __init__(self, args, recorder=None) -> None:
        #根据base.yaml文件初始化高斯参数模型和激活函数（存在三种高斯）
        self.temp_pointcloud = GaussianPointCloud(args)
        self.pointcloud = GaussianPointCloud(args)
        self.stable_pointcloud = GaussianPointCloud(args)
        self.recorder = recorder
        #?2024-11-18 设置物体椭球
        self.object = GaussianPointCloud(args)
        self.Map_global = None
        self.obj_category = []

        #根据base.yaml文件设置渲染参数
        self.renderer = Renderer(args)
        self.optimizer = None
        self.time = 0
        self.iter = 0
        self.gaussian_update_iter = args.gaussian_update_iter#50
        self.gaussian_update_frame = args.gaussian_update_frame#4
        self.final_global_iter = args.final_global_iter#10
        
        # # history management
        self.memory_length = args.memory_length#5 保存5帧
        self.optimize_frames_ids = []
        self.processed_frames = deque(maxlen=self.memory_length)#设置了一个双端队列，最大长度为memory_length
        self.processed_map = deque(maxlen=self.memory_length)
        #设置关键帧参数
        self.keyframe_ids = []
        self.keyframe_list = []
        self.keymap_list = []
        self.global_keyframe_num = args.global_keyframe_num#3
        self.keyframe_trans_thes = args.keyframe_trans_thes#0.3
        self.keyframe_theta_thes = args.keyframe_theta_thes#30
        self.KNN_num = args.KNN_num#15
        self.KNN_threshold = args.KNN_threshold#-1
        self.history_merge_max_weight = args.history_merge_max_weight#0.5
        
        
        # points adding parameters（增加高斯判断）
        self.uniform_sample_num = args.uniform_sample_num#50000
        self.add_depth_thres = args.add_depth_thres#0.1
        self.add_normal_thres = args.add_normal_thres#1000
        self.add_color_thres = args.add_color_thres#0.1
        self.add_transmission_thres = args.add_transmission_thres#0.5

        self.transmission_sample_ratio = args.transmission_sample_ratio#1.0
        self.error_sample_ratio = args.error_sample_ratio#0.05
        self.stable_confidence_thres = args.stable_confidence_thres#200
        self.unstable_time_window = args.unstable_time_window#150

        # all map shape is [H, W, C], please note the raw image shape is [C, H, W]
        self.min_depth, self.max_depth = args.min_depth, args.max_depth #0.3 5.0
        self.depth_filter = args.depth_filter#fasle
        self.frame_map = {
            "depth_map": torch.empty(0),
            "color_map": torch.empty(0),
            "normal_map_c": torch.empty(0),
            "normal_map_w": torch.empty(0),
            "vertex_map_c": torch.empty(0),
            "vertex_map_w": torch.empty(0),
            "confidence_map": torch.empty(0),
        }
        self.model_map = {
            "render_color": torch.empty(0),
            "render_depth": torch.empty(0),
            "render_normal": torch.empty(0),
            "render_color_index": torch.empty(0),
            "render_depth_index": torch.empty(0),
            "render_transmission": torch.empty(0),
            "confidence_map": torch.empty(0),
        }

        # parameters for eval
        #用于进行评测结果
        self.save_path = args.save_path
        self.save_step = args.save_step#2000
        self.verbose = args.verbose#false
        self.mode = args.mode#single process
        self.dataset_type = args.type#tum
        assert self.mode == "single process" or self.mode == "multi process"
        self.use_tensorboard = args.use_tensorboard#true
        self.tb_writer = None

        #设置学习率
        self.feature_lr_coef = args.feature_lr_coef#1.0
        self.scaling_lr_coef = args.scaling_lr_coef#1.0
        self.rotation_lr_coef = args.rotation_lr_coef#1.0
        self.semantic_lr_coef = args.semantic_lr_coef  # 1.0
        self.object_lf_coef = args.object_lf_coef#1.0
        
    def mapping(self, frame, frame_map, frame_id, optimization_params):
        self.frame_map = frame_map
        ###2024-12-17 消除floating 多传递一个参数 frame_id
        self.gaussians_add(frame, frame_id)#获得当前视角下的gs
        self.processed_frames.append(frame)
        self.processed_map.append(frame_map)
        if frame_id == 1069 or frame_id == 1070 or frame_id==1071 or frame_id==1072:
                debug = False
        else:
            debug = False
        if ADD_OBJECT:
            #2024-11-19 完成物体的初始化
            if optimization_params.use_object and frame.detect_results is not None:
                if len(frame.detect_results) > 0: #保证检测结果大于0
                    #if DEBUG:
                    #?测试二维检测框和椭圆，分析数据是否存在问题
                    plot_ellipse_and_bboxes(frame.detect_results, frame_map["color_map"], frame_id)
                    K=frame.get_intrinsic
                    K = K.cpu().numpy()
                    Rt = frame.Rt
                    cur_detections, cur_detections_depth = detections_filter(frame.detect_results, frame_map["depth_map"],frame.image_width, frame.image_height)
                    #initial object
                    if self.Map_global is None:
                        self.Map_global = ObjectsInitialization(cur_detections, cur_detections_depth, Rt, K)
                        has_new_object = True
                    else:
                        proj_bboxes = Occlusions_Check(self.Map_global, K, Rt, frame.image_width, frame.image_height, frame_id)
                        has_new_object,_ =MatchObject(self.Map_global, cur_detections, cur_detections_depth, proj_bboxes,frame_id, frame_map["color_map"], K, Rt)
                        self.Map_global = remove_outlier(self.Map_global, K, Rt, debug)
                    if MODE ==0:
                        self.object.from_Quadrics_to_Mode(self.Map_global)
                    if DEBUG:
                        plot_ellipse_2d(self.Map_global, frame_map["color_map"], K, Rt, frame_id, True)
                    elif frame_id % 100 == 0 :
                        plot_ellipse_2d(self.Map_global, frame_map["color_map"], K, Rt, frame_id, True)

                    # #先注释，怀疑会占用cuda
                    # if has_new_object:# 记录物体初始化所在的帧
                    #     Save_Keyframe_in_Object(self.Map_global, cur_detections, frame, frame_map, frame_id, is_keyframe=False)

        #每4帧更新一次地图
        if (self.time + 1) % self.gaussian_update_frame == 0 or self.time == 0:
            self.optimize_frames_ids.append(frame_id)
            is_keyframe = self.check_keyframe(frame, frame_id)#第一帧返回false
            move_to_gpu(frame)
            if self.dataset_type == "Scannetpp":
                self.local_optimize(frame, optimization_params)
                if is_keyframe:
                    self.global_optimization(
                        optimization_params,
                        select_keyframe_num=self.global_keyframe_num
                    )
            else:
                if not is_keyframe or self.get_stable_num <= 0:
                    # if ADD_OBJECT:
                    #     if self.Map_global is not None:
                    #         self.local_optimize_with_object(frame, optimization_params, self.Map_global, frame_id)
                    #     else:
                        self.local_optimize(frame, optimization_params)#更新状态变量，并将优化后的高斯和历史高斯信息融合

                else:
                    self.global_optimization(
                        optimization_params,
                        select_keyframe_num=self.global_keyframe_num
                    )
                if USE_PURNE:
                    if frame_id==0 or is_keyframe:
                        ###2024-12-17 消除floating
                        self.to_purne(frame, frame_map)
                if (is_keyframe or frame_id==0) and 'cur_detections' in locals():
                    Object_Optimize_only(cur_detections, self.Map_global, K, Rt, frame_id)
                    # try:
                    #     Object_Optimize_only(cur_detections, self.Map_global, K, Rt, frame_id)
                    # except Exception as e:
                    #     print("Error occured in Object_Optimize_only")
                    #     print(e)
                # #2024-12-03  使用物体构建共视关系,保存历史关键帧
                # if MODE==1 and is_keyframe and len(frame.detect_results) > 0:
                #     Save_Keyframe_in_Object(self.Map_global,cur_detections,frame,frame_map,frame_id, is_keyframe)
                self.gaussians_delete(unstable=False)#删除不可靠的高斯


        self.gaussians_fix()#固定可靠的高斯
        self.error_gaussians_remove()
        self.gaussians_delete()#删除过大的高斯和时间长的高斯（不稳定的高斯）
        move_to_cpu(frame)#将图片移动到cpu上
        if ADD_OBJECT:
            if self.Map_global is not None and frame.detect_results is not None:
                if MODE == 0:
                    # 2024-11-21 渲染物体
                    self.object_optimize(frame, optimization_params)
                    Update_Map(self.Map_global, self.object)
                if MODE == 1 and len(frame.detect_results) > 0:
                    K = frame.get_intrinsic
                    K = K.cpu().numpy()
                    Rt = frame.Rt
                    #Object_Optimize(cur_detections, self.Map_global, K, Rt)
                    #Object_Optimize_only(cur_detections, self.Map_global, K, Rt)

            # 2024-11-20 展示优化后的2D椭圆
            if DEBUG:
                K = frame.get_intrinsic
                K = K.cpu().numpy()
                Rt = frame.Rt
                plot_ellipse_2d_net(self.Map_global, frame_map["color_map"], K, Rt, frame_id)
                plot_ellipse_3d_net(self.Map_global, frame_map["color_map"], K, Rt, frame_id)
            elif frame_id % 200 == 0:
                K = frame.get_intrinsic
                K = K.cpu().numpy()
                Rt = frame.Rt
                plot_ellipse_2d(self.Map_global, frame_map["color_map"], K, Rt, frame_id)



    def gaussians_add(self, frame, frame_id=None):
        self.temp_points_init(frame)#添加到temp_points，每次开始temp_points都是空的
        self.temp_points_filter()
        self.temp_points_attach(frame)
        ###2024-12-17 消除floating 多传递一个参数 frame_id
        self.temp_to_optimize(frame_id)

    def update_poses(self, new_poses):
        if new_poses is None:
            return
        for frame in self.processed_frames:
            frame.updatePose(new_poses[frame.uid])

        for frame in self.keyframe_list:
            frame.updatePose(new_poses[frame.uid])

    #2024-11-20 优化物体
    def object_optimize(self,frame,update_args):
        print("===== object optimize =====")
        l = self.object.obj_parametrize(update_args)
        history_stat = {
            "xyz": self.object._xyz.detach().clone(),
            "scaling": self.object._scaling.detach().clone(),
            "rotation": self.object.get_rotation.detach().clone(),
            "rotation_raw": self.object._rotation.detach().clone(),
        }
        #history_stat={}
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        gaussian_update_iter = self.gaussian_update_iter  # 50
        for iter in range(gaussian_update_iter):
            self.iter = iter
            random_index = random.randint(0, len(self.processed_frames) - 1)
            opt_frame = self.processed_frames[random_index]
            opt_frame_map = self.processed_map[random_index]

            params = self.obj_params
            img_obj = get_gt_obj(params["xyz"], params["obj_color"], opt_frame_map["object_img"],
                                 opt_frame.get_intrinsic, opt_frame.R, opt_frame.T)
            image_input = {
                "obj_color": devF(img_obj)
            }

            # compute loss
            render_ouput = self.renderer.render_obj(
                opt_frame,
                self.obj_params,
                tile_mask=None,
            )
            # #debug 展示渲染结果
            # show_render(render_ouput)
            loss, reported_losses = self.loss_update_obj(
                render_ouput,
                image_input,
                history_stat,
                update_args
            )

        new_stat = {
            "xyz": self.object._xyz.detach().clone(),
            "features_dc": self.object._features_dc.detach().clone(),
            "scaling": self.object._scaling.detach().clone(),
            "rotation": self.object.get_rotation.detach().clone(),
            "rotation_raw": self.object._rotation.detach().clone(),
        }
        self.object.detach()  # 清空梯度
        self.iter = 0
        # # 将当前的高斯信息和历史的高斯信息进行融合
        # self.history_merge(history_stat, self.history_merge_max_weight)  # 历史信息的weight

    ##2024-12-04 使用物体构建共视关系
    def local_optimize_with_object(self, frame, update_args, map_global, frame_id):
        print("===== map optimize =====")
        l = self.pointcloud.parametrize(update_args)#要优化哪些参数
        history_stat = {
            "opacity": self.pointcloud._opacity.detach().clone(),
            "confidence": self.pointcloud.get_confidence.detach().clone(),
            "xyz": self.pointcloud._xyz.detach().clone(),
            "features_dc": self.pointcloud._features_dc.detach().clone(),
            "features_rest": self.pointcloud._features_rest.detach().clone(),
            "scaling": self.pointcloud._scaling.detach().clone(),
            "rotation": self.pointcloud.get_rotation.detach().clone(),
            "rotation_raw": self.pointcloud._rotation.detach().clone(),
            "semantics_color": self.pointcloud._semantics.detach().clone(),
        }
        #?语义信息
        if update_args.use_semantics:
            history_stat["semantics_color"] = self.pointcloud._semantics.detach().clone()
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        gaussian_update_iter = self.gaussian_update_iter#50


        # 从物体信息中筛选出可以用的
        ids = []
        obj_frames = []
        obj_frame_maps = []
        max_num = len(self.processed_frames)
        for obj in map_global:
            if len(obj.frame_ids) ==0: continue
            for i, id in enumerate(obj.frame_ids):
                if id not in ids and frame_id - id > max_num:
                    ids.append(id)
                    obj_frames.append(obj.save_keyframe[i])
                    obj_frame_maps.append(obj.save_keyframemap[i])
        #随机选择两帧，加入到优化中去
        if len(ids) > 2:
            combined_frames = random.sample(obj_frames, 2)
            combined_frame_maps = random.sample(obj_frame_maps, 2)
        else:
            combined_frames = obj_frames
            combined_frame_maps = obj_frame_maps
        combined_frames.extend(self.processed_frames)
        combined_frame_maps.extend(self.processed_map)

        render_masks = []
        tile_masks = []
        for frame in combined_frames:#一直为5
            render_mask, tile_mask, render_ratio = self.evaluate_render_range(frame)
            render_masks.append(render_mask)
            tile_masks.append(tile_mask)
            if self.verbose:
                tile_raito = 1
                if tile_mask is not None:
                    tile_raito = tile_mask.sum() / torch.numel(tile_mask)
                print("tile mask ratio: {:f}".format(tile_raito))
        print(
            "unstable gaussian num = {:d}, stable gaussian num = {:d}".format(
                self.get_unstable_num, self.get_stable_num
            )
        )

        with tqdm(total=gaussian_update_iter, desc="map update") as pbar:
            for iter in range(gaussian_update_iter):
                self.iter = iter
                random_index = random.randint(0, len(combined_frames) - 1)
                if iter > gaussian_update_iter / 2:
                    random_index = -1
                opt_frame = combined_frames[random_index]
                opt_frame_map = combined_frame_maps[random_index]
                opt_render_mask = render_masks[random_index]
                opt_tile_mask = tile_masks[random_index]
                # compute loss
                render_ouput = self.renderer.render(
                    opt_frame,
                    self.global_params,
                    tile_mask=opt_tile_mask,
                )
                image_input = {
                    "color_map": devF(opt_frame_map["color_map"]),
                    "depth_map": devF(opt_frame_map["depth_map"]),
                    "normal_map": devF(opt_frame_map["normal_map_w"]),
                    "semantics_color": devF(opt_frame_map["semantics"]) if opt_frame_map["semantics"] is not None else None
                }
                loss, reported_losses = self.loss_update(
                    render_ouput,
                    image_input,
                    history_stat,
                    update_args,
                    render_mask=opt_render_mask,
                    unstable=True,
                )
                pbar.set_postfix({"loss": "{0:1.5f}".format(loss)})
                pbar.update(1)


        self.pointcloud.detach()#清空梯度
        self.iter = 0
        #将当前的高斯信息和历史的高斯信息进行融合
        self.history_merge(history_stat, self.history_merge_max_weight)#历史信息的weight
        del combined_frames,combined_frame_maps,ids,obj_frames,obj_frame_maps,max_num
        del render_masks, tile_masks
        del render_mask, tile_mask, render_ratio

        torch.cuda.empty_cache()  # 强制清理缓存

    ###2024-12-17 消除floating， 改成旋转
    def create_virtual_cameras(self,R,T, focal_point, theta_deg):
        def rotation_matrix(axis, theta):
            """
            创建旋转矩阵
            :param axis: 旋转轴 'x', 'y', or 'z'
            :param theta: 旋转角度 (弧度制)
            """
            c, s = np.cos(theta), np.sin(theta)
            if axis == 'x':
                return np.array([[1, 0, 0],
                                 [0, c, -s],
                                 [0, s, c]])
            elif axis == 'y':
                return np.array([[c, 0, s],
                                 [0, 1, 0],
                                 [-s, 0, c]])
            elif axis == 'z':
                return np.array([[c, -s, 0],
                                 [s, c, 0],
                                 [0, 0, 1]])
            else:
                raise ValueError("Invalid axis, choose from 'x', 'y', or 'z'")

        # 计算实际相机相对于焦点的偏移量
        real_camera_offset = np.array(T) - np.array(focal_point)
        # 将角度转换为弧度
        theta_rad = np.deg2rad(theta_deg)
        # VC1 和 VC2：绕 XZ 平面旋转 ±θ（即绕 Y 轴旋转）
        R_vc1 = np.dot(rotation_matrix('y', theta_rad), R)
        T_vc1 = np.dot(rotation_matrix('y', theta_rad), real_camera_offset.reshape(3, 1)).reshape(-1) + focal_point

        R_vc2 = np.dot(rotation_matrix('y', -theta_rad), R)
        T_vc2 = np.dot(rotation_matrix('y', -theta_rad), real_camera_offset.reshape(3, 1)).reshape(-1) + focal_point

        # VC3 和 VC4：绕 YZ 平面旋转 ±θ（即绕 X 轴旋转）
        R_vc3 = np.dot(rotation_matrix('x', theta_rad), R)
        T_vc3 = np.dot(rotation_matrix('x', theta_rad), real_camera_offset.reshape(3, 1)).reshape(-1) + focal_point

        R_vc4 = np.dot(rotation_matrix('x', -theta_rad), R)
        T_vc4 = np.dot(rotation_matrix('x', -theta_rad), real_camera_offset.reshape(3, 1)).reshape(-1) + focal_point

        # 返回虚拟相机的位姿
        return [(R_vc1, T_vc1), (R_vc2, T_vc2), (R_vc3, T_vc3), (R_vc4, T_vc4)]
    ###2024-12-17 消除floating
    def to_purne(self, frame, frame_map, frame_id=None):
        #?定义了需要渲染那些tile_mask
        render_mask, tile_mask, render_ratio = self.evaluate_render_range(frame)
        cx, cy = int(frame.cy), int(frame.cx)
        center = frame.camera_center
        d = (frame_map["depth_map"][cy, cx]).cpu().numpy()
        if d == 0:
            d = -1
        else:
            d = -d
        camera_T = frame.T
        camera_R = (frame.R).transpose(0, 1)
        focal_point = camera_T + d*camera_R[:,2]
        virtual_cameras =self.create_virtual_cameras(camera_R, camera_T, focal_point, 3)
        ## 展示未修改后的图像
        # render_ouput = self.renderer.render(frame, self.global_params)
        # img_color = render_ouput["render"]
        # img_color = img_color.permute(1, 2, 0).cpu().numpy()
        # img_color = (img_color * 255).astype(np.uint8)
        # img_color = Image.fromarray(img_color)
        # plt.imshow(img_color)
        # plt.show()
        unstable_num = self.pointcloud.get_points_num
        stable_num = self.stable_pointcloud.get_points_num
        unstable_ids = self.pointcloud._frame_ids
        stable_ids = self.stable_pointcloud._frame_ids
        frame_ids = torch.cat([unstable_ids, stable_ids], dim=0)
        n_touched = torch.zeros(unstable_num+stable_num, dtype=torch.int32, device="cuda")
        for i, (R, T) in enumerate(virtual_cameras):
            frame.update(R, T)
            render_ouput = self.renderer.render(frame, self.global_params, tile_mask)
            img_color = render_ouput["render"]
            img_color = img_color.permute(1, 2, 0).cpu().numpy()
            img_color = (img_color * 255).astype(np.uint8)
            img_color = Image.fromarray(img_color)
            plt.imshow(img_color)
            plt.show()

            # 查看n_touched
            if "n_touched" in render_ouput:
                n_touched+= render_ouput["n_touched"]
        # 得到同时满足渲染数量为0，且frame_id为当前帧的mask
        if len(self.keyframe_list) == 1:
            mask = torch.logical_and(n_touched == 0, frame_ids == frame.uid)
        else:
            # 在上一个关键帧和当前关键帧之间的mask
            id_mask = torch.logical_and(frame_ids > self.keyframe_list[-2].uid, frame_ids <= self.keyframe_list[-1].uid)
            mask = torch.logical_and(n_touched == 0, id_mask)
        ##? 不考虑帧id的问题
        # mask =n_touched == 0
        #?得到mask为True的数量
        num_true = torch.sum(mask).item()
        if num_true >0:
            # 区分得到stable 和 unstable的 mask
            unstable_mask = mask[:unstable_num]
            stable_mask = mask[unstable_num:]
            # 删除mask为True的高斯
            self.pointcloud.delete(unstable_mask)
            if len(stable_mask) > 0:
                self.stable_pointcloud.delete(stable_mask)
        # 还原frame
        frame.update(camera_R, camera_T)

    def local_optimize(self, frame, update_args):
        print("===== map optimize =====")
        l = self.pointcloud.parametrize(update_args)#要优化哪些参数
        history_stat = {
            "opacity": self.pointcloud._opacity.detach().clone(),
            "confidence": self.pointcloud.get_confidence.detach().clone(),
            "xyz": self.pointcloud._xyz.detach().clone(),
            "features_dc": self.pointcloud._features_dc.detach().clone(),
            "features_rest": self.pointcloud._features_rest.detach().clone(),
            "scaling": self.pointcloud._scaling.detach().clone(),
            "rotation": self.pointcloud.get_rotation.detach().clone(),
            "rotation_raw": self.pointcloud._rotation.detach().clone(),
            "semantics_color": self.pointcloud._semantics.detach().clone(),
        }
        #?语义信息
        if update_args.use_semantics:
            history_stat["semantics_color"] = self.pointcloud._semantics.detach().clone()
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        gaussian_update_iter = self.gaussian_update_iter#50
        render_masks = []
        tile_masks = []
        for frame in self.processed_frames:#一直为5
            render_mask, tile_mask, render_ratio = self.evaluate_render_range(frame)
            render_masks.append(render_mask)
            tile_masks.append(tile_mask)
            if self.verbose:
                tile_raito = 1
                if tile_mask is not None:
                    tile_raito = tile_mask.sum() / torch.numel(tile_mask)
                print("tile mask ratio: {:f}".format(tile_raito))
        print(
            "unstable gaussian num = {:d}, stable gaussian num = {:d}".format(
                self.get_unstable_num, self.get_stable_num
            )
        )

        with tqdm(total=gaussian_update_iter, desc="map update") as pbar:
            for iter in range(gaussian_update_iter):
                self.iter = iter
                random_index = random.randint(0, len(self.processed_frames) - 1)
                if iter > gaussian_update_iter / 2:
                    random_index = -1
                opt_frame = self.processed_frames[random_index]
                opt_frame_map = self.processed_map[random_index]
                opt_render_mask = render_masks[random_index]
                opt_tile_mask = tile_masks[random_index]
                # compute loss
                render_ouput = self.renderer.render(
                    opt_frame,
                    self.global_params,
                    tile_mask=opt_tile_mask,
                )
                image_input = {
                    "color_map": devF(opt_frame_map["color_map"]),
                    "depth_map": devF(opt_frame_map["depth_map"]),
                    "normal_map": devF(opt_frame_map["normal_map_w"]),
                    "semantics_color": devF(opt_frame_map["semantics"]) if opt_frame_map["semantics"] is not None else None,
                    "instance_img":devF(opt_frame_map["instance_img"]) if opt_frame_map["instance_img"] is not None else None
                }
                loss, reported_losses = self.loss_update(
                    render_ouput,
                    image_input,
                    history_stat,
                    update_args,
                    render_mask=opt_render_mask,
                    unstable=True,
                )
                pbar.set_postfix({"loss": "{0:1.5f}".format(loss)})
                pbar.update(1)


        self.pointcloud.detach()#清空梯度
        self.iter = 0
        #将当前的高斯信息和历史的高斯信息进行融合
        self.history_merge(history_stat, self.history_merge_max_weight)#历史信息的weight
    #将高斯的历史信息和当前信息进行融合 w*c+(1-w)*c_history
    def history_merge(self, history_stat, max_weight=0.5):
        if max_weight <= 0:
            return
        history_weight = (
            max_weight
            * history_stat["confidence"]
            / (self.pointcloud.get_confidence + 1e-6)
        )
        if self.verbose:
            print("===== history merge ======")
            print("history weight: {:.2f}".format(history_weight.mean()))
        xyz_merge = (
            history_stat["xyz"] * history_weight
            + (1 - history_weight) * self.pointcloud.get_xyz
        )

        features_dc_merge = (
            history_stat["features_dc"] * history_weight[0]
            + (1 - history_weight[0]) * self.pointcloud._features_dc
        )

        features_rest_merge = (
            history_stat["features_rest"] * history_weight[0]
            + (1 - history_weight[0]) * self.pointcloud._features_rest
        )

        scaling_merge = (
            history_stat["scaling"] * history_weight[0]
            + (1 - history_weight[0]) * self.pointcloud._scaling
        )
        rotation_merge = slerp(
            history_stat["rotation"], self.pointcloud.get_rotation, 1 - history_weight
        )
        #?语义信息
        if history_stat["semantics_color"] is not None:
            semantic_color = (
                history_stat["semantics_color"] * history_weight[0]
                + (1 - history_weight[0]) * self.pointcloud._semantics
            )
            self.pointcloud._semantics = semantic_color

        self.pointcloud._xyz = xyz_merge
        self.pointcloud._features_dc = features_dc_merge
        self.pointcloud._features_rest = features_rest_merge
        self.pointcloud._scaling = scaling_merge
        self.pointcloud._rotation = rotation_merge


    # Fix the confidence points
    #固定可靠的gs
    def gaussians_fix(self, mask=None):
        if mask is None:
            confidence_mask = (
                self.pointcloud.get_confidence > self.stable_confidence_thres#优化200次的才认为是可靠的
            ).squeeze()
            stable_mask = confidence_mask
        else:
            stable_mask = mask.squeeze()
        if self.verbose:#self.verbose说明是否打印中间变量
            print("===== points fix =====")
            print(
                "fix gaussian num: {:d}".format(stable_mask.sum()),
            )
        if stable_mask.sum() > 0:
            stable_params = self.pointcloud.remove(stable_mask)#从优化参数中移除可靠的gs
            #使用clip确保所有stable_params中的值不会大于stable_confidence_thres
            stable_params["confidence"] = torch.clip(
                stable_params["confidence"], max=self.stable_confidence_thres
            )
            self.stable_pointcloud.cat(stable_params)

    # Fix the confidence points
    def gaussians_release(self, mask):
        if mask.sum() > 0:
            unstable_params = self.stable_pointcloud.remove(mask)
            unstable_params["confidence"] = devF(
                torch.zeros_like(unstable_params["confidence"])
            )
            unstable_params["add_tick"] = self.time * devF(
                torch.ones_like(unstable_params["add_tick"])
            )
            #self.stable_pointcloud.cat(unstable_params)
            self.pointcloud.cat(unstable_params)

    # Remove too small/big gaussians, long time unstable gaussians, insolated_gaussians
    def gaussians_delete(self, unstable=True):
        if unstable:
            pointcloud = self.pointcloud
        else:
            pointcloud = self.stable_pointcloud
        if pointcloud.get_points_num == 0:
            return
        threshold = self.KNN_threshold
        #选出过分大的高斯球
        big_gaussian_mask = (
            pointcloud.get_radius > (pointcloud.get_radius.mean() * 10)
        ).squeeze()
        unstable_time_mask = (
            (self.time - pointcloud.get_add_tick) > self.unstable_time_window
        ).squeeze()
        # isolated_gaussian_mask = self.gaussians_isolated(
        #     pointcloud.get_xyz, self.KNN_num, threshold
        # )
        if unstable:
            # delete_mask = (
            #     big_gaussian_mask | unstable_time_mask | isolated_gaussian_mask
            # )
            delete_mask = (
                big_gaussian_mask | unstable_time_mask 
            )
        else:
            # delete_mask = big_gaussian_mask | isolated_gaussian_mask
            delete_mask = big_gaussian_mask 
        if self.verbose:
            print("===== points delete =====")
            print(
                "threshold: {:.1f} cm, big num: {:d}, unstable num: {:d}, delete num: {:d}".format(
                    threshold * 100,
                    big_gaussian_mask.sum(),
                    unstable_time_mask.sum(),
                    delete_mask.sum(),
                ),
            )
        pointcloud.delete(delete_mask)

    # check if current frame is a keyframe
    #检查当前帧是否为关键帧
    def check_keyframe(self, frame, frame_id):
        # add keyframe
        if self.time == 0:#第一帧
            self.keyframe_list.append(frame.move_to_cpu_clone())
            self.keyframe_ids.append(frame_id)
            image_input = {
                "color_map": self.frame_map["color_map"].detach().cpu(),
                "depth_map": self.frame_map["depth_map"].detach().cpu(),
                "normal_map": self.frame_map["normal_map_w"].detach().cpu(),
                "semantics_color": self.frame_map["semantics"].detach().cpu() if self.frame_map["semantics"] is not None else None,
                ###2024-12-11 使用实例分割结果构建新的loss
                "instance_img":self.frame_map["instance_img"].detach().cpu() if self.frame_map["instance_img"] is not None else None
            }
            self.keymap_list.append(image_input)
            return False
        prev_rot = self.keyframe_list[-1].R.T
        prev_trans = self.keyframe_list[-1].T
        curr_rot = frame.R.T
        curr_trans = frame.T
        _, theta_diff = rot_compare(prev_rot, curr_rot)#返回角度误差
        _, l2_diff = trans_compare(prev_trans, curr_trans)#返回L2范数
        if self.verbose:
            print("rot diff: {:.2f}, move diff: {:.2f}".format(theta_diff, l2_diff))
        if theta_diff > self.keyframe_theta_thes or l2_diff > self.keyframe_trans_thes:
            print("add key frame at frame {:d}!".format(self.time))
            image_input = {
                "color_map": self.frame_map["color_map"].detach().cpu(),
                "depth_map": self.frame_map["depth_map"].detach().cpu(),
                "normal_map": self.frame_map["normal_map_w"].detach().cpu(),
                "semantics_color": self.frame_map["semantics"].detach().cpu() if self.frame_map["semantics"] is not None else None,
                ###2024-12-11 使用实例分割结果构建新的loss
                "instance_img": self.frame_map["instance_img"].detach().cpu() if self.frame_map[
                                                                                     "instance_img"] is not None else None
            }
            self.keyframe_list.append(frame.move_to_cpu_clone())
            self.keymap_list.append(image_input)
            self.keyframe_ids.append(frame_id)
            return True
        else:
            return False

    #2024-11-25
    def loss_update_obj(
            self,
            render_output,
            image_input,
            init_stat=None,
            update_args=None,
            render_mask=None,):
        image = render_output["render_obj"].permute(1, 2, 0)
        obj_loss = l1_loss(image, image_input["obj_color"])
        loss = obj_loss*update_args.object_weight
        #loss = obj_loss
        loss.backward()
        self.optimizer.step()


        # report train loss
        report_losses = {
            "obj_loss": loss.item(),
        }
        self.train_report(self.get_total_iter, report_losses)
        self.optimizer.zero_grad(set_to_none=True)
        return loss, report_losses
    # update confidence by grad
    def loss_update(
        self,
        render_output,
        image_input,
        init_stat,
        update_args,
        render_mask=None,
        unstable=True,
    ):
        if unstable:
            pointcloud = self.pointcloud
        else:
            pointcloud = self.stable_pointcloud
        opacity = pointcloud.opacity_activation(init_stat["opacity"])
        attach_mask = (opacity < 0.9).squeeze()
        attach_loss = torch.tensor(0)
        if attach_mask.sum() > 0:
            attach_loss = 1000 * (
                l2_loss(
                    pointcloud._scaling[attach_mask],
                    init_stat["scaling"][attach_mask],
                )
                + l2_loss(
                    pointcloud._xyz[attach_mask],
                    init_stat["xyz"][attach_mask],
                )
                + l2_loss(
                    pointcloud._rotation[attach_mask],
                    init_stat["rotation_raw"][attach_mask],
                )
            )
        image, depth, normal, depth_index = (
            render_output["render"].permute(1, 2, 0),
            render_output["depth"].permute(1, 2, 0),
            render_output["normal"].permute(1, 2, 0),
            render_output["depth_index_map"].permute(1, 2, 0),
        )
        ssim_loss = devF(torch.tensor(0))
        normal_loss = devF(torch.tensor(0))
        depth_loss = devF(torch.tensor(0))
        if render_mask is None:
            render_mask = devB(torch.ones(image.shape[:2]))
            ssim_loss = 1 - ssim(image.permute(2,0,1), image_input["color_map"].permute(2,0,1))
        else:
            render_mask = render_mask.bool()
        # render_mask include depth == 0
        if self.dataset_type == "Scannetpp":
            render_mask = render_mask & (image_input["depth_map"] > 0).squeeze()
        color_loss = l1_loss(image[render_mask], image_input["color_map"][render_mask])

        if depth is not None and update_args.depth_weight > 0:
            depth_error = depth - image_input["depth_map"]
            valid_depth_mask = (
                (depth_index != -1).squeeze()
                & (image_input["depth_map"] > 0).squeeze()
                & (depth_error < self.add_depth_thres).squeeze()
                & render_mask
            )
            depth_loss = torch.abs(depth_error[valid_depth_mask]).mean()

        if normal is not None and update_args.normal_weight > 0:
            cos_dist = 1 - F.cosine_similarity(
                normal, image_input["normal_map"], dim=-1
            )
            valid_normal_mask = (
                render_mask
                & (depth_index != -1).squeeze()
                & (~(image_input["normal_map"] == 0).all(dim=-1))
            )
            normal_loss = cos_dist[valid_normal_mask].mean()

        total_loss = (
            update_args.depth_weight * depth_loss#深度误差
            + update_args.normal_weight * normal_loss#normal_weight的值就是0
            + update_args.color_weight * color_loss#颜色误差项1
            + update_args.ssim_weight * ssim_loss #应该是正则项
        )
        #?定义语义误差
        if update_args.use_semantics:
            semantics = render_output["semantic_seg"].permute(1, 2, 0)
            semantics_loss = l1_loss(semantics[render_mask], image_input["semantics_color"][render_mask])
            total_loss+=update_args.semantic_color_weight * semantics_loss
        if update_args.use_instance:
            if Method ==1:
                instance = render_output["instance"].permute(1, 2, 0)
                instance_loss = l1_loss(instance[render_mask], image_input["instance_img"][render_mask])
            if Method ==2:
                instance = render_output["T_map"].permute(1, 2, 0)
                instance_gt = torch.sum(image_input["instance_img"],dim=-1, keepdim=True)
                #这个函数会检查 instance_gt 中每个元素是否大于 0。如果条件为真（即元素大于 0），则将该元素替换为 0；否则，将该元素替换为 1。
                instance_gt = torch.where(instance_gt > 0, torch.tensor(0, dtype=instance_gt.dtype, device=instance_gt.device),
                                          torch.tensor(1, dtype=instance_gt.dtype, device=instance_gt.device))
                instance_loss = l1_loss(instance[render_mask], instance_gt[render_mask])
                # instance_loss = (instance - instance_gt)
                # instance_loss = torch.abs(instance_loss[render_mask]).mean()
            if Method == 0:
                instance_loss=0
                pass



            # instance_error = instance - image_input["instance_img"]
            # instance_loss = torch.abs(instance_error[render_mask]).mean()
            total_loss+=update_args.instance_weight * instance_loss

        loss = total_loss
        (loss + attach_loss).backward()
        self.optimizer.step()

        # update confidence by grad
        grad_mask = (pointcloud._features_dc.grad.abs() != 0).any(dim=-1)
        pointcloud._confidence[grad_mask] += 1 #参数每更新一次，增加一点cofidence

        # report train loss
        report_losses = {
            "total_loss": total_loss.item(),
            "depth_loss": depth_loss.item(),
            "ssim_loss": ssim_loss.item(),
            "normal_loss": normal_loss.item(),
            "color_loss": color_loss.item(),
            "scale_loss": attach_loss.item(),
        }
        # ?增加语义信息
        if update_args.use_semantics:
            report_losses["semantic_loss"] = semantics_loss.item()
        if update_args.use_instance and Method>0:
            report_losses["instance_loss"] = instance_loss.item()
        self.train_report(self.get_total_iter, report_losses)
        self.optimizer.zero_grad(set_to_none=True)
        return loss, report_losses

    def evaluate_render_range(#local时unstable为true，全局优化为false
        self, frame, global_opt=False, sample_ratio=-1, unstable=True
    ):
        if unstable:
            render_output = self.renderer.render(
                frame,
                self.unstable_params #全部都在cuda上
            )
        else:
            render_output = self.renderer.render(
                frame,
                self.stable_params
            )
        unstable_T_map = render_output["T_map"]

        if global_opt:
            #!待选
            if sample_ratio > 0:
                render_image = render_output["render"].permute(1, 2, 0)
                gt_image = frame.original_image.permute(1, 2, 0).cuda()
                image_diff = (render_image - gt_image).abs()
                color_error = torch.sum(
                    image_diff, dim=-1, keepdim=False
                )
                filter_mask = (render_image.sum(dim=-1) == 0)
                color_error[filter_mask] = 0
                tile_mask = colorerror2tilemask(color_error, 16, sample_ratio)
                #?加入语义不准确的地方 2024-11-12
                if render_output["semantic_seg"] is not None:
                    semantic_seg = render_output["semantic_seg"].permute(1, 2, 0)
                    gt_semantic = frame.semantics.permute(1, 2, 0).cuda()
                    semantic_diff = (semantic_seg - gt_semantic).abs()
                    semantic_error = torch.sum(
                        semantic_diff, dim=-1, keepdim=False
                    )
                    filter_mask = (semantic_seg.sum(dim=-1) == 0)
                    semantic_error[filter_mask] = 0
                    semantic_mask = colorerror2tilemask(semantic_error, 16, sample_ratio)
                    tile_mask = tile_mask | semantic_mask
                render_mask = (
                    F.interpolate(
                        tile_mask.float().unsqueeze(0).unsqueeze(0),
                        scale_factor=16,
                        mode="nearest",
                    )
                    .squeeze(0)
                    .squeeze(0)
                    .bool()
                )[: color_error.shape[0], : color_error.shape[1]]
            # after training, real global optimization
            else:
                render_mask = (unstable_T_map != 1).squeeze(0)
                tile_mask = None
        else:
            render_mask = (unstable_T_map != 1).squeeze(0)
            tile_mask = transmission2tilemask(render_mask, 16, 0.5)

        render_ratio = render_mask.sum() / self.get_pixel_num
        return render_mask, tile_mask, render_ratio
    def error_gaussians_remove(self):
        if self.get_stable_num <= 0:
            return
        # check error by backprojection
        check_frame = self.processed_frames[-1]
        check_map = self.processed_map[-1]
        render_output = self.renderer.render(
            check_frame, self.global_params
        )
        # [unstable, stable]
        unstable_points_num = self.get_unstable_num
        stable_points_num = self.get_stable_num

        color = render_output["render"].permute(1, 2, 0)
        depth = render_output["depth"].permute(1, 2, 0)
        normal = render_output["normal"].permute(1, 2, 0)
        # #?加入去掉语义误差大的gs 2024-11-12
        # if render_output["semantic_seg"] is not None:
        #     semantic_seg = render_output["semantic_seg"].permute(1, 2, 0)
        #     semantic_error = torch.abs(check_map["semantics"] - semantic_seg)
        #     semantic_error = torch.sum(semantic_error, dim=-1, keepdim=True)
        #     semantic_error[check_map["depth_map"] == 0] = 0

        depth_index = render_output["depth_index_map"].permute(1, 2, 0)
        color_index = render_output["color_index_map"].permute(1, 2, 0)

        depth_error = torch.abs(check_map["depth_map"] - depth)#depth_map应该是观测值
        depth_error[(check_map["depth_map"] - depth) < 0] = 0
        image_error = torch.abs(check_map["color_map"] - color)
        color_error = torch.sum(image_error, dim=-1, keepdim=True)

        normal_error = devF(torch.zeros_like(depth_error))
        invalid_mask = (check_map["depth_map"] == 0) | (depth_index == -1)
        invalid_mask = invalid_mask.squeeze()

        depth_error[invalid_mask] = 0
        color_error[check_map["depth_map"] == 0] = 0
        normal_error[invalid_mask] = 0
        H, W = self.frame_map["color_map"].shape[:2]
        P = unstable_points_num + stable_points_num
        (
            gaussian_color_error,
            gaussian_depth_error,
            gaussian_normal_error,
            outlier_count,
        ) = accumulate_gaussian_error(
            H,
            W,
            P,
            color_error,
            depth_error,
            normal_error,
            color_index,
            depth_index,
            self.add_color_thres,
            self.add_depth_thres,
            self.add_normal_thres,
            True,
        )#调用cuda计算累计误差
        # # ?加入去掉语义误差大的gs 2024-11-12
        # if render_output["semantic_seg"] is not None:
        #     gaussain_semantic_error,_,_,_ = accumulate_gaussian_error(
        #         H,
        #         W,
        #         P,
        #         semantic_error,
        #         depth_error,
        #         normal_error,
        #         color_index,
        #         depth_index,
        #         self.add_color_thres,
        #         self.add_depth_thres,
        #         self.add_normal_thres,
        #         True,
        #     )
        color_filter_thres = 2 * self.add_color_thres
        depth_filter_thres = 2 * self.add_depth_thres
        #误差较大的地方
        depth_delete_mask = (gaussian_depth_error > depth_filter_thres).squeeze()
        color_release_mask = (gaussian_color_error > color_filter_thres).squeeze()
        # # ?加入去掉语义误差大的gs 2024-11-12
        # if render_output["semantic_seg"] is not None:
        #     semantic_release_mask = (gaussain_semantic_error > color_filter_thres).squeeze()
        if self.verbose:
            print("===== outlier remove =====")
            print(
                "color outlier num: {:d}, depth outlier num: {:d}".format(
                    (color_release_mask).sum(),
                    (depth_delete_mask).sum(),
                ),
            )

        depth_delete_mask_stable = depth_delete_mask[unstable_points_num:, ...]#从unstable_points检索剩余的点（只找stable)
        color_release_mask_stable = color_release_mask[unstable_points_num:, ...]
        # # ?加入去掉语义误差大的gs 2024-11-12
        # if render_output["semantic_seg"] is not None:
        #     semantic_release_mask_stable = semantic_release_mask[unstable_points_num:, ...]
        #     #!待选 这里直接认为语义误差其实就是几何误差
        #     self.stable_pointcloud._color_error_counter[semantic_release_mask_stable] +=1

        self.stable_pointcloud._depth_error_counter[depth_delete_mask_stable] += 1
        self.stable_pointcloud._color_error_counter[color_release_mask_stable] += 1

        delete_thresh = 10
        depth_delete_mask = (
            self.stable_pointcloud._depth_error_counter >= delete_thresh
        ).squeeze()
        color_release_mask = (
            self.stable_pointcloud._color_error_counter >= delete_thresh
        ).squeeze()
        # move_to_cpu(keyframe)
        # move_to_cpu_map(keymap)
        self.stable_pointcloud.delete(depth_delete_mask)#深度误差大，直接删除
        self.gaussians_release(color_release_mask[~depth_delete_mask])#颜色误差大，只清空高斯的'confidence'和‘add_tick’

    # update all stable gaussians by keyframes render（通过关键帧更新所有的stable gs）
    def global_optimization(
        self, update_args, select_keyframe_num=-1, is_end=False
    ):
        print("===== global optimize =====")
        if select_keyframe_num == -1:
            self.gaussians_fix(mask=(self.pointcloud.get_confidence > -1))
        print(
            "keyframe num = {:d}, stable gaussian num = {:d}".format(
                self.get_keyframe_num, self.get_stable_num
            )
        )
        if self.get_stable_num == 0:
            return
        #只对不稳定的高斯进行优化
        l = self.stable_pointcloud.parametrize(update_args)#这里设置哪些参数需要求导
        if select_keyframe_num != -1:
            l[0]["lr"] = 0
            for i in range(1, len(l)):
                l[i]["lr"] *= 0.1
        else:
            l[0]["lr"] = 0.0000
            l[1]["lr"] *= self.feature_lr_coef
            l[2]["lr"] *= self.feature_lr_coef
            l[4]["lr"] *= self.scaling_lr_coef
            l[5]["lr"] *= self.rotation_lr_coef
            if update_args.use_semantics:
                l[6]["lr"] *= self.semantic_lr_coef
        is_final = False
        #利用稳定的高斯进行渲染，不需要计算梯度
        init_stat = {
            "opacity": self.stable_pointcloud._opacity.detach().clone(),
            "scaling": self.stable_pointcloud._scaling.detach().clone(),
            "xyz": self.stable_pointcloud._xyz.detach().clone(),
            "rotation_raw": self.stable_pointcloud._rotation.detach().clone(),
        }
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        total_iter = int(self.gaussian_update_iter)
        sample_ratio = 0.4
        if select_keyframe_num == -1:
            total_iter = self.get_keyframe_num * self.final_global_iter#final_global_iter=20
            is_final = True
            select_keyframe_num = self.get_keyframe_num
            update_args.depth_weight = 0
            sample_ratio = -1

        # test random kframes
        random_kframes = False
        
        select_keyframe_num = min(select_keyframe_num, self.get_keyframe_num)
        if random_kframes:
            if select_keyframe_num >= self.get_keyframe_num:
                select_kframe_indexs = list(range(0, self.get_keyframe_num))
            else:
                select_kframe_indexs = np.random.choice(np.arange(1, min(select_keyframe_num * 2, self.get_keyframe_num)),
                                                        select_keyframe_num-1,
                                                        replace=False).tolist() + [0]
        else:
            select_kframe_indexs = list(range(select_keyframe_num))
        
        select_kframe_indexs = [i*-1-1 for i in select_kframe_indexs]
            
            
        select_frame = []
        select_map = []
        select_render_mask = []
        select_tile_mask = []
        select_id = []
        # TODO: only sample some pixels of keyframe for global optimization
        for index in select_kframe_indexs:
            move_to_gpu(self.keyframe_list[index])
            move_to_gpu_map(self.keymap_list[index])
            select_frame.append(self.keyframe_list[index])
            select_map.append(self.keymap_list[index])
            render_mask, tile_mask, _ = self.evaluate_render_range(
                self.keyframe_list[index],
                global_opt=True,
                unstable=False,
                sample_ratio=sample_ratio,
            )
            if select_keyframe_num == -1:
                move_to_cpu(self.keyframe_list[index])
                move_to_cpu_map(self.keymap_list[index])
            select_render_mask.append(render_mask)
            select_tile_mask.append(tile_mask)
            select_id.append(self.keyframe_ids[index])
            if self.verbose:
                tile_raito = 1
                if tile_mask is not None:
                    tile_raito = tile_mask.sum() / torch.numel(tile_mask)

        with tqdm(total=total_iter, desc="global optimization") as pbar:
            for iter in range(total_iter):
                self.iter = iter
                random_index = random.randint(0, select_keyframe_num - 1)#随机选择k个关键帧
                frame_input = select_frame[random_index]
                image_input = select_map[random_index]
                if select_keyframe_num == -1:
                    move_to_gpu(frame_input)
                    move_to_gpu_map(image_input)
                if not random_kframes and iter > total_iter / 2 and not is_final:
                    random_index = -1
                render_ouput = self.renderer.render(
                    frame_input,
                    self.stable_params,
                    tile_mask=select_tile_mask[random_index],
                )
                loss, reported_losses = self.loss_update(
                    render_ouput,
                    image_input,
                    init_stat,
                    update_args,
                    render_mask=select_render_mask[random_index],
                    unstable=False,
                )
                if select_keyframe_num == -1:
                    move_to_cpu(frame_input)
                    move_to_cpu_map(image_input)
                pbar.set_postfix({"loss": "{0:1.5f}".format(loss)})
                pbar.update(1)

        for index in range(-select_keyframe_num, 0):
            move_to_cpu(self.keyframe_list[index])
            move_to_cpu_map(self.keymap_list[index])
        self.stable_pointcloud.detach()

    # Sample some pixels as the init gaussians
    def temp_points_init(self, frame: Camera):
        # print("===== temp points add =====")
        #对于第一帧情况
        if self.time == 0:
            depth_range_mask = self.frame_map["depth_map"] > 0
            #采样像素
            #返回的xyz 和 normal是在世界坐标系下的坐标和法向量
            xyz, normal, color, semantic_color, instance = sample_pixels(
                self.frame_map["vertex_map_w"],
                self.frame_map["normal_map_w"],
                self.frame_map["color_map"],
                self.uniform_sample_num,
                depth_range_mask,
                self.frame_map["semantics"],
                self.frame_map["instance_img"]
            )
            self.temp_pointcloud.add_empty_points(xyz, normal, color, self.time, semantic_color,instance)
        else:
            self.get_render_output(frame)
            #这里应该新观察到的，需要添加高斯
            transmission_sample_mask = (
                self.model_map["render_transmission"] > self.add_transmission_thres
            ) & (self.frame_map["depth_map"] > 0)
            transmission_sample_ratio = (
                transmission_sample_mask.sum() / self.get_pixel_num
            )
            #根据按照比例选择需要添加的高斯数
            transmission_sample_num = devI(
                self.transmission_sample_ratio
                * transmission_sample_ratio
                * self.uniform_sample_num
            )
            if self.verbose:
                print(
                    "transmission empty num = {:d}, sample num = {:d}".format(
                        transmission_sample_mask.sum(), transmission_sample_num
                    )
                )
            results = sample_pixels(
                self.frame_map["vertex_map_w"],
                self.frame_map["normal_map_w"],
                self.frame_map["color_map"],
                transmission_sample_num,
                transmission_sample_mask,
                self.frame_map["semantics"],
                self.frame_map["instance_img"]
            )

            xyz_trans, normal_trans, color_trans,  semantic_color_trans,instance_trans= sample_pixels(
                self.frame_map["vertex_map_w"],
                self.frame_map["normal_map_w"],
                self.frame_map["color_map"],
                transmission_sample_num,
                transmission_sample_mask,
                self.frame_map["semantics"],
                self.frame_map["instance_img"]
            )
            self.temp_pointcloud.add_empty_points(
                xyz_trans, normal_trans, color_trans, self.time,semantic_color_trans,instance_trans
            )

            depth_error = torch.abs(
                self.frame_map["depth_map"] - self.model_map["render_depth"]
            )
            color_error = torch.abs(
                self.frame_map["color_map"] - self.model_map["render_color"]
            ).mean(dim=-1, keepdim=True)

            #下面是为具有较大颜色和深度误差情况添加高斯
            depth_sample_mask = (
                (depth_error > self.add_depth_thres)
                & (self.frame_map["depth_map"] > 0)
                & (self.model_map["render_depth_index"] > -1)
            )
            color_sample_mask = (
                (color_error > self.add_color_thres)
                & (self.frame_map["depth_map"] > 0)
                & (self.model_map["render_transmission"] < self.add_transmission_thres)
            )
            # ?增加语义误差项
            if self.frame_map["semantics"] is not None:
                semantic_error = torch.abs(
                    self.frame_map["semantics"] - self.model_map["semantic_seg"]
                ).mean(dim=-1, keepdim=True)
                sematic_sample_mask = (
                    (semantic_error > self.add_color_thres)
                    & (self.frame_map["depth_map"] > 0)
                    & (self.model_map["render_transmission"] < self.add_transmission_thres)
                )

            sample_mask = color_sample_mask | depth_sample_mask
            #?增加语义误差项
            if self.frame_map["semantics"] is not None:
                sample_mask = sample_mask | sematic_sample_mask

            sample_mask = sample_mask & (~transmission_sample_mask)
            sample_num = devI(sample_mask.sum() * self.error_sample_ratio)
            if self.verbose:
                print(
                    "wrong depth num = {:d}, wrong color num = {:d}, sample num = {:d}".format(
                        depth_sample_mask.sum(),
                        color_sample_mask.sum(),
                        sample_num,
                    )
                )
            xyz_error, normal_error, color_error, semantic_color_error, instance_error = sample_pixels(
                self.frame_map["vertex_map_w"],
                self.frame_map["normal_map_w"],
                self.frame_map["color_map"],
                sample_num,
                sample_mask,
                self.frame_map["semantics"],
                self.frame_map["instance_img"]
            )
            self.temp_pointcloud.add_empty_points(
                xyz_error, normal_error, color_error, self.time,semantic_color_error,instance_error
            )

    # Remove temp points that fall within the existing unstable Gaussian.
    #从temp_points删除存在的ubstable的点
    def temp_points_filter(self, topk=3):
        if self.get_unstable_num > 0:#返回的是pointcloud的数量
            temp_xyz = self.temp_pointcloud.get_xyz
            if self.verbose:
                print("init {} temp points".format(self.temp_pointcloud.get_points_num))
            exist_xyz = self.unstable_params["xyz"]
            exist_raidus = self.unstable_params["radius"]
            if torch.numel(exist_xyz) > 0 and torch.numel(temp_xyz)>0:
                inbbox_mask = bbox_filter(temp_xyz, exist_xyz)#返回unstable在temppoints包围盒中的点
                exist_xyz = exist_xyz[inbbox_mask]
                exist_raidus = exist_raidus[inbbox_mask]

            if torch.numel(exist_xyz) == 0:
                return
            #将tmp_xyz进行聚类
            nn_dist, nn_indices, _ = knn_points(
                temp_xyz[None, ...],
                exist_xyz[None, ...],
                norm=2,
                K=topk,
                return_nn=True,
            )
            nn_dist = torch.sqrt(nn_dist).squeeze(0)
            nn_indices = nn_indices.squeeze(0)
            #根据聚类的结果，删除落在unstable范围内的点
            corr_radius = exist_raidus[nn_indices] * 0.6
            inside_mask = (nn_dist < corr_radius).any(dim=-1)
            if self.verbose:
                print("delete {} temp points".format(inside_mask.sum().item()))
            self.temp_pointcloud.delete(inside_mask)

    # Attach gaussians fall with in the stable gaussians. attached gaussians is set to low opacity and fix scale
    #增加低不透明度的高斯
    def temp_points_attach(self, frame: Camera, unstable_opacity_low=0.1):
        if self.get_stable_num == 0:
            return
        # project unstable gaussians and compute uv
        unstable_xyz = self.temp_pointcloud.get_xyz
        origin_indices = torch.arange(unstable_xyz.shape[0]).cuda().long()
        unstable_opacity = self.temp_pointcloud.get_opacity
        unstable_opacity_filter = (unstable_opacity > unstable_opacity_low).squeeze(-1)#self.temp_pointcloud.get_opacity不透明度应该都为0.99,很神奇
        unstable_xyz = unstable_xyz[unstable_opacity_filter]
        unstable_uv = frame.get_uv(unstable_xyz)#从世界坐标系转到像素坐标系
        indices = torch.arange(unstable_xyz.shape[0]).cuda().long()
        unstable_mask = (
            (unstable_uv[:, 0] >= 0)
            & (unstable_uv[:, 0] < frame.image_width)
            & (unstable_uv[:, 1] >= 0)
            & (unstable_uv[:, 1] < frame.image_height)
        )#限制像素范围
        project_uv = unstable_uv[unstable_mask]

        # get the corresponding stable gaussians（只通过stable渲染得到结果）
        stable_render_output = self.renderer.render(
            frame, self.stable_params,
        )
        #找到temp_point中和 stable_point 重合的区域
        stable_index = stable_render_output["color_index_map"].permute(1, 2, 0)#[480,640,1]
        intersect_mask = stable_index[project_uv[:, 1], project_uv[:, 0]] >= 0 #[1987,1]color_index_map应该代表的是被几个高斯渲染
        indices = indices[unstable_mask][intersect_mask[:, 0]]#两次选择

        # compute point to plane distance（找到stable_point中重合的点）
        intersect_stable_index = (
            (stable_index[unstable_uv[indices, 1], unstable_uv[indices, 0]])
            .squeeze(-1)
            .long()
        )
        #获得最小尺度方向的法向量，
        stable_normal_check = self.stable_pointcloud.get_normal[intersect_stable_index]
        stable_xyz_check = self.stable_pointcloud.get_xyz[intersect_stable_index]#intersect_stable_index是stable中重和的点
        unstable_xyz_check = self.temp_pointcloud.get_xyz[indices]#indices是temp_point中重合的点
        point_to_plane_distance = (
            (stable_xyz_check - unstable_xyz_check) * stable_normal_check
        ).sum(dim=-1)#计算temp_point点到stable_point平面距离
        intersect_check = point_to_plane_distance.abs() < 0.5 * self.add_depth_thres
        indices = indices[intersect_check]
        indices = origin_indices[unstable_opacity_filter][indices]

        # set opacity
        self.temp_pointcloud._opacity[indices] = inverse_sigmoid(
            unstable_opacity_low
            * torch.ones_like(self.temp_pointcloud._opacity[indices])
        )
        if self.verbose:
            print("attach {} unstable gaussians".format(indices.shape[0]))

    # Initialize temp points as unstable gaussian.
    def temp_to_optimize(self, frame_id=None):###2024-12-17 消除floating 多增加一个frame_id参数
        #更新当前临时保存的高斯
        self.temp_pointcloud.update_geometry(
            self.global_params["xyz"],#stable+unstable
            self.global_params["radius"],#stable+unstable
        )
        if self.verbose:#false
            print("===== points add =====")
            print(
                "add new gaussian num: {:d}".format(self.temp_pointcloud.get_points_num)
            )
        remove_mask = devB(torch.ones(self.temp_pointcloud.get_points_num))

        ###2024-12-17 消除floating，记录frame_id
        if frame_id is not None:
            ids = devI(torch.ones(self.temp_pointcloud.get_points_num) * frame_id)

        # 这里是为了获得参数，方便下面传递给pointcloud,同时将temp_point中的点删除
        temp_params = self.temp_pointcloud.remove(remove_mask)

        ###2024-12-17 消除floating，记录frame_id
        if frame_id is not None:
            temp_params["frame_id"] = ids

        self.pointcloud.cat(temp_params)#保留当前存在的gs



    # detect isolated gaussians by KNN
    def gaussians_isolated(self, points, topk=5, threshold=0.005):
        if threshold < 0:
            isolated_mask = devB(torch.zeros(points.shape[0]))
            return isolated_mask
        nn_dist, nn_indices, _ = knn_points(
            points[None, ...],
            points[None, ...],
            norm=2,
            K=topk + 1,
            return_nn=True,
        )
        dist_mean = nn_dist[0, :, 1:].mean(1)
        isolated_mask = dist_mean > threshold
        return isolated_mask

#保存 渲染 模型 轨迹 和评估结果
    def create_workspace(self):
        #如果保存路径下存在内容，那么删除所有内容
        if os.path.exists(self.save_path):
            shutil.rmtree(self.save_path)
        print(self.save_path)
        os.makedirs(self.save_path, exist_ok=True)
        render_save_path = os.path.join(self.save_path, "eval_render")
        os.makedirs(render_save_path, exist_ok=True)
        model_save_path = os.path.join(self.save_path, "save_model")
        os.makedirs(model_save_path, exist_ok=True)
        traj_save_path = os.path.join(self.save_path, "save_traj")
        os.makedirs(traj_save_path, exist_ok=True)
        traj_save_path = os.path.join(self.save_path, "eval_metric")
        os.makedirs(traj_save_path, exist_ok=True)

        if self.mode == "single process" and self.use_tensorboard:
            self.tb_writer = SummaryWriter(self.save_path)#用于将数据写入tensorboard
        else:
            self.tb_writer = None

    def record_iou(self,cam,path=None, save_iou=True):
        if path == None:
            frame_name = "frame_{:04d}".format(self.time)
            model_save_path = os.path.join(self.save_path, "save_obj", frame_name)
            os.makedirs(model_save_path, exist_ok=True)
            path = os.path.join(
                model_save_path,
                "iter_{:04d}".format(self.iter),
            )
        if save_iou and self.Map_global is not None:
            K = cam.get_intrinsic.detach().cpu().numpy()
            for obj in self.Map_global:
                cat_id = obj.category_id_
                iou_sum = 0
                cnt = 0
                for i in range(len(obj.Rts_)):
                    Rt = obj.Rts_[i]
                    obv_bbox = obj.bboxes_[i]
                    P = K@Rt
                    ell = obj.ellipsoid_.project(P)
                    bb_proj = ell.ComputeBbox()
                    iou = bboxes_iou(obv_bbox, bb_proj)
                    if iou>0:
                        iou_sum+=iou
                        cnt+=1
                if cnt>0:
                    result = iou_sum/cnt
                else:
                    result=0
                with open(path+"iou.txt", 'a') as file:
                    file.write(f"ID:{cat_id} IOU:{result}\n")
                file.close()

    def save_obj(self, path=None, save_data=True):
        if path == None:
            frame_name = "frame_{:04d}".format(self.time)
            model_save_path = os.path.join(self.save_path, "save_obj", frame_name)
            os.makedirs(model_save_path, exist_ok=True)
            path = os.path.join(
                model_save_path,
                "iter_{:04d}".format(self.iter),
            )
        if save_data and self.Map_global is not None:
            from scene.dataset_readers import c2w0
            with open(path+".txt", 'w') as file:
                for obj in self.Map_global:
                    cat_id = obj.category_id_
                    R = obj.ellipsoid_.R_
                    center = obj.ellipsoid_.center_
                    pose = np.eye(4)
                    pose[:3, :3] = R
                    pose[:3, 3] = center
                    #pose = c2w0 @ pose
                    R = pose[:3, :3]
                    center = pose[:3, 3]
                    q = rot_to_quat(R)#w x y z

                    axes_ = obj.ellipsoid_.axes_
                    axes_ = axes_
                    #.join使用是迭代方式
                    file.write(f"{cat_id} {' '.join(map(str, center))} {' '.join(map(str, q[1:]))} {str(q[0])} {' '.join(map(str, axes_))}\n")



            file.close()



    def save_model(self, path=None, save_data=True, save_sibr=True, save_merge=True):
        if path == None:
            frame_name = "frame_{:04d}".format(self.time)
            model_save_path = os.path.join(self.save_path, "save_model", frame_name)
            os.makedirs(model_save_path, exist_ok=True)
            path = os.path.join(
                model_save_path,
                "iter_{:04d}".format(self.iter),
            )
        if save_data:
            self.pointcloud.save_model_ply(path + ".ply", include_confidence=True)
            self.stable_pointcloud.save_model_ply(
                path + "_stable.ply", include_confidence=True
            )
            ###2024-12-19 给每一个gs增加物体属性
            if SAVE_obj_ply:
                self.pointcloud.save_model_ply_obj(path + "_obj", include_confidence=False)
                self.stable_pointcloud.save_model_ply_obj(path + "_stable_obj", include_confidence=False)
        if save_sibr:
            self.pointcloud.save_model_ply(path + "_sibr.ply", include_confidence=False)
            self.stable_pointcloud.save_model_ply(
                path + "_stable_sibr.ply", include_confidence=False
            )
        if self.get_unstable_num > 0 and self.get_stable_num > 0:
            if save_data and save_merge:
                merge_ply(
                    path + ".ply",
                    path + "_stable.ply",
                    path + "_merge.ply",
                    include_confidence=True,
                )
            if save_sibr and save_merge:
                merge_ply(
                    path + "_sibr.ply",
                    path + "_stable_sibr.ply",
                    path + "_merge_sibr.ply",
                    include_confidence=False,
                )

    def train_report(self, iteration, losses):
        if self.tb_writer is not None:
            for loss in losses:
                self.tb_writer.add_scalar(
                    "train/{}".format(loss), losses[loss], iteration
                )

    def eval_report(self, iteration, losses):
        if self.tb_writer is not None:
            for loss in losses:
                self.tb_writer.add_scalar(
                    "eval/{}".format(loss), losses[loss], iteration
                )

    ### 2025-01-02 gui可视化 获得gaussain的参数
    def get_gaussian_for_gui(self, frame, frame_id, q_main2vis=None, global_opt=False, finish=False):
        frame_window.append(frame)
        # instance = frame.instance_img.clone()
        # instance = instance.sum(dim=0)
        # #instance = instance.permute(1, 2, 0).contiguous().cpu().numpy()
        # #instance = instance.squeeze(0)
        # instance = instance.numpy()
        # instance = instance.astype(np.float32)
        # instance_min, instance_max = instance.min(), instance.max()
        # instance_normalized = (instance - instance_min) / (instance_max - instance_min)
        # instance = imgviz.depth2rgb(instance_normalized,  min_value=0.0, max_value=1.0, colormap="jet")
        rgb = torch.clamp(frame.instance_img.clone(), min=0, max=1.0) * 255
        rgb = rgb.byte().permute(1, 2, 0).contiguous().cpu().numpy()
        #cv2.imwrite('/home/lihy/3DGS/RTG-SLAM/output/instance_image.png', rgb)

        if frame_id == self.keyframe_ids[-1]:
            paket = GaussianPacket(
                current_frame=frame,
                gaussians=clone_obj(self.global_params),
                gtcolor=frame.gui_color,
                gtdepth=frame.gui_depth,
                keyframe=self.keyframe_list[-1],
                keyframes=frame_window,
                instance_img=frame.instance_img,
                object = self.Map_global
            )
            frame_window.clear()
        else:
            paket = GaussianPacket(
                current_frame = frame,
                gaussians=clone_obj(self.global_params),
                gtcolor = frame.gui_color,
                gtdepth=frame.gui_depth,
                instance_img=frame.instance_img,
                object=self.Map_global
            )
        if finish:
            #paket = GaussianPacket(finish=True)
            paket = GaussianPacket(
                current_frame = frame,
                gaussians=clone_obj(self.global_params),
                gtcolor = frame.gui_color,
                gtdepth=frame.gui_depth,
                instance_img=frame.instance_img,
                object=self.Map_global,
                finish=True
            )
        q_main2vis.put(paket)
    def get_render_output(self, frame):
        render_output = self.renderer.render(
            frame, self.global_params,#self.global_params：将stable和unstable放在一起
        )
        self.model_map["render_color"] = render_output["render"].permute(1, 2, 0)
        self.model_map["render_depth"] = render_output["depth"].permute(1, 2, 0)
        self.model_map["render_normal"] = render_output["normal"].permute(1, 2, 0)
        self.model_map["render_color_index"] = render_output["color_index_map"].permute(
            1, 2, 0
        )
        self.model_map["render_depth_index"] = render_output["depth_index_map"].permute(
            1, 2, 0
        )
        self.model_map["render_transmission"] = render_output["T_map"].permute(1, 2, 0)
        if "semantic_seg" in render_output and render_output["semantic_seg"] is not None:
            self.model_map["semantic_seg"] = render_output["semantic_seg"].permute(1, 2, 0)

    @property
    def stable_params(self):
        have_stable = self.get_stable_num > 0
        xyz = self.stable_pointcloud.get_xyz if have_stable else torch.empty(0)
        opacity = self.stable_pointcloud.get_opacity if have_stable else torch.empty(0)
        scales = self.stable_pointcloud.get_scaling if have_stable else torch.empty(0)
        rotations = (
            self.stable_pointcloud.get_rotation if have_stable else torch.empty(0)
        )
        shs = self.stable_pointcloud.get_features if have_stable else torch.empty(0)
        radius = self.stable_pointcloud.get_radius if have_stable else torch.empty(0)
        normal = self.stable_pointcloud.get_normal if have_stable else torch.empty(0)
        confidence = (
            self.stable_pointcloud.get_confidence if have_stable else torch.empty(0)
        )
        semantics_color = self.stable_pointcloud.get_semantic if have_stable else torch.empty(0)
        instance = self.stable_pointcloud.get_instance if have_stable else torch.empty(0)
        stable_params = {
            "xyz": devF(xyz),
            "opacity": devF(opacity),
            "scales": devF(scales),
            "rotations": devF(rotations),
            "shs": devF(shs),
            "radius": devF(radius),
            "normal": devF(normal),
            "confidence": devF(confidence),
            "semantics_color": devF(semantics_color),
            "instance": devF(instance)
        }
        return stable_params

    @property
    #2024-11-21 获取和物体相关的参数
    def obj_params(self):
        have_obj = self.get_obj_num >0
        ## 使用原版RTG-SLAM的激活函数
        # xyz = self.object.get_xyz if have_obj else torch.empty(0)
        # opacity = self.object.get_opacity if have_obj else torch.empty(0)
        # scales = self.object.get_scaling if have_obj else torch.empty(0)
        # rotations = self.object.get_rotation if have_obj else torch.empty(0)
        # color = self.object.get_obj_color if have_obj else torch.empty(0)

        ## 直接将物体的各个参数返回
        xyz = self.object.get_xyz if have_obj else torch.empty(0)
        opacity = self.object.get_opacity_obj if have_obj else torch.empty(0)
        scales = self.object.get_scaling_obj if have_obj else torch.empty(0)
        scales = scales/1.5 ##2024-12-3 将轴长缩小为原来的3倍
        rotations = self.object.get_rotation if have_obj else torch.empty(0)
        color = self.object.get_obj_color if have_obj else torch.empty(0)

        obj_params = {
            "xyz": devF(xyz),
            "opacity": devF(opacity),
            "scales": devF(scales),
            "rotations": devF(rotations),
            "obj_color": devF(color)
        }
        return obj_params

    @property
    def unstable_params(self):
        have_unstable = self.get_unstable_num > 0
        xyz = self.pointcloud.get_xyz if have_unstable else torch.empty(0)
        opacity = self.pointcloud.get_opacity if have_unstable else torch.empty(0)
        scales = self.pointcloud.get_scaling if have_unstable else torch.empty(0)
        rotations = self.pointcloud.get_rotation if have_unstable else torch.empty(0)
        shs = self.pointcloud.get_features if have_unstable else torch.empty(0)
        radius = self.pointcloud.get_radius if have_unstable else torch.empty(0)
        normal = self.pointcloud.get_normal if have_unstable else torch.empty(0)
        confidence = self.pointcloud.get_confidence if have_unstable else torch.empty(0)
        semantics_color = self.pointcloud.get_semantic if have_unstable else torch.empty(0)
        instance = self.pointcloud.get_instance if have_unstable else torch.empty(0)
        unstable_params = {
            "xyz": devF(xyz),
            "opacity": devF(opacity),
            "scales": devF(scales),
            "rotations": devF(rotations),
            "shs": devF(shs),
            "radius": devF(radius),
            "normal": devF(normal),
            "confidence": devF(confidence),
            "semantics_color": devF(semantics_color),
            "instance": devF(instance)
        }
        return unstable_params

    @property
    def global_params_detach(self):
        unstable_params = self.unstable_params
        stable_params = self.stable_params
        for k in unstable_params:
            unstable_params[k] = unstable_params[k].detach()
        for k in stable_params:
            stable_params[k] = stable_params[k].detach()

        xyz = torch.cat([unstable_params["xyz"], stable_params["xyz"]])
        opacity = torch.cat([unstable_params["opacity"], stable_params["opacity"]])
        scales = torch.cat([unstable_params["scales"], stable_params["scales"]])
        rotations = torch.cat(
            [unstable_params["rotations"], stable_params["rotations"]]
        )
        shs = torch.cat([unstable_params["shs"], stable_params["shs"]])
        radius = torch.cat([unstable_params["radius"], stable_params["radius"]])
        normal = torch.cat([unstable_params["normal"], stable_params["normal"]])
        confidence = torch.cat(
            [unstable_params["confidence"], stable_params["confidence"]]
        )
        global_prams = {
            "xyz": xyz,
            "opacity": opacity,
            "scales": scales,
            "rotations": rotations,
            "shs": shs,
            "radius": radius,
            "normal": normal,
            "confidence": confidence,
        }
        return global_prams

    @property
    def global_params(self):
        unstable_params = self.unstable_params
        stable_params = self.stable_params

        xyz = torch.cat([unstable_params["xyz"], stable_params["xyz"]])
        opacity = torch.cat([unstable_params["opacity"], stable_params["opacity"]])
        scales = torch.cat([unstable_params["scales"], stable_params["scales"]])
        rotations = torch.cat(
            [unstable_params["rotations"], stable_params["rotations"]]
        )
        shs = torch.cat([unstable_params["shs"], stable_params["shs"]])
        radius = torch.cat([unstable_params["radius"], stable_params["radius"]])
        normal = torch.cat([unstable_params["normal"], stable_params["normal"]])
        confidence = torch.cat(
            [unstable_params["confidence"], stable_params["confidence"]]
        )
        semantics_color = torch.cat([unstable_params["semantics_color"], stable_params["semantics_color"]])
        instance = torch.cat([unstable_params["instance"], stable_params["instance"]])
        global_prams = {
            "xyz": xyz,
            "opacity": opacity,
            "scales": scales,
            "rotations": rotations,
            "shs": shs,
            "radius": radius,
            "normal": normal,
            "confidence": confidence,
            "semantics_color": semantics_color,
            "instance": instance
        }
        return global_prams


    @property
    def get_pixel_num(self):
        return (
            self.frame_map["depth_map"].shape[0] * self.frame_map["depth_map"].shape[1]
        )

    @property
    def get_total_iter(self):
        return self.iter + self.time * self.gaussian_update_iter

    @property
    def get_stable_num(self):
        return self.stable_pointcloud.get_points_num

    @property
    def get_unstable_num(self):
        return self.pointcloud.get_points_num

    @property
    #2024-11-21 获取物体数量
    def get_obj_num(self):
        return self.object.get_points_num

    @property
    def get_total_num(self):
        return self.get_stable_num + self.get_unstable_num

    @property
    def get_curr_frame(self):
        return self.optimize_frames_ids[-1]

    @property
    def get_keyframe_num(self):
        return len(self.keyframe_list)


class MappingProcess(Mapping):
    def __init__(self, map_params, optimization_params, slam):
        super().__init__(map_params)
        self.recorder = Recorder(map_params.device_list[0])
        print("finish init")

        self.slam = slam
        # tracker 2 mapper
        self._tracker2mapper_call = slam._tracker2mapper_call
        self._tracker2mapper_frame_queue = slam._tracker2mapper_frame_queue

        # mapper 2 system
        self._mapper2system_call = slam._mapper2system_call
        self._mapper2system_map_queue = slam._mapper2system_map_queue
        self._mapper2system_tb_queue = slam._mapper2system_tb_queue
        self._mapper2system_requires = slam._mapper2system_requires

        # mapper 2 tracker
        self._mapper2tracker_call = slam._mapper2tracker_call
        self._mapper2tracker_map_queue = slam._mapper2tracker_map_queue

        self._requests = [False, False]  # [frame process, global optimization]
        self._stop = False
        self.input = {}
        self.output = {}
        self.processed_tick = []
        self.time = 0
        self.optimization_params = optimization_params
        self._end = slam._end
        self.max_frame_id = -1

        self.finish = mp.Condition()

    def set_input(self):
        self.frame_map["depth_map"] = self.input["depth_map"]
        self.frame_map["color_map"] = self.input["color_map"]
        self.frame_map["normal_map_c"] = self.input["normal_map_c"]
        self.frame_map["normal_map_w"] = self.input["normal_map_w"]
        self.frame_map["vertex_map_c"] = self.input["vertex_map_c"]
        self.frame_map["vertex_map_w"] = self.input["vertex_map_w"]
        self.frame = self.input["frame"]
        self.time = self.input["time"]
        self.last_send_time = -1

    def send_output(self):
        self.output = {
            "pointcloud": self.pointcloud,
            "stable_pointcloud": self.stable_pointcloud,
            "time": self.time,
            "iter": self.iter,
        }
        print("send output: ", self.time)
        self._mapper2system_map_queue.put(copy.deepcopy(self.output))
        self._mapper2system_requires[1] = True
        with self._mapper2system_call:
            self._mapper2system_call.notify()

    def release_receive(self):
        while (
            not self._tracker2mapper_frame_queue.empty()
            and self._tracker2mapper_frame_queue.qsize() > 1
        ):
            x = self._tracker2mapper_frame_queue.get()
            self.max_frame_id = max(self.max_frame_id, x["time"])
            print("release: ", x["time"])
            if x["time"] == -1:
                self.input = x
            else:
                del x

    def pack_map_to_tracker(self):
        map_info = {
            "frame": copy.deepcopy(self.frame),
            "global_params": self.global_params_detach,
            "frame_id": self.processed_tick[-1],
        }
        print("mapper send map {} to tracker".format(self.processed_tick[-1]))
        with self._mapper2tracker_call:
            self._mapper2tracker_map_queue.put(map_info)
            self._mapper2tracker_call.notify()

    def run(self):
        while True:
            print("map run...")
            with self._tracker2mapper_call:
                if self._tracker2mapper_frame_queue.qsize() == 0:
                    print("waiting tracker to wakeup")
                    self._tracker2mapper_call.wait()
                self.input = self._tracker2mapper_frame_queue.get()
                self.max_frame_id = max(self.max_frame_id, self.input["time"])

            # TODO: debug input is None
            if "time" in self.input and self.input["time"] == -1:
                del self.input
                break
            
            # run frame map update
            self.set_input()
            self.processed_tick.append(self.time)
            self.mapping(self.frame, self.frame_map, self.input["time"], self.optimization_params)
            self.pack_map_to_tracker()


        # self.release_receive()
        self.global_optimization(self.optimization_params)
        self.time = -1
        self.send_output()
        print("processed frames: ", self.optimize_frames_ids)
        print("keyframes: ", self.keyframe_ids)
        self._end[1] = 1
        with self._mapper2system_call:
            self._mapper2system_call.notify()

        with self.finish:
            print("mapper wating finish")
            self.finish.wait()
        print("map finish")

    def stop(self):
        with self.finish:
            self.finish.notify()