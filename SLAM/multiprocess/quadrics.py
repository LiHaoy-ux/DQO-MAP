import json
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches
from scipy.cluster.hierarchy import single
from scipy.constants import sigma
from scipy.stats import false_discovery_control
from PIL import Image,ImageDraw
from plyfile import PlyData, PlyElement
from scipy.spatial.transform import Rotation as R
import random
import cv2
import sys
from torch import dtype
from scipy.ndimage import label
from torch import dtype, nn
from torch.cuda import device
import torch.optim as optim
from collections import deque
import math

DEBUG=False
show_grad = False
Record_img = False #是否记录渲染的图像
PLT_SHOW = False #是否显示plt图像
Only_IOU = True #只使用IOU计算关联

#?测试帧率，下列均为False
record_white_quadrics=False #将椭圆画在白色背景上，用于论文
save_img = True #记录关联的图像，和DEBUG不能共存
Use_own_color = True #使用自己的颜色(不使用json中的颜色)
Show_room_real = False #展示ROOM的真值3D
SHOW_3D = False #展示3D图像
img_quadric=np.array([])
img_render = np.array([])
#?使用RTG-SLAM的方式对齐image depth 和pose
def associate_frames(tstamp_image, tstamp_depth, tstamp_pose, tstsamp_detect, max_dt=0.08):
    associations = []
    for i, t in enumerate(tstamp_image):
        if tstamp_pose is None:
            j = np.argmin(np.abs(tstamp_depth - t))
            if np.abs(tstamp_depth[j] - t) < max_dt:
                associations.append((i, j))

        else:
            j = np.argmin(np.abs(tstamp_depth - t))
            k = np.argmin(np.abs(tstamp_pose - t))
            l = np.argmin(np.abs(tstsamp_detect - t))

            if (np.abs(tstamp_depth[j] - t) < max_dt) and (
                    np.abs(tstamp_pose[k] - t) < max_dt
            ) and (np.abs(tstsamp_detect[l] - t) < max_dt):
                associations.append((i, j, k,l))

    return associations

def check_bbox(bbox, H, W):
    '''
    检查bbox是否在图像范围内
    :param bbox:
    :param H:
    :param W:
    :return:
    '''
    bounding = 5
    if bbox[0] < bounding or bbox[1] < bounding or bbox[2] > W-bounding or bbox[3] > H-bounding:
        return False
    return True
def read_from_json(file_path, img_width, img_height, cam_infos=None):
    with open(file_path, 'r') as f:
        data = json.load(f)
    detections = []
    detect_timestamps = []

    sys.stdout.write("\r")

    idx_=0
    for entry in data:
        sys.stdout.write("\r")
        # the exact output you're looking for:
        sys.stdout.write("Reading object in camera {}/{}".format(idx_ + 1, len(data)))
        sys.stdout.flush()


        file_name = entry['file_name']
        time = float(os.path.splitext(file_name)[0])
        if type(time)==float:
            detect_timestamps.append(np.array(time, dtype=float))
        detections_info=[]

        #解决json文件可能和RGB图像长度不等的问题
        if cam_infos is not None:
            cam = cam_infos[idx_]
            time_stamp= float(cam.uid)
            while time_stamp<time:
                idx_+=1
                detection_info = None
                detections.append(detection_info)
                cam = cam_infos[idx_]
                time_stamp = float(cam.uid)

        idx_ += 1

        for detection in entry['detections']:
            if not check_bbox(detection['bbox'], img_height, img_width): continue
            detection_info = {
                "category_id": detection['category_id'],
                "detection_score": detection['detection_score'],
                "bbox": detection['bbox'],
                "label":None
            }
            if 'ellipse' in detection:
                detection_info["ellipse"] = detection['ellipse']
            if 'color' in detection:
                detection_info["color"] = detection['color']

            detections_info.append(detection_info)
        # 将 file_name 和 detections 保存到结果列表中
        detections.append({
            "file_name": file_name,
            "detections": detections_info
        })
    sys.stdout.write("\n")
    return detect_timestamps, detections
# def Ellipse(axes, angle, center):
#     axes_half = 0.5 * np.array(axes)
#     axes_square = np.array(axes_half ** 2)
#     axes_square = np.append(axes_square, -1)#?np增加一个元素
#     C_star = np.diag(axes_square)
#     T_center = np.eye(3)
#     T_center[:2,2] = center
#     Rw_e = np.array([[np.cos(angle), -np.sin(angle),0.0],
#                      [np.sin(angle), np.cos(angle),0.0],
#                     [0.0,0.0,1.0]])
#     transf = T_center*Rw_e
#     C_star = transf * C_star * transf.T
#     C_ = 0.5*(C_star + C_star.T)
#     C_/=-C_[2,2]
#     axex_ = axes_half
#     angle_=angle
#     center_ = center
#     has_changes = False


class Ellipse:
    def __init__(self,axes,angle,center):
        axes_half = 0.5 * np.array(axes)
        axes_square = np.array(axes_half ** 2)
        axes_square = np.append(axes_square, -1)  # ?np增加一个元素
        C_star = np.diag(axes_square)
        T_center = np.eye(3)
        T_center[:2, 2] = center
        Rw_e = np.array([[np.cos(angle), -np.sin(angle), 0.0],
                         [np.sin(angle), np.cos(angle), 0.0],
                         [0.0, 0.0, 1.0]])
        transf = T_center @ Rw_e
        C_star = transf @ C_star @ transf.T
        C_ = 0.5 * (C_star + C_star.T)
        C_ /= -C_[2, 2]
        if np.any(axes_half) == 0:
            print("there is a in axes")
        self.axes_ = axes_half
        self.angle_ = angle
        self.center_ = center
        self.has_changed_ = False
        self.C_ = C_
    @classmethod
    def Ellipse(cls,C):
        C =0.5*(C + C.T)
        instance = cls.__new__(cls)
        instance.C_ = C
        C /= -C[2, 2]
        instance.has_changed_ = True
        return instance
    def decompose(self):
        self.center_=-self.C_[:2,2]
        T_c = np.eye(3)
        T_c[:2,2] = -self.center_
        temp = T_c @ self.C_ @ T_c.T
        C_center = 0.5*(temp + temp.T)

        eig_vals, eig_vecs = np.linalg.eigh(C_center[:2, :2])
        if np.linalg.det(eig_vecs) < 0:
            eig_vecs[:, 1] *= -1
        if eig_vecs[0, 0] < 0:
            eig_vecs *= -1
        self.axes_ = np.sqrt(np.abs(eig_vals))
        self.angle_ = np.arctan2(eig_vecs[1, 0], eig_vecs[0, 0])
        self.has_changed_=False

    @classmethod
    def FromDual(cls, C):
        C_sys = 0.5*(C + C.T)
        instance = cls.__new__(cls)
        if np.abs(C_sys.T - C_sys).sum() > 1e-3:
            print("Warning: Matrix should be symmetric")

        C_sys /= -C_sys[2, 2]
        instance.C_ = C_sys
        instance.has_changed_ = True
        return instance
    def GetAngle(self):
        if(self.has_changed_):
            self.decompose()
            self.has_changed_ = False
        return self.angle_
    def GetCenter(self):
        if(self.has_changed_):
            self.decompose()
            self.has_changed_ = False
        return self.center_
    def GetAxes(self):
        if(self.has_changed_):
            self.decompose()
            self.has_changed_ = False
        return self.axes_


    def ComputeBbox(self):
        if(self.has_changed_):
            self.decompose()
            self.has_changed_ = False

        c = np.cos(self.angle_)
        s = np.sin(self.angle_)
        xmax = np.sqrt(self.axes_[0]**2*c**2+self.axes_[1]**2*s**2)
        ymax = np.sqrt(self.axes_[0]**2*s**2+self.axes_[1]**2*c**2)

        bbox = np.array([self.center_[0]-xmax,self.center_[1]-ymax,self.center_[0]+xmax,self.center_[1]+ymax])
        return bbox
    def AsGaussian(self):
        if(self.has_changed_):
            self.decompose()
            self.has_changed_ = False
        A_dual = np.array([
            [self.axes_[0]**2,0.0],
            [0.0,self.axes_[1]**2]
        ])
        R = np.array([
            [np.cos(self.angle_),-np.sin(self.angle_)],
            [np.sin(self.angle_),np.cos(self.angle_)]
        ])
        cov = R @ A_dual @ R.T
        cov = np.clip(cov, 0, None)
        return self.center_, cov
def get_2dim_quarics(detections_info):
    ellipse=[]
    cat_id=[]
    bboxes=[]
    detections=[]
    num =0
    for i in detections_info["detections"]:
        ell=i["ellipse"]
        cat=i["category_id"]
        bbox=i["bbox"]
        score=i["detection_score"]
        color=i["color"]##2024-11-28 增加json颜色信息
        try:
            ell=Ellipse(ell[2:4], ell[4], ell[0:2])
        except TypeError as e:
            print(f"错误：{e}。'ell' 对象可能为 None 或无效。")
            raise TypeError("请检查输入的 'ell' 对象。")

        detections.append(
            {
                "ell": ell,
                "cat": cat,
                "bbox": bbox,
                "score": score,
                "node_id": num,
                "obj":None,
                "color":color,##2024-11-28 增加json颜色信息
                "is_validate": True
            }
        )
        cat_id.append(cat)
        bboxes.append(bbox)
        num +=1
    return detections
def bbox_area(bbox):
    return (bbox[2]-bbox[0])*(bbox[3]-bbox[1])
def bboxes_iou(bb1, bb2):
    #?np.maximum()返回两个数组中对应位置的最大值（一定要是数组）
    inter_w = max(min(bb1[2], bb2[2])-max(bb1[0], bb2[0]),0)
    inter_h = max(min(bb1[3], bb2[3])-max(bb1[1], bb2[1]),0)
    area_inter = inter_w*inter_h
    return area_inter/(bbox_area(bb1)+bbox_area(bb2)-area_inter)
def bboxes_rotated_iou(bbox_obv, bbox):

    iou = torch.sqrt((bbox_obv[0]-bbox[0])**2 + (bbox_obv[1]-bbox[1])**2+
            (bbox_obv[2]-bbox[2])**2 + (bbox_obv[3]-bbox[3])**2)
    return iou
def is_cover(bb1, bb2):
    '''
    是否完全覆盖了bb1
    :param bb1:
    :param bb2:
    :return:
    '''
    inter_w = max(min(bb1[2], bb2[2])-max(bb1[0], bb2[0]),0)
    inter_h = max(min(bb1[3], bb2[3])-max(bb1[1], bb2[1]),0)
    area_inter = inter_w*inter_h
    # 避免除以零错误
    if bbox_area(bb1) == 0:
        return False

    return area_inter / bbox_area(bb1) > 0.5 and area_inter / bbox_area(bb2)<0.5

def bboxes_complete(bb1, bb2):
    '''
    检测是否为一个完整的bbox
    :param bb1:
    :param bb2:
    :return:
    '''
    inter_w = max(min(bb1[2], bb2[2])-max(bb1[0], bb2[0]),0)
    inter_h = max(min(bb1[3], bb2[3])-max(bb1[1], bb2[1]),0)
    area_inter = inter_w*inter_h
    if area_inter / bbox_area(bb1) > 0.95:
        return True
    else:
        return False
def bboxes_intersection(bb1, bb2):
    '''
    计算两个bbox的交集
    :param bb1:
    :param bb2:
    :return:
    '''
    inter_w = max(min(bb1[2], bb2[2]) - max(bb1[0], bb2[0]), 0)
    inter_h = max(min(bb1[3], bb2[3]) - max(bb1[1], bb2[1]), 0)
    return inter_h * inter_w
def detections_filter(detections, image_depth,W,H):
    current_frame_detections_ = []
    for i in detections:
        has_similar_det = False
        if i["score"]<0.2 or bbox_area(i["bbox"])<300\
                or bbox_area(i["bbox"])>0.5*H*W\
                or bboxes_iou(i["bbox"],i["ell"].ComputeBbox())<0.2:
            continue
        for j in current_frame_detections_:
            bbox_iou = bboxes_iou(i["bbox"],j["bbox"])
            if i["cat"] == j["cat"]:
                if bbox_iou>0.3:
                    has_similar_det = True
            else:
                if bbox_iou>0.6:
                    has_similar_det = True
        if not has_similar_det:
            current_frame_detections_.append(i)

    #计算平均深度
    current_depth_data_per_det_ = np.zeros((len(current_frame_detections_), 2))
    number_pixels = 30
    for i, det in enumerate(current_frame_detections_):
        bbox = det["bbox"]
        sum_d = 0.0
        min_d = 100.0
        max_d = -1.0
        count_avg = 0.0
        for _ in range(number_pixels):
            u = random.randint(int(bbox[0]), int(bbox[2]))
            v = random.randint(int(bbox[1]), int(bbox[3]))
            u = min(max(u, 0), W - 1)
            v = min(max(v, 0), H - 1)

            d = image_depth[v, u]
            if d>0.0:
                sum_d += d
                count_avg += 1.0
                if d<min_d:
                    min_d = d
                if d>max_d:
                    max_d = d
        if count_avg > 0.0:
            current_depth_data_per_det_[i][0] = min(sum_d / count_avg, 5.0)
            current_depth_data_per_det_[i][1] = min(max(max_d - min_d, 0.05), 0.2)
            if current_depth_data_per_det_[i][0]==0 or current_depth_data_per_det_[i][1]==0:
                print("HAVE A ZEOR IN DEPTH")



    return current_frame_detections_, current_depth_data_per_det_

class Ellipsoid:
    def __init__(self, axes, R, center):
        Q_star = np.diag([axes[0]**2, axes[1]**2, axes[2]**2, -1])
        T_center = np.eye(4)
        T_center[:3, 3] = center
        Rw_e = np.eye(4)
        Rw_e[:3, :3] = R
        #?将椭球自身平移和旋转结合到一起
        transf = T_center @ Rw_e
        #?得到椭球在世界坐标系下的Q_star矩阵
        Q_star = transf @ Q_star @ transf.T
        self.Q_ = 0.5 * (Q_star + Q_star.T)
        self.Q_ /= -self.Q_[3, 3]
        self.center_ = center
        self.axes_ = axes
        self.R_ = R

        self.has_changed_ = False
    def project(self,P):
        C = P @ self.Q_ @ P.T
        return Ellipse.FromDual(C)
    def decompose(self):
        self.center_ = -self.Q_[:3, 3]
        T_c = np.eye(4)
        T_c[:3, 3] = -self.center_
        temp = T_c @ self.Q_ @ T_c.T
        Q_center = 0.5 * (temp + temp.T)
        eig_vals, eig_vecs = np.linalg.eigh(Q_center[:3, :3])
        if np.linalg.det(eig_vecs) < 0:
            eig_vecs[:, 2] *= -1
        self.axes_ = np.sqrt(np.abs(eig_vals))
        self.R_ = eig_vecs
        self.has_changed_ = False
    def Get_Center(self):
        if(self.has_changed_):
            self.decompose()
            self.has_changed_ = False
        return self.center_


factory_id=0
class Object:
    def __init__(self,cat, bb, ell, score, depth_data, K, Rt, frame_idx, kf):
        global factory_id #?需要先说明全局变量
        self.id_ = factory_id
        factory_id+=1
        self.category_id_ = cat
        self.N_=1
        self.K_ = K
        self.last_obs_frame_id_ = frame_idx
        self.flag_optimized = False
        self.mbBad = False
        self.bboxes_ = [bb]
        self.Rts_ = [Rt]
        self.color = generate_random_color()
        self.last_obs_ids_and_max_iou= [-1,-1,-1]
        self.complete = True #是否是一个完整的物体

        ##2024-12-04 使用物体构建共视关系
        self.save_keyframe = deque(maxlen=3)  # 设置了一个双端队列，最大长度为memory_length
        self.save_keyframemap = deque(maxlen=3)
        self.frame_ids = deque(maxlen=3) #记录每一个帧的id

        avg_depth = depth_data[0]
        diff_depth = depth_data[1]
        bb_center = np.array([(bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2])
        u = (bb_center[0]-self.K_[0,2])/self.K_[0,0]
        v = (bb_center[1]-self.K_[1,2])/self.K_[1,1]
        bb_center_cam = np.array([u*avg_depth, v*avg_depth, avg_depth])
        Rcw = Rt[:3,:3]
        tcw = Rt[:3,3]
        center_world = Rcw.T @ bb_center_cam +(-Rcw.T @ tcw)
        #center_world = Rcw @ bb_center_cam + tcw

        #Rotation
        zc = bb_center_cam / np.linalg.norm(bb_center_cam)
        up_vec = np.array([0,-1,0])
        xc = np.cross(-up_vec, zc)
        xc = xc / np.linalg.norm(xc)
        yc = np.cross(zc, xc)
        rot_cam = np.eye(3)
        rot_cam[:,0] = xc
        rot_cam[:,1] = yc
        rot_cam[:,2] = zc
        rot_world = Rcw.T @ rot_cam

        #Axes
        width_in_img = bb[2] - bb[0]
        height_in_img = bb[3] - bb[1]
        width_in_world = width_in_img * avg_depth / self.K_[0,0]
        height_in_world = height_in_img * avg_depth / self.K_[1,1]
        axes = np.array([width_in_world*0.5, height_in_world*0.5, diff_depth*0.5])
        if np.any(axes) == 0:
            print("HABE A ZERO IN AXES")
        self.ellipsoid_= Ellipsoid(axes, rot_world, center_world)
        #?是否为关键帧，关键帧保存信息
        #!暂且不继续写了
        if(kf):
            self.ellipses_ = [ell]
            self.kf_Rts_ = [Rt]

class dect_results:
    ellipse = []
    cat_id = []
    def __init__(self,detections_info):
        detections=[]
        for i in detections_info["detections"]:
            ellipse = torch.tensor(i["ellipse"]).to('cuda')
            category_id = torch.tensor(i["category_id"]).to('cuda')
            bbox = torch.tensor(i["bbox"]).to('cuda')
            detection_score = torch.tensor(i["detection_score"]).to('cuda')
            ell = Ellipse(ellipse[2:4], ellipse[4], ellipse[0:2])
            dect(ell, category_id, bbox, detection_score, None)
            detections.append(dect)
        self.detections = detections
class dect:
    def __init__(self, ell, cat, bbox, score, obj):
        self.ell = ell
        self.cat = cat
        self.bbox = bbox
        self.score = score
        self.obj = None



#?进行椭球初始化
def ObjectsInitialization(current_frame_detections_, current_depth_data_per_det_, Rt, K):
    mpMap =[]
    count =0
    for i, det in enumerate(current_frame_detections_):
       if current_depth_data_per_det_[i][0] >0.0 \
           and current_depth_data_per_det_[i][0] <15.0:
           frame_idx=0
           kf = False
           obj = Object(det["cat"], det["bbox"], det["ell"], det["score"],
                        current_depth_data_per_det_[i], K, Rt, frame_idx, kf)
           mpMap.append(obj)
           if Use_own_color:
               obj.color = generate_random_color()
           else:
            ##2024-11-28 增加json颜色信息
            obj.color = det["color"]

            ## 2024-12-1
           det["node_id"] = count
           det["obj"] = obj

           count+=1
    if SHOW_3D:
        show_results_3D(mpMap)
    return mpMap

    print("Number of objects initialized: ", count, "Map has", len(mpMap), "objects")


# 2025-1-18 增加绘制立方体的代码
def generate_bbox_corners(scale, rotation, translation):
    """
    Generate the 8 corners of a 3D bounding box given its scale, rotation, and translation.

    Args:
        scale (numpy.ndarray): 3x1 scale vector representing lengths of the box along each axis.
        rotation (numpy.ndarray): 3x3 rotation matrix.
        translation (numpy.ndarray): 3x1 translation vector.

    Returns:
        numpy.ndarray: 8x3 array representing the corners of the bounding box.
    """
    # Local coordinates of a unit cube centered at the origin
    local_corners = np.array([
        [-0.5, -0.5, -0.5],
        [-0.5, -0.5,  0.5],
        [-0.5,  0.5, -0.5],
        [-0.5,  0.5,  0.5],
        [ 0.5, -0.5, -0.5],
        [ 0.5, -0.5,  0.5],
        [ 0.5,  0.5, -0.5],
        [ 0.5,  0.5,  0.5]
    ])

    # Scale, rotate, and translate
    scaled_corners = local_corners * scale
    rotated_corners = scaled_corners @ rotation.T
    translated_corners = rotated_corners + translation.T

    return translated_corners

bbox_edges = [
    [0, 1], [1, 3], [3, 2], [2, 0],  # Bottom face
    [4, 5], [5, 7], [7, 6], [6, 4],  # Top face
    [0, 4], [1, 5], [2, 6], [3, 7]   # Vertical edges
]

def show_results_3D(mpMap, debug = False):
    # if DEBUG:
    print("map global size: ", len(mpMap))
    ##debug：检查哪一个物体消失了
    ids = []
    for obj in mpMap:
        ids.append(obj.category_id_)
    print("ids:", ' '.join(map(str, ids)))

    from scene.dataset_readers import c2w0
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect([1, 1, 1])  # 保持比例一致
    ax.clear()
    for obj in mpMap:
        ell = obj.ellipsoid_
        radii = ell.axes_
        pose = np.eye(4)
        pose[:3, :3] = ell.R_
        pose[:3, 3] = ell.center_
        pose =  c2w0 @ pose
        # color = generate_random_color()
        #color.append(1.0)
        #obj.color = color
        SZ = 50
        u, v = np.linspace(0, 2 * np.pi, SZ), np.linspace(0, np.pi, SZ)
        x, y, z = (radii[0] * np.outer(np.cos(u), np.sin(v)),
                   radii[1] * np.outer(np.sin(u), np.sin(v)),
                   radii[2] * np.outer(np.ones_like(u), np.cos(v)))  # 生成网络
        ps = pose @ np.vstack([
            x.reshape(-1),
            y.reshape(-1),
            z.reshape(-1),
            np.ones(z.reshape(-1).shape)
        ])
        ax.plot_wireframe(
            ps[0, :].reshape(SZ, SZ),
            ps[1, :].reshape(SZ, SZ),
            ps[2, :].reshape(SZ, SZ),
            rstride=4,
            cstride=4,
            edgecolors=tuple(c / 255.0 for c in obj.color),
            linewidth=0.5,
        )
        if Show_room_real:
            # Add bounding box visualization
            bbox_corners = generate_bbox_corners(radii*2, pose[:3, :3], pose[:3, 3])
            for edge in bbox_edges:
                ax.plot(
                    bbox_corners[edge, 0],
                    bbox_corners[edge, 1],
                    bbox_corners[edge, 2],
                    color='r', linewidth=0.8
                )
        if debug:
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
            ax.autoscale()
            plt.show()
            plt.close(fig)  # 关闭当前图形窗口
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.clear()

    if Show_room_real:
        path = "/home/lihy/3DGS/RTG-SLAM/eval_obj/room_gt.txt"
        with open(path, 'r') as f:
            lines = f.readlines()
        for line in lines[1:]:
            line = line.strip().split()
            id = float(line[0])
            center = np.array([float(line[1]), float(line[2]), float(line[3])])
            q = np.array([float(line[4]), float(line[5]), float(line[6]), float(line[7])])
            rot = R.from_quat(q).as_matrix()
            axes = np.array([float(line[8]), float(line[9]), float(line[10])])
            # 生成立方体的顶点
            bbox_corners = generate_bbox_corners(axes*2, rot, center)

            # 绘制立方体
            for edge in bbox_edges:
                ax.plot(
                    [bbox_corners[edge[0], 0], bbox_corners[edge[1], 0]],
                    [bbox_corners[edge[0], 1], bbox_corners[edge[1], 1]],
                    [bbox_corners[edge[0], 2], bbox_corners[edge[1], 2]],
                    color='blue', linewidth=0.8, alpha=0.8
                )

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.autoscale()
    if PLT_SHOW:
        plt.show()
    plt.show()
def show_results(mpMap,object_info,K, Rt):
    """
    直接显示三维的结果
    :param obh_params:
    :return:
    """
    img_color = object_info["rgb"]
    draw = ImageDraw.Draw(img_color)
    for i in object_info["detections"]:
        bbox = i["bbox"]
        draw.rectangle(bbox, outline='red', width=2)
        x_center, y_center, width, height, angle = i["ellipse"]
        bbox = [
            (x_center - width / 2, y_center - height / 2),
            (x_center + width / 2, y_center + height / 2)
        ]
        # draw.ellipse(bbox, outline='red', width=2)
        # img_color = img_color.rotate(np.degrees(angle), center=(x_center, y_center), resample=Image.BICUBIC,
        #                              expand=True)
    P = K @ Rt

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # ax.set_box_aspect([1, 1, 1])  # 保持比例一致
    ax.clear()
    for obj in mpMap:
        ell = obj.ellipsoid_
        radii = ell.axes_
        pose = np.eye(4)
        pose[:3, :3] = ell.R_
        pose[:3, 3] = ell.center_
        color = generate_random_color()
        color.append(1.0)
        SZ = 50
        u, v = np.linspace(0, 2 * np.pi, SZ), np.linspace(0, np.pi, SZ)
        x, y, z = (radii[0] * np.outer(np.cos(u), np.sin(v)),
                   radii[1] * np.outer(np.sin(u), np.sin(v)),
                   radii[2] * np.outer(np.ones_like(u), np.cos(v)))  # 生成网络
        ps = pose @ np.vstack([
            x.reshape(-1),
            y.reshape(-1),
            z.reshape(-1),
            np.ones(z.reshape(-1).shape)
        ])
        ax.plot_wireframe(
            ps[0, :].reshape(SZ, SZ),
            ps[1, :].reshape(SZ, SZ),
            ps[2, :].reshape(SZ, SZ),
            rstride=4,
            cstride=4,
            edgecolors='blue',
            linewidth=0.5,
        )
    ax.set_xlabel('y')
    ax.set_ylabel('z')
    ax.set_zlabel('x')
    ax.autoscale()
    if PLT_SHOW:
        plt.pause(1)
        plt.show()

def save_model_ply2(obj_params,path, include_confidence=True):
    '''
    使用chat-gpt提供的方式保存模型
    :param obj_params:
    :param path:
    :param include_confidence:
    :return:
    '''
    # 将数据转换为numpy格式
    points = obj_params["xyz"].cpu().numpy()
    rotations = obj_params["rotation"].cpu().numpy()
    opacities = obj_params["opacity"].cpu().numpy()
    scalings = obj_params["scaling"].cpu().numpy()
    features = obj_params["features_dc"].cpu().numpy().reshape(-1, 3)
    vertices = np.array(
        [
            (
                points[i, 0], points[i, 1], points[i, 2],  # xyz坐标
                rotations[i, 0], rotations[i, 1], rotations[i, 2], rotations[i, 3],  # 旋转
                opacities[i, 0],  # 透明度
                scalings[i, 0], scalings[i, 1], scalings[i, 2],  # 缩放
                features[i, 0], features[i, 1], features[i, 2],  # 特征
            )
            for i in range(points.shape[0])
        ],
        dtype=[
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4'),
            ('opacity', 'f4'),
            ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'),
            ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4'),
        ]
    )
    vertex_element = PlyElement.describe(vertices, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def save_model_ply(obj_params,path, include_confidence=True):
    if len(obj_params)==0:
        return
    xyz_tensor = torch.tensor(obj_params["xyz"])
    f_dc_tensor = torch.tensor(obj_params["features_dc"])
    f_dc_tensor = f_dc_tensor.view(-1,1,3)
    opacity_tensor = torch.tensor(obj_params["opacity"])
    rotation_tensor = torch.tensor(obj_params["rotation"])
    scale_tensor = torch.tensor(obj_params["scaling"])

    xyz = xyz_tensor.detach().cpu().numpy()
    f_dc = f_dc_tensor.transpose(1,2).flatten(start_dim=1).contiguous().cpu().numpy()
    opaciteis = opacity_tensor.detach().cpu().numpy()
    rotation =  rotation_tensor.detach().cpu().numpy()
    scale = scale_tensor.detach().cpu().numpy()
    dtype_full = [
        (attribude, "f4")
        for attribude in construct_list_of_attributes(obj_params)
    ]
    elements = np.empty(xyz.shape[0], dtype= dtype_full)
    attributes = np.concatenate(
        (xyz, f_dc, opaciteis, scale, rotation),
        axis = 1,
    )
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, "vertex")
    PlyData([el]).write(path)

def save_Model(obj_params, path=None, save_data=True, save_sibr=True, save_merge=True):
    if path == None:
        print("HAVE NO PATH TO SAVE MODEL")
    else:
        save_model_ply(obj_params,path+".ply", include_confidence=True)
        #save_model_ply2(obj_params, path + ".ply", include_confidence=True)
def construct_list_of_attributes(obj_params):
    '''
    :param obj_params:
    :return:  RTG-SLAM中的self.construct_list_of_attributes
    '''
    l = ["x", "y", "z"]
    for i in range(obj_params["features_dc"].shape[1]* obj_params["features_dc"].shape[2]):
        l.append("f_dc_{}".format(i))
    l.append("opacity")
    for i in range(obj_params["scaling"].shape[1]):
        l.append("scale_{}".format(i))
    for i in range(obj_params["rotation"].shape[1]):
        l.append("rot_{}".format(i))
    return l
def Update_Map(mpMap, obj_params):
    '''
    从3DGS格式更新到VOOM的格式
    :param mpMap:
    :param obj_params:
    :return:
    '''
    obj_xyz, obj_rots, obj_scales = obj_params.get_obj_params
    obj_xyz = obj_xyz.detach().cpu().numpy()
    obj_rots = obj_rots.detach().cpu().numpy()
    obj_scales = obj_scales.detach().cpu().numpy()

    for i, obj in enumerate(mpMap):
        xyz = obj_xyz[i]
        rots = obj_rots[i]
        scales = obj_scales[i]
        obj.ellipsoid_ = Ellipsoid(scales, R.from_quat(rots).as_matrix(), xyz)


#从VOOM的quadrics格式转为3DGS的格式
def from_Quadircs_to_Mode(mpMap):
    obj_params={}
    xyz = []
    rots = []
    opacities =[]
    features = []
    scales =[]
    opacities=[]

    for obj in mpMap:
            ell = obj.ellipsoid_
            xyz.append(ell.center_)
            #rots.append(rot_to_quat(ell.R_))
            rots.append(R.from_matrix(ell.R_).as_quat())
            opacities.append(0.99)
            if  np.any(ell.axes_ == 0):
                print("HAVE A ZERO AXES")
            scales.append(ell.axes_)
            featue = generate_random_color()
            features.append(featue)

    xyz = np.array(xyz)
    rots = np.array(rots)
    opacities = np.array(opacities)
    scales = np.array(scales)
    features = np.array(features)

    obj_params={
        "xyz":torch.tensor(xyz).view(-1,3),
        "rotation":torch.tensor(rots).view(-1,4),
        "opacity":torch.tensor(opacities).view(-1,1),
        "scaling":torch.tensor(scales).view(-1,3),
        "features_dc":torch.tensor(features).view(-1,1,3),
    }
    return obj_params


# 随机生成 RGB 颜色
def generate_random_color():
    return [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]


def rot_to_quat(R):
    """
    将3x3旋转矩阵转换为四元数。
    输入：
        R - 3x3 旋转矩阵
    输出：
        四元数 (w, x, y, z)
    """
    # 计算旋转矩阵的迹
    trace = np.trace(R)

    if trace > 0:
        # 使用迹为正的情况，直接计算
        s = np.sqrt(trace + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    else:
        # 如果迹为负或接近负数，选择最大对角元素来避免除零错误
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s

    return np.array([w, x, y, z])


def Occlusions_Check(Map,K, Rt, W, H, frame_id=None):
    '''
    检测是否有遮挡，并去除遮挡
    :param Map:
    :param K:
    :param Rt:
    :param W:
    :param H:
    :return:
    '''
    if frame_id == 600:
        debug=False
    P=K @ Rt
    img_bbox = np.array([0,0,W,H])
    proj_bboxes = {}

    # ?检测是否有遮挡
    i = -1
    for obj in Map:
        proj = obj.ellipsoid_.project(P)
        c3d = obj.ellipsoid_.Get_Center()
        bb_proj = proj.ComputeBbox()
        z=Rt[2,:] @ np.append(c3d, 1)
        i += 1
        if z<0 or bboxes_intersection(bb_proj, img_bbox)<0.3*bbox_area(bb_proj):
            continue
        #proj_bboxes[obj] = proj
        # 2025-1-14 记录每一个物体的id
        proj_bboxes[obj] = (proj, i)
        hidden = []
        for it in proj_bboxes.items():
            if it[0] != obj and bboxes_iou(it[1][0].ComputeBbox(), bb_proj)>0.8:
                c2 = it[0].ellipsoid_.Get_Center()
                z2 = Rt[2,:] @ np.append(c2, 1)
                if z<z2:
                    hidden.append(it[0])
                else:
                    hidden.append(obj)
                break
        #删除被遮挡的椭球
        for hid in hidden:
            proj_bboxes.pop(hid)
    return proj_bboxes

def Calculate_distance(ell1,ell2, constant_C):
    '''
    计算两个二维椭球之间的距离
    :param ell1:
    :param ell2:
    :param constant_C:
    :return:
    '''
    mu1, sigma1 = ell1.AsGaussian()
    mu2, sigma2 = ell2.AsGaussian()

    sigma11 = np.sqrt(sigma1)
    s121 = sigma11 @ sigma2 @ sigma11
    sigma121 = np.sqrt(s121)

    d = np.linalg.norm(mu1 - mu2)**2 + np.trace(sigma1 + sigma2 - 2 * sigma121)
    if d <0:
        d=0
    return np.exp(-np.sqrt(d)/constant_C)


def Calculate_distance_tensor(ell1,ell2, constant_C):
    '''
    计算两个二维椭球之间的距离
    :param ell1:
    :param ell2:
    :param constant_C:
    :return:
    '''
    mu1, sigma1 = ell1.AsGaussian()
    mu2, sigma2 = ell2.AsGaussian()
    device = mu1.device
    mu2 = torch.tensor(mu2).to(device)
    sigma2 = torch.tensor(sigma2).to(device).float()
    sigma11 = torch.sqrt(sigma1)
    s121 = sigma11 @ sigma2 @ sigma11
    sigma121 = torch.sqrt(s121)

    d = torch.norm(mu1 - mu2) ** 2 + torch.trace(sigma1 + sigma2 - 2 * sigma121)
    d = torch.clamp(d, min=0.0)
    return d

if Only_IOU:
    def MatchObject(Map_global, cur_detections, cur_detections_depth, proj_bboxes, frame_id, image_color, K, Rt):
        debug =False
        if frame_id == 1069:
            debug = False

        if DEBUG or debug:
            image_np1 = image_color.cpu().numpy()
            image_color1 = Image.fromarray((image_np1 * 255).astype(np.uint8))
            img_color1 = image_color1.copy()
            draw1 = ImageDraw.Draw(img_color1)
        if save_img:
            image_np = image_color.cpu().numpy()
            image_color = Image.fromarray((image_np * 255).astype(np.uint8))
            img_color = image_color.copy()
            draw = ImageDraw.Draw(img_color)
        nmatches = 0
        new_detections = []

        for cur_order, det in enumerate(cur_detections):
            dis_max = -1
            iou_max = 0
            node_id = -1  # 2024-12-02 用来记录是Map_global中的哪个物体
            matched_obj = None
            bb_det = det["bbox"]
            cat_id = det["cat"]  ##2024-12-05解决物体遮挡问题，获取物体种类

            # #测试图像
            # img_color = image_color.copy()
            # draw = ImageDraw.Draw(img_color)
            if DEBUG or debug:
                img_color1 = image_color1.copy()
                draw1 = ImageDraw.Draw(img_color1)
                draw1.rectangle(bb_det, outline='red', width=2)
                plt.imshow(img_color1)
                plt.show()
            if save_img:
                draw.rectangle(bb_det, outline='red', width=2)
            # find the best match
            i = -1  # 用来索引在map_global的对象
            for obj, arr in proj_bboxes.items():
                proj =arr[0]
                i=arr[1]
                if DEBUG or debug:
                    bb_proj = proj.ComputeBbox()
                    draw1.rectangle([(bb_proj[0] - 3, bb_proj[1] - 3), (bb_proj[2] + 3, bb_proj[3] + 3)],
                                   outline='yellow',
                                   width=2)
                    if PLT_SHOW or debug:
                        plt.imshow(img_color1)
                        plt.show()
                # i += 1
                # 2024-12-05 解决物体遮挡问题
                obj_cat = obj.category_id_
                iou = bboxes_iou(proj.ComputeBbox(), bb_det)
                if obj_cat == cat_id and iou < 0.5:
                    if is_cover(proj.ComputeBbox(), bb_det):  # 如果检测到的物体比已经保存到的大
                        ##新建一个物体替换map_global中的
                        obj_new = Object(det["cat"], det["bbox"], det["ell"], det["score"],
                                         cur_detections_depth[cur_order], K, Rt, frame_id, False)
                        matched_obj = obj_new
                        node_id = i
                        dis_max = 0
                        iou_max = 1
                        Map_global[i] = obj_new

                        if save_img:
                            P = K @ Rt
                            proj = obj_new.ellipsoid_.project(P)
                            bb_proj = proj.ComputeBbox()
                            draw.rectangle([(bb_proj[0], bb_proj[1]), (bb_proj[2], bb_proj[3])], outline='yellow', \
                                           width=2)
                        if DEBUG or debug:
                            P = K @ Rt
                            proj = obj_new.ellipsoid_.project(P)
                            bb_proj = proj.ComputeBbox()
                            draw1.rectangle([(bb_proj[0], bb_proj[1]), (bb_proj[2], bb_proj[3])], outline='black', \
                                           width=2)
                            plt.imshow(img_color1)
                            plt.show()

                        break

                    elif is_cover(bb_det, proj.ComputeBbox()):  # 如果保存的物体比检测到的大
                        det["is_validate"] = False
                        matched_obj = None
                        iou_max=0
                        dis_max = 0
                        if save_img:
                            bb_proj = proj.ComputeBbox()
                            draw.rectangle([(bb_proj[0], bb_proj[1]), (bb_proj[2], bb_proj[3])], outline='pink',
                                           width=2)
                        if DEBUG or debug:
                            bb_proj = proj.ComputeBbox()
                            draw1.rectangle([(bb_proj[0], bb_proj[1]), (bb_proj[2], bb_proj[3])], outline='purple',
                                           width=2)
                            plt.imshow(img_color1)
                            plt.show()
                        break

                wasser_dis = Calculate_distance(proj, det["ell"], 10)

                if  iou > iou_max and iou >0.5:
                    dis_max = wasser_dis
                    iou_max = iou
                    matched_obj = obj
                    node_id = i  # 2024-12-02 是Map_global中的哪一个物体
                    # ? 检查是否原来的检测是物体的一部分
                    if DEBUG or debug:
                        # 测试图像
                        bb_proj = proj.ComputeBbox()
                        draw1.rectangle([(bb_proj[0]-6, bb_proj[1]-6), (bb_proj[2]+6, bb_proj[3]+6)], outline=tuple(matched_obj.color), width=4)
                        plt.imshow(img_color1)
                        plt.show()
                    if save_img:
                        bb_proj = proj.ComputeBbox()
                        draw.rectangle([(bb_proj[0], bb_proj[1]), (bb_proj[2], bb_proj[3])], outline=tuple(matched_obj.color), width=2)

            # 是否已经在前几个检测中匹配到了
            # if dis_max > 0.00001:#可能会出现1e-7的情况
            if iou_max > 0.5:
                try:
                    matched_obj.last_obs_ids_and_max_iou[0] == frame_id
                except Exception as e:
                    print("Error: ", e)
                    matched_obj.last_obs_ids_and_max_iou = [frame_id, node_id, dis_max]
                if matched_obj.last_obs_ids_and_max_iou[0] == frame_id:
                    iou_last = matched_obj.last_obs_ids_and_max_iou[2]
                    if iou_max < iou_last:
                        continue
                    else:
                        cur_detections[matched_obj.last_obs_ids_and_max_iou[1]]["obj"] = None
                        nmatches -= 1

                nmatches += 1
                # node_id = det["node_id"]
                det["node_id"] = node_id  ## 2024-12-01 记录这个检测框对应的是哪个物体
                # obj.last_obs_ids_and_max_iou = [frame_id, node_id, dis_max]
                det["obj"] = matched_obj
                det["obj"].last_obs_ids_and_max_iou = [frame_id, cur_order, iou_max]

                ##2024-12-23 对每一个物体独立优化 保存检测结果
                proj = Map_global[node_id].ellipsoid_.project(K @ Rt)
                iou = bboxes_iou(proj.ComputeBbox(), bb_det)
                if(iou < 0.01) and det["is_validate"] is False: continue
                axes = proj.GetAxes()
                if (axes[0] <= 0.001 or axes[1] <= 0.001): continue;
                Map_global[node_id].bboxes_.append(det["bbox"])
                Map_global[node_id].Rts_.append(Rt)
        if SHOW_3D:
            print(nmatches)

        # initialization new objects
        num = len(Map_global)  ## 2024-12-01 记录数量
        has_new_object = False  # 是否存在新的物体
        for i, det in enumerate(cur_detections):
            if det["obj"] is None and det["is_validate"]:
                if cur_detections_depth[i][0] > 0.01 and cur_detections_depth[i][0] < 15.0:
                    if DEBUG or debug:
                        img_color1 = image_color1.copy()
                        draw1 = ImageDraw.Draw(img_color1)
                        bb_det = det["bbox"]
                        draw1.rectangle(bb_det, outline='red', width=2)
                        plt.imshow(img_color1)
                        plt.show()
                    obj = Object(det["cat"], det["bbox"], det["ell"], det["score"],
                                 cur_detections_depth[i], K, Rt, frame_id, False)
                    Map_global.append(obj)
                    has_new_object = True
                    # 2024-12-01 记录
                    det["node_id"] = num
                    num += 1

                    det["obj"] = obj
                    obj.color = det["color"]
                    if DEBUG or debug:
                        P = K @ Rt
                        proj = obj.ellipsoid_.project(P)
                        bb_proj = proj.ComputeBbox()
                        draw1.rectangle([(bb_proj[0] - 2, bb_proj[1] - 2), (bb_proj[2] + 2, bb_proj[3] + 2)],
                                       outline='green', \
                                       width=4)
                        draw1.rectangle([(bb_proj[0] - 2, bb_proj[1] - 2), (bb_proj[2] + 2, bb_proj[3] + 2)],
                                       outline=tuple(obj.color), \
                                       width=2)
                        plt.imshow(img_color1)
                        plt.show()
                    if save_img:
                        P = K @ Rt
                        proj = obj.ellipsoid_.project(P)
                        bb_proj = proj.ComputeBbox()
                        draw.rectangle([(bb_proj[0] - 6, bb_proj[1] - 6), (bb_proj[2] + 6, bb_proj[3] + 6)],
                                       outline='green', \
                                       width=4)
                        draw.rectangle([(bb_proj[0] - 2, bb_proj[1] - 2), (bb_proj[2] + 2, bb_proj[3] + 2)],
                                       outline=tuple(obj.color), \
                                       width=2)

        # if save_img:
        #     output_path = "/home/lihy/3DGS/RTG-SLAM/output/object/" + "match" + str(frame_id) + ".jpg"
        #     img_color.save(output_path)

        # DEBUG:show the 3D results
        if DEBUG:
            show_results_3D(Map_global)
        return has_new_object, Map_global

else:
    def MatchObject(Map_global,cur_detections, cur_detections_depth,proj_bboxes,frame_id,image_color, K, Rt):
        # if DEBUG:
        #     image_np = image_color.cpu().numpy()
        #     image_color = Image.fromarray((image_np * 255).astype(np.uint8))
        #     img_color = image_color.copy()
        #     draw = ImageDraw.Draw(img_color)
        # if save_img:
        #     image_np = image_color.cpu().numpy()
        #     image_color = Image.fromarray((image_np * 255).astype(np.uint8))
        #     img_color = image_color.copy()
        #     draw = ImageDraw.Draw(img_color)
        nmatches = 0
        new_detections =[]

        for cur_order, det in enumerate(cur_detections):
            dis_max = -1
            node_id = -1  # 2024-12-02 用来记录是Map_global中的哪个物体
            matched_obj = None
            bb_det = det["bbox"]
            cat_id = det["cat"]##2024-12-05解决物体遮挡问题，获取物体种类

            # #测试图像
            # img_color = image_color.copy()
            # draw = ImageDraw.Draw(img_color)
            # if DEBUG:
            #     img_color = image_color.copy()
            #     draw = ImageDraw.Draw(img_color)
            #     draw.rectangle(bb_det, outline='red', width=2)
            #     plt.imshow(img_color)
            #     plt.show()
            # if save_img:
            #     draw.rectangle(bb_det, outline='red', width=2)
            # #find the best match
            i=-1 #用来索引在map_global的对象
            for obj, arr in proj_bboxes.items():
                proj =arr[0]
                i = arr[1]
                # if DEBUG:
                #     bb_proj = proj.ComputeBbox()
                #     draw.rectangle([(bb_proj[0] - 3, bb_proj[1] - 3), (bb_proj[2] + 3, bb_proj[3] + 3)], outline='yellow',
                #                width=2)
                #     plt.imshow(img_color)
                #     plt.show()
                # i+=1
                # 2024-12-05 解决物体遮挡问题
                obj_cat = obj.category_id_
                iou = bboxes_iou(proj.ComputeBbox(), bb_det)
                if obj_cat == cat_id and iou<0.5:
                    if is_cover(proj.ComputeBbox(), bb_det):  # 如果检测到的物体比已经保存到的大
                        ##新建一个物体替换map_global中的
                        obj_new = Object(det["cat"], det["bbox"], det["ell"], det["score"],
                                     cur_detections_depth[cur_order], K, Rt, frame_id, False)
                        matched_obj = obj_new
                        node_id = i
                        dis_max = 0
                        Map_global[i] = obj_new

                        # if save_img:
                        #     P = K @ Rt
                        #     proj = obj_new.ellipsoid_.project(P)
                        #     bb_proj = proj.ComputeBbox()
                        #     draw.rectangle([(bb_proj[0], bb_proj[1]), (bb_proj[2], bb_proj[3])], outline='yellow', \
                        #                    width=2)
                        # if DEBUG:
                        #     P = K @ Rt
                        #     proj = obj_new.ellipsoid_.project(P)
                        #     bb_proj = proj.ComputeBbox()
                        #     draw.rectangle([(bb_proj[0], bb_proj[1]), (bb_proj[2], bb_proj[3])], outline='black', \
                        #                    width=2)
                        #     plt.imshow(img_color)
                        #     plt.show()

                        break

                    elif is_cover(bb_det, proj.ComputeBbox()):  # 如果保存的物体比检测到的大
                        det["is_validate"] = False
                        matched_obj = None
                        dis_max = 0
                        # if save_img:
                        #     bb_proj = proj.ComputeBbox()
                        #     draw.rectangle([(bb_proj[0], bb_proj[1]), (bb_proj[2], bb_proj[3])], outline='pink', width=2)
                        # if DEBUG:
                        #     bb_proj = proj.ComputeBbox()
                        #     draw.rectangle([(bb_proj[0], bb_proj[1]), (bb_proj[2], bb_proj[3])], outline='purple', width=2)
                        #     plt.imshow(img_color)
                        #     plt.show()
                        break

                wasser_dis = Calculate_distance(proj, det["ell"], 10)

                if wasser_dis>dis_max and iou > 0.001:
                    dis_max = wasser_dis
                    matched_obj = obj
                    node_id = i  # 2024-12-02 是Map_global中的哪一个物体
                    # #? 检查是否原来的检测是物体的一部分
                    # if DEBUG:
                    #     # 测试图像
                    #     bb_proj = proj.ComputeBbox()
                    #     draw.rectangle([(bb_proj[0], bb_proj[1]), (bb_proj[2], bb_proj[3])], outline='blue', width=2)
                    #     plt.imshow(img_color)
                    #     plt.show()
                    if save_img:
                        bb_proj = proj.ComputeBbox()
                        #draw.rectangle([(bb_proj[0], bb_proj[1]), (bb_proj[2], bb_proj[3])], outline=matched_obj.color, width=2)

            #是否已经在前几个检测中匹配到了
            #if dis_max > 0.00001:#可能会出现1e-7的情况
            if dis_max > 0.001:
                if matched_obj.last_obs_ids_and_max_iou[0] == frame_id:
                    iou_last = matched_obj.last_obs_ids_and_max_iou[2]
                    if dis_max < iou_last:
                        continue
                    else:
                        cur_detections[matched_obj.last_obs_ids_and_max_iou[1]]["obj"] = None
                        nmatches -= 1



                nmatches += 1
                # node_id = det["node_id"]
                det["node_id"] = node_id  ## 2024-12-01 记录这个检测框对应的是哪个物体
                # obj.last_obs_ids_and_max_iou = [frame_id, node_id, dis_max]
                det["obj"] = matched_obj
                det["obj"].last_obs_ids_and_max_iou =[frame_id, cur_order, dis_max]

                ##2024-12-23 对每一个物体独立优化 保存检测结果
                proj = Map_global[node_id].ellipsoid_.project(K @ Rt)
                iou = bboxes_iou(proj.ComputeBbox(), bb_det)
                if(iou < 0.01): continue
                axes = proj.GetAxes()
                if (axes[0] <= 0.001 or axes[1] <= 0.001): continue;
                Map_global[node_id].bboxes_.append(det["bbox"])
                Map_global[node_id].Rts_.append(Rt)



        print(nmatches)

        #initialization new objects
        num = len(Map_global) ## 2024-12-01 记录数量
        has_new_object = False # 是否存在新的物体
        for i,det in enumerate(cur_detections):
            if det["obj"] is None and det["is_validate"]:
                if cur_detections_depth[i][0] > 0.01 and cur_detections_depth[i][0] < 15.0:
                    # if DEBUG:
                    #     img_color = image_color.copy()
                    #     draw = ImageDraw.Draw(img_color)
                    #     draw.rectangle(bb_det, outline='red', width=2)
                    #     plt.imshow(img_color)
                    #     plt.show()
                    obj = Object(det["cat"], det["bbox"], det["ell"], det["score"],
                             cur_detections_depth[i], K, Rt, frame_id, False)
                    Map_global.append(obj)
                    has_new_object = True
                    #2024-12-01 记录
                    det["node_id"] = num
                    num+=1

                    det["obj"] = obj
                    if Use_own_color:
                        obj.color = generate_random_color()
                    else:
                        obj.color = det["color"]
                    if DEBUG:
                        P = K @ Rt
                        proj = obj.ellipsoid_.project(P)
                        bb_proj = proj.ComputeBbox()
                        # draw.rectangle([(bb_proj[0]-2 , bb_proj[1]-2 ), (bb_proj[2]+2 , bb_proj[3]+2)], outline='green',\
                        #        width=4)
                        #plt.imshow(img_color)
                        #plt.show()
                    if save_img:
                        P = K @ Rt
                        proj = obj.ellipsoid_.project(P)
                        bb_proj = proj.ComputeBbox()
                        # draw.rectangle([(bb_proj[0]-4 , bb_proj[1]-4 ), (bb_proj[2]+4 , bb_proj[3]+4)], outline='green',\
                        #        width=4)
                        # draw.rectangle([(bb_proj[0]-2 , bb_proj[1]-2 ), (bb_proj[2]+2 , bb_proj[3]+2)], outline=obj.color,\
                        #        width=2)

        #if save_img:
            #output_path = "/home/lihy/3DGS/RTG-SLAM/output/object/"+"match"+str(frame_id)+".jpg"
            #img_color.save(output_path)


        # DEBUG:show the 3D results
        # if DEBUG:
        #     show_results_3D(Map_global)
        return has_new_object, Map_global


def plot_ellipsoid(ax, center, scale, rotation,map):
    # 创建单位球
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    SZ=50
    x, y, z = (scale[0] * np.outer(np.cos(u), np.sin(v)),
               scale[1] * np.outer(np.sin(u), np.sin(v)),
               scale[2] * np.outer(np.ones_like(u), np.cos(v)))
    pose = np.eye(4)
    pose[:3, :3] = R.from_quat(rotation).as_matrix()
    pose[:3, 3] = center

    ps = pose @ np.vstack([
        x.reshape(-1),
        y.reshape(-1),
        z.reshape(-1),
        np.ones(z.reshape(-1).shape)
    ])
    ax.plot_wireframe(
        ps[0, :].reshape(SZ, SZ),
        ps[1, :].reshape(SZ, SZ),
        ps[2, :].reshape(SZ, SZ),
        rstride=4,
        cstride=4,
        edgecolors='red',
        linewidth=0.5,
    )

    ell = map.ellipsoid_
    radii = ell.axes_
    pose = np.eye(4)
    pose[:3, :3] = ell.R_
    pose[:3, 3] = ell.center_
    color = generate_random_color()
    color.append(1.0)
    SZ = 50
    u, v = np.linspace(0, 2 * np.pi, SZ), np.linspace(0, np.pi, SZ)
    x, y, z = (radii[0] * np.outer(np.cos(u), np.sin(v)),
               radii[1] * np.outer(np.sin(u), np.sin(v)),
               radii[2] * np.outer(np.ones_like(u), np.cos(v)))  # 生成网络
    ps = pose @ np.vstack([
        x.reshape(-1),
        y.reshape(-1),
        z.reshape(-1),
        np.ones(z.reshape(-1).shape)
    ])
    ax.plot_wireframe(
        ps[0, :].reshape(SZ, SZ),
        ps[1, :].reshape(SZ, SZ),
        ps[2, :].reshape(SZ, SZ),
        rstride=4,
        cstride=4,
        edgecolors='blue',
        linewidth=0.5,
    )
##2024-11-27 展示渲染后的图像
def show_render(render_output):
    img_color =render_output["render_obj"].permute(1, 2, 0)
    image_np = img_color.detach().cpu().numpy()
    image_color = Image.fromarray((image_np * 255).astype(np.uint8))
    plt.imshow(image_color)
    plt.show()

    ##2024-11-30 比较椭圆大小
    global img_quadric, img_render
    img_render = image_np.copy()
    img_result = img_render + img_quadric
    img_result = np.clip(img_result, 0, 1)  # 限制在 [0, 1] 范围内
    img_result = Image.fromarray((img_result * 255).astype(np.uint8))
    plt.imshow(img_result)
    plt.show()


def plot_net(axes, center, theta_rad, color, img):
    a,b =axes
    theta = 30  # 椭圆旋转角度（度）
    num_radial = 8  # 径向网格线数量
    num_ellipses = 3  # 椭圆数量（环形网格）
    # 生成网格点
    angles = np.linspace(0, 2 * np.pi, num_radial, endpoint=False)  # 环形方向
    rotation_matrix = np.array([
        [np.cos(theta_rad), -np.sin(theta_rad)],
        [np.sin(theta_rad), np.cos(theta_rad)]
    ])
    theta = np.degrees(theta_rad)
    # 绘制环形网格线
    for i in range(1, num_ellipses + 1):
        current_a = int(a * i / num_ellipses)  # 当前椭圆的长轴半径
        current_b = int(b * i / num_ellipses)  # 当前椭圆的短轴半径
        cv2.ellipse(
            img, (int(center[0]), int(center[1])), axes=(current_a, current_b), angle=theta,
            startAngle=0, endAngle=360, color=color, thickness=2
        )
    # 绘制径向网格线
    for angle in angles:
        x_start, y_start = center
        x_end = center[0] + a * np.cos(angle)
        y_end = center[1] + b * np.sin(angle)
        rotated_start = rotation_matrix @ np.array([0, 0])
        rotated_end = rotation_matrix @ np.array([x_end - center[0], y_end - center[1]])
        start_point = (int(rotated_start[0] + center[0]), int(rotated_start[1] + center[1]))
        end_point = (int(rotated_end[0] + center[0]), int(rotated_end[1] + center[1]))
        cv2.line(img, start_point, end_point, color=color, thickness=2)

def plot_ellipse_3d_net(global_map, img_color, K, Rt,frame_id,opt=None, gui_use=False):
    if frame_id ==599:
        debug = True
    if gui_use:
        img_color = np.copy(img_color)
    else:
        image_np = img_color.cpu().numpy()
        image_color = Image.fromarray((image_np * 255).astype(np.uint8))
        img_color = image_color.copy()
        img_color = np.array(img_color)  # 将 PIL 图像转换为 numpy 数组
    if record_white_quadrics:
        white_image = np.ones_like(img_color) * 255
    # # 创建图形和坐标轴
    # fig, ax = plt.subplots(figsize=(8, 8))
    # ax.set_aspect('equal')
    # ax.set_xlim(np.min(x_proj) - 1, np.max(x_proj) + 1)
    # ax.set_ylim(np.min(y_proj) - 1, np.max(y_proj) + 1)
    #plt.title("3D Ellipsoid Projection with Camera")
    outlier = np.max([img_color.shape[0], img_color.shape[1]])
    lines = []
    # 初始化绘图范围
    x_min, x_max = np.inf, -np.inf
    y_min, y_max = np.inf, -np.inf

    for obj in global_map:
        a,b,c = obj.ellipsoid_.axes_
        # 生成椭球网格
        theta = np.linspace(0, 2 * np.pi, 20)  # 经度角
        phi = np.linspace(0, np.pi, 20)  # 纬度角
        theta, phi = np.meshgrid(theta, phi)

        # 参数方程转换为笛卡尔坐标
        x = a * np.sin(phi) * np.cos(theta)
        y = b * np.sin(phi) * np.sin(theta)
        z = c * np.cos(phi)

        # 将坐标展平为3xN数组
        points = np.vstack([x.ravel(), y.ravel(), z.ravel()])

        # 绕 Z 轴旋转 90 度的旋转矩阵
        theta = np.radians(90)  # 90 度转换为弧度
        R_swap = np.array([[0, 0, 1],
                           [0, 1, 0],
                           [1, 0, 0]])

        # 椭球的变换矩阵 Rt_obj
        Rt_obj = np.eye(4)
        Rt_obj[:3, :3] = obj.ellipsoid_.R_  # 椭球的旋转矩阵
        Rt_obj[:3, 3] = obj.ellipsoid_.center_  # 椭球的平移向量


        # 相机的变换矩阵 Rt
        Rt_new = np.eye(4)
        Rt_new[:3, :3] = Rt[:3,:3]  # 相机的旋转矩阵
        Rt_new[:3, 3] = Rt[:3,3]  # 相机的平移向量

        Rt_combined = Rt @ Rt_obj  # 先应用椭球变换，再应用相机变换
        P = K @ Rt_combined[:3, :]  # P = K * [R|t] * Rt_obj

        # 将椭球的点变换到相机坐标系
        points_homogeneous = np.vstack([points, np.ones(points.shape[1])])  # 齐次坐标
        points_camera = P @ points_homogeneous  # 投影到相机坐标系

        # 归一化处理（除以 Z 坐标）
        points_camera = points_camera[:2, :] / points_camera[2, :]

        # 提取投影坐标并重塑形状
        x_proj = points_camera[0].reshape(x.shape)
        y_proj = points_camera[1].reshape(y.shape)

        # # 更新绘图范围
        # x_min = min(x_min, np.min(x_proj))
        # x_max = max(x_max, np.max(x_proj))
        # y_min = min(y_min, np.min(y_proj))
        # y_max = max(y_max, np.max(y_proj))
        color = tuple(c * 255 / 255 for c in obj.color)
        # # 预创建所有线对象
        # for _ in range(x.shape[0] + x.shape[1]):
        #     line, = ax.plot([], [], 'b-', lw=0.5)
        #     lines.append(line)

        # 更新经线
        for i in range(x.shape[0]):
            # lines[i].set_data(x_proj[i, :], y_proj[i, :])
            x_proj_int = np.round(x_proj[i, :]).astype(int)
            y_proj_int = np.round(y_proj[i, :]).astype(int)
            for j in range(1, len(x_proj_int)):
                cv2.line(img_color, (x_proj_int[j - 1], y_proj_int[j - 1]),
                         (x_proj_int[j], y_proj_int[j]), color, 1)  # 画蓝色线
                if record_white_quadrics:
                    cv2.line(white_image, (x_proj_int[j - 1], y_proj_int[j - 1]),
                             (x_proj_int[j], y_proj_int[j]), color, 1)

        # 更新纬线
        offset = x.shape[0]
        for j in range(x.shape[1]):
            # lines[offset + j].set_data(x_proj[:, j], y_proj[:, j])

            x_proj_int = np.round(x_proj[:, j]).astype(int)
            y_proj_int = np.round(y_proj[:, j]).astype(int)
            for i in range(1, len(x_proj_int)):
                cv2.line(img_color, (x_proj_int[i - 1], y_proj_int[i - 1]),
                         (x_proj_int[i], y_proj_int[i]), color, 1)  # 画蓝色线
                if record_white_quadrics:
                    cv2.line(white_image, (x_proj_int[i - 1], y_proj_int[i - 1]),
                            (x_proj_int[i], y_proj_int[i]), color, 1)
    # cv2.imwrite("/home/lihy/3DGS/RTG-SLAM/output/object/" + str(frame_id) + "just.png", img_color)
    # if record_white_quadrics:
    #     cv2.imwrite("/home/lihy/3DGS/RTG-SLAM/output/object/" + str(frame_id) + "white.png", white_image)

    if gui_use:
        gui_img_color = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)
        return gui_img_color
    # # 动态设置绘图范围
    # ax.set_xlim(x_min - 1, x_max + 1)
    # ax.set_ylim(y_min - 1, y_max + 1)
    # 在循环外部调用 fig.canvas.draw_idle()
    #fig.canvas.draw_idle()
    #plt.show()

def plot_ellipse_2d_net(global_map, img_color, K, Rt,frame_id,opt=None, gui_use=False):
    '''
    绘制二维椭球在图像上
    :param global_map:
    :param img_color:
    :return:
    '''
    if frame_id ==599:
        debug = True
    if gui_use:
        img_color = np.copy(img_color)
    else:
        image_np = img_color.cpu().numpy()
        image_color = Image.fromarray((image_np * 255).astype(np.uint8))
        img_color = image_color.copy()
        img_color = np.array(img_color)  # 将 PIL 图像转换为 numpy 数组
    P = K@Rt
    outlier = np.max([img_color.shape[0], img_color.shape[1]])
    for obj in global_map:
        c3d = obj.ellipsoid_.Get_Center()
        z=Rt[2,:] @ np.append(c3d, 1)
        if z<0: continue
        proj = obj.ellipsoid_.project(P)
        center = proj.GetCenter()
        axes = proj.GetAxes()
        if np.any(axes > outlier) or np.any(axes < 0) or np.any(center < -(outlier / 2)):
            continue
        color = tuple(c*255/ 255 for c in obj.color)
        plot_net(axes, center, proj.GetAngle(), color, img_color)
    img_test = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)
    # if opt is not None:#2024-11-25 优化后结果
    #     cv2.imwrite("/home/lihy/3DGS/RTG-SLAM/output/object/"+str(frame_id)+"before.png", img_test)
    # else:
    #     cv2.imwrite("/home/lihy/3DGS/RTG-SLAM/output/object/" + str(frame_id) + ".png", img_test)

    if Record_img:
        ##2024-11-30 比较椭圆大小
        h,w,c = img_color.shape
        global  img_quadric
        img_quadric = np.zeros((h,w,c), dtype=np.uint8)*255
        for obj in global_map:
            proj = obj.ellipsoid_.project(P)
            center = proj.GetCenter()
            axes = proj.GetAxes()
            angle = np.degrees(proj.GetAngle())
            axes = (int(axes[0]), int(axes[1]))
            cv2.ellipse(img_quadric, (int(center[0]), int(center[1])), axes, angle, 0, 360, (255,0,0), 2)
    if gui_use:
        gui_img_color = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)
        return gui_img_color
def filter_object(obj, Rt):
    '''
    过滤掉不在视野内的物体
    :param center:
    :param Rt:
    :return:
    '''
    center = obj.ellipsoid_.center_
    z = Rt@ np.hstack((center, 1))
    if z[2]<0:
        return False
    return True
def plot_ellipse_2d(global_map, img_color, K, Rt,frame_id,opt=None, gui_use=False):
    '''
    绘制二维椭球在图像上
    :param global_map:
    :param img_color:
    :return:
    '''
    if gui_use:
        img_color = np.copy(img_color)
    else:
        image_np = img_color.cpu().numpy()
        image_color = Image.fromarray((image_np * 255).astype(np.uint8))
        img_color = image_color.copy()
        img_color = np.array(img_color)  # 将 PIL 图像转换为 numpy 数组
    P = K@Rt
    for obj in global_map:
        proj = obj.ellipsoid_.project(P)
        center = proj.GetCenter()
        axes = proj.GetAxes()
        if not filter_object(obj, Rt): continue
        angle = np.degrees(proj.GetAngle())  # 将角度转换为度数
        # angle = proj.GetAngle()
        axes = (int(axes[0]), int(axes[1]))
        cv2.ellipse(img_color, (int(center[0]), int(center[1])), axes, angle, 0, 360, (255,0,0), 2)
    # if opt is not None:#2024-11-25 before为优化前结果
    #     cv2.imwrite("/home/lihy/3DGS/RTG-SLAM/output/object/"+str(frame_id)+"before.png", img_color)
    # else:
    #     cv2.imwrite("/home/lihy/3DGS/RTG-SLAM/output/object/" + str(frame_id) + ".png", img_color)

    if Record_img:
        ##2024-11-30 比较椭圆大小
        h,w,c = img_color.shape
        global  img_quadric
        img_quadric = np.zeros((h,w,c), dtype=np.uint8)*255
        for obj in global_map:
            proj = obj.ellipsoid_.project(P)
            center = proj.GetCenter()
            axes = proj.GetAxes()
            angle = np.degrees(proj.GetAngle())
            axes = (int(axes[0]), int(axes[1]))
            cv2.ellipse(img_quadric, (int(center[0]), int(center[1])), axes, angle, 0, 360, (255,0,0), 2)
    if gui_use:
        gui_img_color = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)
        return gui_img_color
def plot_ellipse_and_bboxes(detections, img_color,frame_id):
    '''
    显示二维椭圆和检测框
    :param detections:
    :param img_color:
    :return:
    '''
    image_np = img_color.cpu().numpy()
    image_color = Image.fromarray((image_np * 255).astype(np.uint8))
    img_color = image_color.copy()
    img_color = np.array(img_color)
    #draw = ImageDraw.Draw(img_color)
    for obj in detections:
        bbox1 = obj["bbox"]
        # draw.rectangle(bbox1, outline='red', width=2)
        bbox2 = obj["ell"].ComputeBbox()
        center = obj["ell"].GetCenter()
        axes = obj["ell"].GetAxes()
        angle = np.degrees(obj["ell"].GetAngle())
        axes = (int(axes[0]), int(axes[1]))
        cv2.ellipse(img_color, (int(center[0]), int(center[1])), axes, angle, 0, 360, (255, 0, 0), 2)
        x1, y1, x2, y2 = map(int, bbox1)

        # Draw the bounding box using cv2.line
        cv2.line(img_color, (x1, y1), (x2, y1), (0, 255, 0), 2)  # Top edge
        cv2.line(img_color, (x2, y1), (x2, y2), (0, 255, 0), 2)  # Right edge
        cv2.line(img_color, (x2, y2), (x1, y2), (0, 255, 0), 2)  # Bottom edge
        cv2.line(img_color, (x1, y2), (x1, y1), (0, 255, 0), 2)  # Left edge

        #draw.rectangle([(bbox2[0],bbox2[1]),(bbox2[2],bbox2[3])], outline='blue', width=2)
    # cv2.imwrite("/home/lihy/3DGS/RTG-SLAM/output/object/" + str(frame_id) + "origin.png", img_color)

    # if PLT_SHOW:
    #     plt.imshow(img_color)
    #     plt.show()
    # plt.imshow(img_color)
    # plt.show()


## 2024-11-28 生成bbox
def generate_bbox_from_mask(mask):
    # Find the coordinates of the mask
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None  # No object found

    # Get the bounding box coordinates
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    # Create the bounding box
    bbox = [x_min, y_min, x_max, y_max]

    return bbox

#2024-11-25 获得物体的真值图 使用投影过来的方式
def get_gt_obj_copy(xyz, obj_color, img_color, K, R, t):
    """
    将3D点投影到图像上，并根据obj_color更改图像颜色。
    未映射的颜色将被设置为白色。

    参数：
    - xyz: N x 3 的张量，表示3D点
    - obj_color: N x 3 的张量，表示每个点的颜色 (RGB)
    - img_color: H x W x 3 的张量，表示原始图像颜色 (RGB)
    - K: 3 x 3 的张量，表示相机内参矩阵
    - R: 3 x 3 的数组或张量，表示旋转矩阵
    - t: 3 的数组或张量，表示平移向量

    返回：
    - new_image_tensor: H x W x 3 的张量，表示更改后的图像颜色
    """
    # 将 R 和 t 转换为 CUDA 上的张量
    R = torch.tensor(R, dtype=torch.float32, device='cuda')  # 3x3
    t = torch.tensor(t, dtype=torch.float32, device='cuda')  # 3

    # 确保 xyz 是 N x 3 的张量
    if xyz.dim() != 2 or xyz.size(1) != 3:
        raise ValueError("xyz 必须是 N x 3 的张量")

    # 计算相机坐标系下的点 (N x 3)
    xyz_3d = (R @ xyz.T).T + t  # N x 3

    # 投影到齐次坐标系 (N x 3)
    projected_hom = (K @ xyz_3d.T).T  # N x 3

    # 透视除法，避免除零
    z = projected_hom[:, 2].unsqueeze(1)  # N x 1
    z = z + 1e-7  # 防止除0
    xyz_img = projected_hom / z  # N x 3

    # 提取像素坐标 (N)
    x_pix = xyz_img[:, 0].long()
    y_pix = xyz_img[:, 1].long()

    # 图像尺寸
    height, width = img_color.shape[:2]

    # 限制像素坐标在图像范围内
    x_pix = torch.clamp(x_pix, 0, width - 1)
    y_pix = torch.clamp(y_pix, 0, height - 1)

    # 将坐标和颜色转移到 CPU
    x_pix_cpu = x_pix.cpu().numpy()
    y_pix_cpu = y_pix.cpu().numpy()
    obj_color_cpu = obj_color.cpu().numpy()
    img_color_cpu = img_color.cpu().numpy()

    color_map = {}
    for i in range(len(x_pix_cpu)):
        x = x_pix_cpu[i]
        y = y_pix_cpu[i]
        old_color = tuple(img_color_cpu[y, x].tolist())
        new_color = tuple((obj_color_cpu[i] * 255).astype('uint8').tolist())
        color_map[old_color] = new_color

    # 定义默认颜色为黑色
    default_color = (0, 0, 0)

    # 创建新图像并初始化为白色
    new_image_np = np.full_like(img_color_cpu, fill_value=0)

    # 替换颜色
    for old_color, new_color in color_map.items():
        mask = np.all(img_color_cpu == old_color, axis=-1)
        new_image_np[mask] = new_color


    # 测试更换后的颜色
    new_image_np = new_image_np.astype('uint8')  # 确保数据为 uint8 类型
    new_image_pil = Image.fromarray(new_image_np)
    if PLT_SHOW:
        plt.imshow(new_image_pil)
        plt.axis('off')  # 隐藏坐标轴
        plt.show()

    # 返回结果
    normalized_image_tensor = torch.tensor(new_image_np, dtype=torch.float32, device='cuda') / 255.0



    return normalized_image_tensor

def rgb_to_hsv(rgb):
    return cv2.cvtColor(np.uint8([[rgb]]), cv2.COLOR_RGB2HSV)[0][0]


#直接将除物体之外的颜色去掉
def get_gt_obj(xyz, obj_color, img_color, K, R, t):
    # 将 R 和 t 转换为 CUDA 上的张量
    R = torch.tensor(R, dtype=torch.float32, device='cuda')  # 3x3
    t = torch.tensor(t, dtype=torch.float32, device='cuda')  # 3

    # 确保 xyz 是 N x 3 的张量
    if xyz.dim() != 2 or xyz.size(1) != 3:
        raise ValueError("xyz 必须是 N x 3 的张量")

    # 计算相机坐标系下的点 (N x 3)
    xyz_3d = (R @ xyz.T).T + t  # N x 3

    # 投影到齐次坐标系 (N x 3)
    projected_hom = (K @ xyz_3d.T).T  # N x 3

    # 透视除法，避免除零
    z = projected_hom[:, 2].unsqueeze(1)  # N x 1
    z = z + 1e-7  # 防止除0
    xyz_img = projected_hom / z  # N x 3

    # 提取像素坐标 (N)
    x_pix = xyz_img[:, 0].long()
    y_pix = xyz_img[:, 1].long()
    # 限制像素坐标在图像范围内
    height, width = img_color.shape[:2]
    x_pix = torch.clamp(x_pix, 0, width - 1)
    y_pix = torch.clamp(y_pix, 0, height - 1)
    # 将坐标和颜色转移到 CPU
    x_pix_cpu = x_pix.cpu().numpy()
    y_pix_cpu = y_pix.cpu().numpy()





    # 图像尺寸
    height, width = img_color.shape[:2]
    obj_color_cpu = obj_color.cpu().numpy()
    img_color_cpu = img_color.cpu().numpy()
    color_map = {}
    for i in range(len(obj_color_cpu)):
        new_color = tuple((obj_color_cpu[i] * 255).astype('uint8').tolist())
        old_color = tuple(obj_color_cpu[i].tolist())
        color_map[old_color] = new_color
    # 创建新图像并初始化为黑色
    new_image_np = np.full_like(img_color_cpu, fill_value=0)
    # 替换颜色
    for old_color, new_color in color_map.items():
        mask = np.all(np.abs(img_color_cpu*255 - np.array(old_color)*255) < 10, axis=-1)  # 允许 10 的容差
        #直接给mask赋值
        #new_image_np[mask] = new_color
        ## 2024-11-28 将两个物体区分开
        # labeled_mask, num_features = label(mask)
        # if num_features >1:
        #     # Iterate over each connected component
        #     for i in range(1, num_features + 1):
        #         component_mask = (labeled_mask == i)
        #         if np.any(component_mask[y_pix_cpu, x_pix_cpu]) or np.sum(component_mask)<50:
        #             new_image_np[component_mask] = new_color
        #             break
        # else:
        #     new_image_np[mask] = new_color




        # ## 2024-11-28 生成bbox
        # bbox = generate_bbox_from_mask(mask)
        # if bbox:
        #     x_min, y_min, x_max, y_max = bbox
        #     new_image_np[y_min:y_max+1, x_min:x_max+1] = new_color

    # 测试更换后的颜色
    new_image_np = new_image_np.astype('uint8')  # 确保数据为 uint8 类型
    # new_image_pil = Image.fromarray(new_image_np)
    # plt.imshow(new_image_pil)
    # plt.axis('off')  # 隐藏坐标轴
    # plt.show()

    # 返回结果
    normalized_image_tensor = torch.tensor(new_image_np, dtype=torch.float32, device='cuda') / 255.0




    # ## 2024-12-1 测试将mask分别提取出来
    # img_color_cpu2 = new_image_np.copy()
    # hsv_image = cv2.cvtColor(img_color_cpu2, cv2.COLOR_BGR2HSV)
    #
    # for old_color, new_color in color_map.items():
    #     lower = rgb_to_hsv(new_color)
    #     upper = rgb_to_hsv(new_color)
    #     mask = cv2.inRange(hsv_image, lower, upper)
    #     num_labels, labels =cv2.connectedComponents(mask)
    #     masks = []
    #     for i in range(1, num_labels):
    #         single_mask = (labels == i).astype(np.uint8)*255
    #         masks.append(single_mask)
    #         for idx,m in enumerate(masks):
    #             cv2.imwrite(f"mask_{idx}.png", m)


    return normalized_image_tensor



## 2024-12-4 voom是使用下列距离进行优化
def gaussian_wasserstein_2d(ell1,ell2, constant_C):
    '''
    计算两个二维椭球之间的距离
    :param ell1:
    :param ell2:
    :param constant_C:
    :return:
    '''
    mu1, sigma1 = ell1.AsGaussian()
    mu2, sigma2 = ell2.AsGaussian()

    sigma11 = np.sqrt(sigma1)
    s121 = sigma11 @ sigma2 @ sigma11
    sigma121 = np.sqrt(s121)

    d = np.linalg.norm(mu1 - mu2)**2 + np.trace(sigma1 + sigma2 - 2 * sigma121)
    if d <0:
        d=0
    return  d


## 2024-12-1改成tensor形式
class Ellipse_tensor:
    def __init__(self, C):
        # 将输入矩阵转换为 torch tensor，确保其类型一致
        C = torch.tensor(C, dtype=torch.float32,device="cuda") if not isinstance(C, torch.Tensor) else C

        # 计算对称矩阵 C_sys
        C_sys = 0.5 * (C + C.T)
        if show_grad:
            print(f"C_sys  Ellipse requires_grad: {C_sys.requires_grad}")
        # 检查对称性
        if torch.abs(C_sys.T - C_sys).sum() > 1e-3:
            print("Warning: Matrix should be symmetric")

        # 标准化 C_sys
        C_sys /= -C_sys[2, 2]
        self.C_ = C_sys
        if show_grad:
            print(f"self.C_  Ellipse requires_grad: {self.C_.requires_grad}")
        self.center_ = -self.C_[:2, 2]
        if show_grad:
            print(f"self.center_  Ellipse requires_grad: {self.center_.requires_grad}")
        T_c = torch.eye(3, dtype=torch.float32,device="cuda")
        T_c[:2, 2] = -self.center_
        # Perform the transformation
        temp = T_c @ self.C_ @ T_c.T
        C_center = 0.5 * (temp + temp.T)
        if torch.isnan(C_center).any() or torch.isinf(C_center).any():
            print("C_center contains invalid values!")
            exit()
        det = torch.linalg.det(C_center[:2, :2])
        if det == 0 or torch.isclose(det, torch.tensor(0.0)):
            print("Matrix is singular or near-singular!")
            exit()

        # Eigenvalue decomposition
        eig_vals, eig_vecs = torch.linalg.eig(C_center[:2, :2])
        eig_vals = eig_vals.real  # The real part of eigenvalues

        eig_vecs = eig_vecs.real if eig_vecs.is_complex() else eig_vecs

        # Ensure correct sign for eigenvectors
        if torch.det(eig_vecs) < 0:
            eig_vecs = eig_vecs.clone()  # 创建副本避免 inplace
            eig_vecs[:, 1] *= -1
        if eig_vecs[0, 0] < 0:
            eig_vecs = eig_vecs.clone()
            eig_vecs *= -1

        # Set the axes and angles
        self.axes_ = torch.sqrt(torch.abs(eig_vals))
        self.angle_ = torch.atan2(eig_vecs[1, 0], eig_vecs[0, 0])
        self.has_changed_ = False
        if show_grad:
            print(f"self.axes_  Ellipse requires_grad: {self.axes_.requires_grad}")
            print(f"self.angle_  Ellipse requires_grad: {self.angle_.requires_grad}")



    def ComputeBbox(self):

        c = torch.cos(self.angle_)
        s = torch.sin(self.angle_)

        xmax = torch.sqrt(self.axes_[0] ** 2 * c ** 2 + self.axes_[1] ** 2 * s ** 2)
        ymax = torch.sqrt(self.axes_[0] ** 2 * s ** 2 + self.axes_[1] ** 2 * c ** 2)

        # 使用 stack 来创建 bbox，并确保它跟踪梯度
        bbox = torch.stack([self.center_[0] - xmax, self.center_[1] - ymax,
                            self.center_[0] + xmax, self.center_[1] + ymax])

        # 确保 bbox 跟踪梯度
        bbox.requires_grad_(True)

        return bbox

    def decompose(self):
        self.center_ = -self.Q_[:3, 3]
        T_c = np.eye(4)
        T_c[:3, 3] = -self.center_
        temp = T_c @ self.Q_ @ T_c.T
        Q_center = 0.5 * (temp + temp.T)
        eig_vals, eig_vecs = np.linalg.eigh(Q_center[:3, :3])
        if np.linalg.det(eig_vecs) < 0:
            eig_vecs[:, 2] *= -1
        self.axes_ = np.sqrt(np.abs(eig_vals))
        self.R_ = eig_vecs
        self.has_changed_ = False

    def AsGaussian(self):
        device = self.axes_.device  # Ensure the tensors are on the same device

        A_dual = torch.tensor([
            [self.axes_[0] ** 2, 0.0],
            [0.0, self.axes_[1] ** 2]
        ], device=device)

        R = torch.tensor([
            [torch.cos(self.angle_), -torch.sin(self.angle_)],
            [torch.sin(self.angle_), torch.cos(self.angle_)]
        ], device=device)

        cov = R @ A_dual @ R.T
        cov = torch.clamp(cov, min=0)  # Use torch.clamp to ensure the values are non-negative

        return self.center_, cov

    def ComputeBbox_rotated(self):
        c = torch.cos(self.angle_)
        s = torch.sin(self.angle_)

        # 计算半轴向量 (旋转前的 x 轴 和 y 轴 方向的偏移量)
        dx = torch.tensor([self.axes_[0] * c, self.axes_[0] * s],device=self.axes_.device)  # 确保dtype一致
        dy = torch.tensor([-self.axes_[1] * s, self.axes_[1] * c],device=self.axes_.device)
        # 计算四个顶点
        p1 = self.center_ - dx - dy  # 左下角
        p2 = self.center_ + dx - dy  # 右下角
        p3 = self.center_ + dx + dy  # 右上角
        p4 = self.center_ - dx + dy  # 左上角

        bbox = torch.stack([p1, p2, p3, p4])
        # 确保 bbox 跟踪梯度
        bbox.requires_grad_(True)

        return bbox  # 返回 4 个角点坐标 (2D)

## 2024-12-1改成tensor形式
class Ellipsoid_tensor(nn.Module):
    def __init__(self, axes, R, center, bbox):
        super(Ellipsoid_tensor, self).__init__()
        # 将 axes, R, center 转换为 PyTorch 张量，使用浮点数类型，通常是 float32
        self.axes_ = nn.Parameter(torch.tensor(axes, dtype=torch.float32, requires_grad=True,device="cuda"))
        self.R_ = nn.Parameter(torch.tensor(R, dtype=torch.float32,requires_grad=True,device="cuda"))

        # 将 R 转换为 PyTorch 张量，但不设置 requires_grad
        #self.R_ = torch.tensor(R, dtype=torch.float32, device="cuda")
        # 提取 yaw 角并设置为可优化参数
        #self.yaw_ = nn.Parameter(self.extract_yaw(self.R_), requires_grad=True)

        self.center_ =  nn.Parameter(torch.tensor(center, dtype=torch.float32,requires_grad=True,device="cuda"))
        self.bbox = nn.Parameter(torch.tensor(bbox, dtype=torch.float32,device="cuda"), requires_grad=False)
        if torch.any(torch.isnan(self.axes_)):
            print("Matrix contains invalid values in __init__（）, cleaning up.")
    def extract_yaw(self, R):
        # 从旋转矩阵中提取 yaw 角
        yaw = math.atan2(R[1, 0], R[0, 0])
        return torch.tensor(yaw, dtype=torch.float32, device="cuda")

    def build_rotation_matrix(self, yaw):
        # 使用 yaw 角重新构建旋转矩阵，pitch 和 roll 固定为 0
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)

        R = torch.eye(3, device="cuda")
        R[0, 0] = cos_yaw
        R[0, 1] = -sin_yaw
        R[1, 0] = sin_yaw
        R[1, 1] = cos_yaw

        return R

    def forward(self,P):
        if torch.any(torch.isnan(self.axes_)):
            print("Matrix contains invalid values in forward(), cleaning up.")
        # 使用优化后的 yaw 角重新构建旋转矩阵
        #R_optimized = self.build_rotation_matrix(self.yaw_)

        # 计算 Q_star 矩阵
        Q_star = torch.diag(torch.cat([self.axes_[:3] ** 2, torch.tensor([-1.0], device=self.axes_.device)]))  # [3,3] 轴长平方，并且最后一项为 -1

        # 创建平移矩阵 T_center
        T_center = torch.eye(4,device="cuda")
        T_center[:3, 3] = self.center_

        # 创建旋转矩阵 Rw_e
        Rw_e = torch.eye(4,device="cuda")
        Rw_e[:3, :3] = self.R_
        #Rw_e[:3, :3] = R_optimized

        # 将椭球自身平移和旋转结合到一起
        transf = T_center @ Rw_e

        # 得到椭球在世界坐标系下的 Q_star 矩阵
        Q_star = transf @ Q_star @ transf.T

        # 保证 Q_star 是对称的，并调整符号
        Q_ = 0.5 * (Q_star + Q_star.T)
        Q_ /= -Q_[3, 3]

        C = P @ Q_ @ P.T
        if torch.isnan(C).any() or torch.isinf(C).any():
            print("C contains invalid values!")
            exit()
        ell = Ellipse_tensor(C)


        bbox_cur = ell.ComputeBbox()
        #?? 2025-2-26 计算旋转后的bbox
        #bbox_cur = ell.ComputeBbox_rotated()
        if show_grad:
            print(f"self.axes_ requires_grad: {self.axes_.requires_grad}")
            print(f"C requires_grad: {C.requires_grad}")
            print(f"bbox_cur requires_grad: {bbox_cur.requires_grad}")
        return bbox_cur

    def project(self, P):
        C = P @ self.Q_ @ P.T
        ell = Ellipse_tensor(C)
        return  ell

## 2024-12-23 对每一个物体独立优化
def object_to_tensor(obj):
    axes = obj.ellipsoid_.axes_
    R = obj.ellipsoid_.R_
    center = obj.ellipsoid_.center_
    bbox = obj.bboxes_
    return Ellipsoid_tensor(axes, R, center, bbox)
def Object_Optimize_only(curr_dections, Map_global, K, Rt, frame_id=None):
    if SHOW_3D:
        show_results_3D(Map_global)
    debug=False
    if frame_id == 599:
        debug = False
        print("Debug")
    mp_tensor = []
    P = K @ Rt
    node_ids = []
    objec_iter = 20 #iter is 10
    for det in curr_dections:
        if det["obj"] is None: continue
        if not det["is_validate"]:continue
        obj = det["obj"]
        if len(obj.bboxes_) < 2: continue
        opt_ellipsoid = object_to_tensor(obj)
        params_to_optimize = [
            {'params': [opt_ellipsoid.axes_ ], 'lr': 0.01},  # 1：0.002 2：0.002#0.01
            {'params': [opt_ellipsoid.center_ ], 'lr': 0.001},  # 1： 0.001 2：0.002#0.002
            {'params': [opt_ellipsoid.R_ ], 'lr': 0.01}  # 1： 0.001 2：0.04#0.01
            #room：0.001 0.01 0.01
            #aithor1: 0.01 0.01 0.01
            #real 0.001 0.01 0.01
        ]
        optimizer = optim.Adam(params_to_optimize, eps=1e-15)

        for iter in range(objec_iter):
            optimizer.zero_grad()  # 清空梯度
            total_loss=0
            random_index = random.randint(0, len(obj.bboxes_)-1)
            if iter > objec_iter / 4:
                random_index = -1
            obv_bbox = obj.bboxes_[random_index]
            Rt = obj.Rts_[random_index]
            P = K @ Rt
            P = torch.tensor(P, dtype=torch.float32, device="cuda")
            output_bbox = opt_ellipsoid(P)
            output = bboxes_iou(obv_bbox, output_bbox)
            #wasser_dis = Calculate_distance_tensor(ell, det["ell"], 10)
            #output = bboxes_rotated_iou(obv_bbox, output_bbox)
            try:
                loss = 1.0 - output
                #loss = wasser_dis
                if loss == 1:
                    raise ValueError("Loss is 1")
            except ValueError as e:
                print(f"Error: {e}", "id= ", obj.category_id_)
                continue
            total_loss += loss
            total_loss.backward()
            optimizer.step()
            if SHOW_3D:
                if (iter + 1) % 5 == 0:
                    print(f"Epoch [{iter + 1}/10], Loss: {total_loss.item()}")
        axes = opt_ellipsoid.axes_.detach().cpu().numpy()
        R = opt_ellipsoid.R_.detach().cpu().numpy()
        center = opt_ellipsoid.center_.detach().cpu().numpy()
        obj_id = det["node_id"]
        ell = Ellipsoid(axes, R, center)
        det["obj"].ellipsoid_ = ell
        Map_global[obj_id].ellipsoid_ = ell

    if SHOW_3D:
        show_results_3D(Map_global, debug)
## 2024-12-1 使用voom内部的优化方法
def Object_Optimize(curr_dections, Map_global, K, Rt):
    show_results_3D(Map_global)
    mp_tensor = []
    P = K @ Rt
    node_ids = []
    for det in curr_dections:
        if det["obj"] is None: continue  # 没有对应物体，跳出
        obj = det["obj"].ellipsoid_
        node_ids.append(det["node_id"])
        center_ = obj.center_
        axes_ = obj.axes_
        R_ = obj.R_

        proj = obj.project(P)
        bbox_cur = proj.ComputeBbox()

        save_bbox = det["bbox"]
        bbox_tensor = torch.tensor(save_bbox, dtype=torch.float32, device="cuda")
        mp_tensor.append(Ellipsoid_tensor(axes_, R_, center_, bbox_tensor))
    if len(node_ids) ==0: return
    # 定义一个优化器，优化所有对象的参数
    params_to_optimize = [obj.axes_ for obj in mp_tensor] + [obj.center_ for obj in mp_tensor] + [obj.R_ for obj in
                                                                                          mp_tensor]
    # optimizer = optim.Adam(params_to_optimize, lr=0.001, eps=1e-15)
    params_to_optimize = [
        {'params': [obj.axes_ for obj in mp_tensor], 'lr': 0.002},  # 学习率为 0.001 的参数组
        {'params': [obj.center_ for obj in mp_tensor], 'lr': 0.001},  # 学习率为 0.0005 的参数组
        {'params': [obj.R_ for obj in mp_tensor], 'lr': 0.001}  # 学习率为 0.0001 的参数组
    ]
    optimizer = optim.Adam(params_to_optimize, eps=1e-15)

    P = torch.tensor(P, dtype=torch.float32, device="cuda")
    for epoch in range(50):  # 假设训练100轮
        optimizer.zero_grad()  # 清空梯度
        total_loss = 0
        for obj in mp_tensor:
            output_bbox = obj(P)
            output = bboxes_iou(obj.bbox, output_bbox)
            loss = 1.0 - output
            total_loss += loss
            if torch.isnan(loss) or torch.isinf(loss):
                print("Loss is unstable!")
                exit()
            if show_grad:
                print(f"Output_bbox requires_grad: {output_bbox.requires_grad}")
                print(f"Total loss requires_grad: {total_loss.requires_grad}")
        if show_grad:
            torch.autograd.set_detect_anomaly(True)
            with torch.autograd.detect_anomaly():
                total_loss.backward()  # 反向传播
        else:
            total_loss.backward()  # 反向传播
        # torch.nn.utils.clip_grad_norm_(list(params_to_optimize), max_norm=10.0)
        torch.nn.utils.clip_grad_norm_(
            [p for group in params_to_optimize for p in group['params']],
            max_norm=10.0
        )
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/100], Loss: {total_loss.item()}")
        # if epoch ==0 or (epoch+1) == 100:
        #     print(f"Epoch [{epoch + 1}/100], Loss: {total_loss.item()}")
    ## 保存结果
    i = 0
    for det in curr_dections:
        if det["obj"] is None: continue
        ell = mp_tensor[i]
        axes = ell.axes_.detach().cpu().numpy()
        R = ell.R_.detach().cpu().numpy()
        center = ell.center_.detach().cpu().numpy()

        ell = Ellipsoid(axes, R, center)
        det["obj"].ellipsoid_ = ell
        Map_global[node_ids[i]].ellipsoid_ = ell
        i += 1
    if DEBUG:
        print("After Object optimize Map_global size= ", len(Map_global))
    show_results_3D(Map_global)

##2024-12-04 使用物体构建共视关系
def Save_Keyframe_in_Object(map_global, curr_dections, frame, frame_map, frame_id, is_keyframe):
    # 是关键帧的话，更新信息
    if is_keyframe:
        for det in curr_dections:
            if det["obj"] is None: continue
            id = det["node_id"]
            map_global[id].save_keyframe.append(frame)
            map_global[id].save_keyframemap.append(frame_map)
            map_global[id].frame_ids.append(frame_id)
    else:
        # 将新建立物体的帧信息记录下来
        for obj in map_global:
            if len(obj.frame_ids) >0: continue
            obj.frame_ids.append(frame_id)
            obj.save_keyframe.append(frame)
            obj.save_keyframemap.append(frame_map)

def remove_outlier(map_global,K, Rt, debug=False):

    if debug:
        img = np.full((680, 1200, 3), 255, dtype=np.uint8)
        img = torch.tensor(img, dtype=torch.float32)
        plot_ellipse_2d_net(map_global, img, K, Rt, 0, opt=None, gui_use=False)
    for i in range(len(map_global) - 1, -1, -1):
        obj1 = map_global[i]
        for j in range(len(map_global) - 1, i, -1):
            obj2 = map_global[j]
            if debug:
                img = np.full((680, 1200, 3), 255, dtype=np.uint8)
                img = plot_for_debug(obj1, img, K, Rt)
                img = plot_for_debug(obj2, img, K, Rt)
                # cv2.imwrite(f"/home/lihy/3DGS/RTG-SLAM/output/debug/{i}_{j}.png", img)
            if obj1.category_id_ == obj2.category_id_:
                P = K @ Rt
                proj1 = obj1.ellipsoid_.project(P)
                proj2 = obj2.ellipsoid_.project(P)
                wasser_dis = Calculate_distance(proj1, proj2, 10)
                if wasser_dis < 0.1:
                    map_global.pop(j)  # 从后向前删除，不会影响未遍历的索引

    if debug:
        img = np.full((680, 1200, 3), 255, dtype=np.uint8)
        img = torch.tensor(img, dtype=torch.float32)
        plot_ellipse_2d_net(map_global, img, K, Rt, 0, opt=True, gui_use=False)

    return  map_global

def plot_for_debug(obj, img_color, K, Rt):
    '''
    将单个二维椭球绘制在图像上
    :param global_map:
    :param img_color:
    :return:
    '''
    P = K @ Rt
    proj = obj.ellipsoid_.project(P)
    center = proj.GetCenter()
    axes = proj.GetAxes()
    color = tuple(c * 255 / 255 for c in obj.color)
    plot_net(axes, center, proj.GetAngle(), color, img_color)
    return img_color
