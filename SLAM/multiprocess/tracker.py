import copy
import matplotlib.pyplot as plt
from SLAM.gaussian_pointcloud import *

import torch.multiprocessing as mp
from SLAM.render import Renderer
from collections import defaultdict
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

from SLAM.icp import IcpTracker
from threading import Thread
from utils.camera_utils import loadCam


Record_TXT=True
def convert_poses(trajs):
    poses = []
    stamps = []
    for traj in trajs:
        stamp, r00, r01, r02, t0, r10, r11, r12, t1, r20, r21, r22, t2 = traj
        pose_ = np.eye(4)
        pose_[:3, :3] = np.array([[r00, r01, r02], [r10, r11, r12], [r20, r21, r22]])
        pose_[:3, 3] = np.array([t0, t1, t2])
        poses.append(pose_)
        stamps.append(stamp)
    return poses, stamps
def record_poses_txt(path, trajs):
    poses=[]
    from scene.dataset_readers import c2w0
    for traj in trajs:
        stamp, r00, r01, r02, t0, r10, r11, r12, t1, r20, r21, r22, t2 = traj
        pose = np.eye(4)
        pose[:3, :3] = np.array([[r00, r01, r02], [r10, r11, r12], [r20, r21, r22]])
        pose[:3, 3] = np.array([t0, t1, t2])
        # pose = c2w0@pose

        pose_ = np.eye(3)
        pose_[:3, :3] = pose[:3, :3]
        T= pose[:3, 3]
        q=R.from_matrix(pose_[:3, :3]).as_quat()
        poses.append([stamp, T[0], T[1], T[2], q[0], q[1], q[2], q[3]])

    with open (path, 'w') as f:
        for pose in poses:
            f.write(' '.join([str(i) for i in pose])+'\n')

class Tracker(object):
    def __init__(self, args):
        self.use_gt_pose = args.use_gt_pose#fasle
        self.mode = args.mode
        self.K = None

        self.min_depth = args.min_depth#0.3
        self.max_depth = args.max_depth#5.0
        self.depth_filter = args.depth_filter#fasle
        self.verbose = args.verbose#false

        #创建Icp追踪
        self.icp_tracker = IcpTracker(args)

        self.status = defaultdict(bool)#创建一个bool值的默认字典
        self.pose_gt = []
        self.pose_es = []
        self.pose_txt=[]
        self.timestampes = []
        self.finish = mp.Condition()#用于多个线程之间的通信

        self.icp_success_count = 0

        self.use_orb_backend = args.use_orb_backend#true
        self.orb_vocab_path = args.orb_vocab_path#orb的字典
        self.orb_settings_path = args.orb_settings_path#configs/orb_config/tum1.yaml
        self.orb_backend = None
        self.orb_useicp = args.orb_useicp#true

        self.invalid_confidence_thresh = args.invalid_confidence_thresh#0.2

        if self.mode == "single process":
            self.initialize_orb()#初始化orb参数

    def get_new_poses_byid(self, frame_ids):
        if self.use_orb_backend and not self.use_gt_pose:
            new_poses = convert_poses(self.orb_backend.get_trajectory_points())
            frame_poses = [new_poses[frame_id] for frame_id in frame_ids]
        else:
            frame_poses = [self.pose_es[frame_id] for frame_id in frame_ids]
        return frame_poses

    def get_new_poses(self):
        if self.use_orb_backend and not self.use_gt_pose:
            new_poses, _ = convert_poses(self.orb_backend.get_trajectory_points())
        else:
            new_poses = None
        return new_poses

    def save_invalid_traing(self, path):
        if np.linalg.norm(self.pose_es[-1][:3, 3] - self.pose_gt[-1][:3, 3]) > 0.15:
            if self.track_mode == "icp":
                frame_id = len(self.pose_es)
                torch.save(
                    self.icp_tracker.vertex_pyramid_t1,
                    os.path.join(path, "vertex_pyramid_t1_{}.pt".format(frame_id)),
                )
                torch.save(
                    self.icp_tracker.vertex_pyramid_t0,
                    os.path.join(path, "vertex_pyramid_t0_{}.pt".format(frame_id)),
                )
                torch.save(
                    self.icp_tracker.normal_pyramid_t1,
                    os.path.join(path, "normal_pyramid_t1_{}.pt".format(frame_id)),
                )
                torch.save(
                    self.icp_tracker.normal_pyramid_t0,
                    os.path.join(path, "normal_pyramid_t0_{}.pt".format(frame_id)),
                )

    def map_preprocess(self, frame, frame_id):
        depth_map, color_map = (
            frame.original_depth.permute(1, 2, 0) * 255,#把深度图也当成rgb图进行处理
            frame.original_image.permute(1, 2, 0),#从[1,640,480]调整成[640,480,1]
        )  # [H, W, C], the image is scaled by 255 in function "PILtoTorch"
        if frame.semantics is not None:
            semantics = frame.semantics.permute(1, 2, 0)
            # semantics_id = frame.semantics_id.permute(1, 2, 0)
        if frame.object_img is not None:
            object_img = frame.object_img.permute(1, 2, 0)
        depth_map_orb = (
            frame.original_depth.permute(1, 2, 0).cpu().numpy()
            * 255
            * frame.depth_scale#!不太理解这里为什么又要把depth_scale乘上,相当于返回原始深度图并*255
        ).astype(np.uint16)
        intrinsic = frame.get_intrinsic
        # depth filter
        if self.depth_filter:
            depth_map_filter = bilateralFilter_torch(depth_map, 5, 2, 2)
        else:
            depth_map_filter = depth_map

        valid_range_mask = (depth_map_filter > self.min_depth) & (depth_map_filter < self.max_depth)
        depth_map_filter[~valid_range_mask] = 0.0#不在范围内的深度取0
        # update depth map
        frame.original_depth = depth_map_filter.permute(2, 0, 1) / 255.0#这里又把255去掉了（看来处理深度图要先*255）
        # compute geometry info
        vertex_map_c = compute_vertex_map(depth_map_filter, intrinsic)#获得相机坐标系下坐标
        normal_map_c = compute_normal_map(vertex_map_c)#获得每一个像素点的单位法向量
        confidence_map = compute_confidence_map(normal_map_c, intrinsic)#获得像素法向量和相机投影向量之间的可信度

        # confidence_threshold tum: 0.5, others: 0.2
        #在最后一个维度检查是否存在0，和置信度不达标，设置为invalid_confidence_mask
        invalid_confidence_mask = ((normal_map_c == 0).all(dim=-1)) | (
            confidence_map < self.invalid_confidence_thresh
        )[..., 0]
        #不满足条件的值全部设置为0
        depth_map_filter[invalid_confidence_mask] = 0
        normal_map_c[invalid_confidence_mask] = 0
        vertex_map_c[invalid_confidence_mask] = 0
        confidence_map[invalid_confidence_mask] = 0

        color_map_orb = (
            (frame.original_image * 255).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        )
        #得到图像金字塔每一层法向量
        self.update_curr_status(
            frame,
            frame_id,
            depth_map,
            depth_map_filter,
            vertex_map_c,
            normal_map_c,
            color_map,
            color_map_orb,
            depth_map_orb,
            intrinsic,
        )

        frame_map = {}
        frame_map["depth_map"] = depth_map_filter
        frame_map["color_map"] = color_map
        frame_map["normal_map_c"] = normal_map_c#法向量
        frame_map["vertex_map_c"] = vertex_map_c#相机坐标系下坐标
        frame_map["confidence_map"] = confidence_map#法线和相机向量置信度
        frame_map["invalid_confidence_mask"] = invalid_confidence_mask
        frame_map["time"] = frame_id

        if frame.semantics is not None:
            frame_map["semantics"] = semantics
            # frame_map["semantics_id"] = semantics_id
        else:
            frame_map["semantics"] = None
            # frame_map["semantics_id"] = None

        if frame.object_img is not None:
            frame_map["object_img"] = object_img
        if frame.instance_img is not None:
            frame_map["instance_img"] = frame.instance_img.permute(1, 2, 0)
        else:
            frame_map["instance_img"] = None
        return frame_map

    def update_curr_status(
        self,
        frame,
        frame_id,
        depth_t1,
        depth_t1_filter,
        vertex_t1,
        normal_t1,
        color_t1,
        color_orb,
        depth_orb,
        K,
    ):
        if self.K is None:
            self.K = K
        self.curr_frame = {
            "K": frame.get_intrinsic,
            "normal_map": normal_t1,
            "depth_map": depth_t1,
            "depth_map_filter": depth_t1_filter,
            "vertex_map": vertex_t1,
            "frame_id": frame_id,
            "pose_gt": frame.get_c2w.cpu().numpy(), # 1
            "color_map": color_t1,
            "timestamp": frame.timestamp, # 1
            "color_map_orb": color_orb, # 1
            "depth_map_orb": depth_orb, # 1
        }
        self.icp_tracker.update_curr_status(depth_t1_filter, self.K)
        
    def update_last_status_v2(
        self, frame, render_depth, frame_depth, render_normal, frame_normal
    ):
        intrinsic = frame.get_intrinsic
        normal_mask = (
            1 - F.cosine_similarity(render_normal, frame_normal, dim=-1)
        ) < self.icp_sample_normal_threshold
        depth_filling_mask = (
            (
                torch.abs(render_depth - frame_depth)
                > self.icp_sample_distance_threshold
            )[..., 0]
            | (render_depth == 0)[..., 0]
            | (normal_mask)
        ) & (frame_depth > 0)[..., 0]

        render_depth[depth_filling_mask] = frame_depth[depth_filling_mask]
        render_depth[(frame_depth == 0)[..., 0]] = 0
        
        self.last_model_vertex = compute_vertex_map(render_depth, intrinsic)
        self.last_model_normal = compute_normal_map(self.last_model_vertex)

    def update_last_status(
        self,
        frame,
        render_depth,
        frame_depth,
        render_normal,
        frame_normal,
    ):
        self.icp_tracker.update_last_status(
            frame, render_depth, frame_depth, render_normal, frame_normal
        )

    def refine_icp_pose(self, pose_t1_t0, tracking_success):
        if tracking_success and self.orb_useicp:
            print("success")
            #给一个当前的初值
            self.orb_backend.track_with_icp_pose(
                self.curr_frame["color_map_orb"],
                self.curr_frame["depth_map_orb"],
                pose_t1_t0.astype(np.float32),
                self.curr_frame["timestamp"],
            )
            time.sleep(0.005)
        else:
            #不使用初值，还是依靠orb-slam
            self.orb_backend.track_with_orb_feature(
                self.curr_frame["color_map_orb"],
                self.curr_frame["depth_map_orb"],
                self.curr_frame["timestamp"],
            )
            time.sleep(0.005)
        traj_history = self.orb_backend.get_trajectory_points()#得到位姿
        pose_es_t1, _ = convert_poses(traj_history[-2:])
        return pose_es_t1[-1]#返回最新一帧位姿

    def initialize_orb(self):
        if not self.use_gt_pose and self.use_orb_backend and self.orb_backend is None:
            import orbslam2
            print("init orb backend")
            self.orb_backend = orbslam2.System(
                self.orb_vocab_path, self.orb_settings_path, orbslam2.Sensor.RGBD
            )
            self.orb_backend.set_use_viewer(False)
            self.orb_backend.initialize(self.orb_useicp)#初始化orb（应该是直接调用的动态链接库）

    def initialize_tracker(self):
        if self.use_orb_backend:
            self.orb_backend.process_image_rgbd(
                self.curr_frame["color_map_orb"],
                self.curr_frame["depth_map_orb"],
                self.curr_frame["timestamp"],
            )#直接调用orb程序
        self.status["initialized"] = True

    def tracking(self, frame, frame_map):
        self.pose_gt.append(self.curr_frame["pose_gt"])
        self.timestampes.append(self.curr_frame["timestamp"])
        p2loss = 0
        tracking_success = True
        if self.use_gt_pose:
            pose_t1_w = self.pose_gt[-1]
        else:
            # initialize
            if not self.status["initialized"]:
                self.initialize_tracker()
                pose_t1_w = np.eye(4)
            else:
                pose_t1_t0, tracking_success = self.icp_tracker.predict_pose(self.curr_frame)#使用icp得到初始位姿
                if self.use_orb_backend:
                    pose_t1_w = self.refine_icp_pose(pose_t1_t0, tracking_success)#传入的是一个两帧相对位姿
                else:
                    pose_t1_w = self.pose_es[-1] @ pose_t1_t0#不使用orb，需要对齐到第0帧

        self.icp_tracker.move_last_status()#保存当前帧
        self.pose_es.append(pose_t1_w)#添加新的位姿
        if Record_TXT:
            pose = [self.curr_frame["timestamp"],pose_t1_w[0][0],pose_t1_w[0][1],pose_t1_w[0][2],pose_t1_w[0][3],pose_t1_w[1][0],pose_t1_w[1][1],pose_t1_w[1][2],pose_t1_w[1][3],pose_t1_w[2][0],pose_t1_w[2][1],pose_t1_w[2][2],pose_t1_w[2][3]]
            self.pose_txt.append(pose)
        frame.updatePose(pose_t1_w)#更新位姿
        frame_map["vertex_map_w"] = transform_map(
            frame_map["vertex_map_c"], frame.get_c2w
        )
        frame_map["normal_map_w"] = transform_map(
            frame_map["normal_map_c"], get_rot(frame.get_c2w)
        )

        return tracking_success

    def eval_total_ate(self, pose_es, pose_gt):
        ates = []
        for i in tqdm(range(1, len(pose_gt) + 1)):
            ates.append(self.eval_ate(pose_es, pose_gt, i))
        ates = np.array(ates)
        return ates

    def save_ate_fig(self, ates, save_path, save_name):
        plt.plot(range(len(ates)), ates)
        plt.ylim(0, max(ates) + 0.1)
        plt.title("ate:{}".format(ates[-1]))
        plt.savefig(os.path.join(save_path, "{}.png".format(save_name)))
    

    def save_keyframe_traj(self, save_file):
        if self.use_orb_backend:
            poses, stamps = convert_poses(self.orb_backend.get_keyframe_points())
            with open(save_file, "w") as f:
                for pose_id, pose_es_ in enumerate(poses):
                    t = pose_es_[:3, 3]
                    q = R.from_matrix(pose_es_[:3, :3])
                    f.write(str(stamps[pose_id]) + " ")
                    for i in t.tolist():
                        f.write(str(i) + " ")
                    for i in q.as_quat().tolist():
                        f.write(str(i) + " ")
                    f.write("\n")

    def save_traj_tum(self, save_file):
        poses, stamps = convert_poses(self.orb_backend.get_trajectory_points())
        with open(save_file, "w") as f:
            for pose_id, pose_es_ in enumerate(self.pose_es):
                t = pose_es_[:3, 3]
                q = R.from_matrix(pose_es_[:3, :3])
                f.write(str(stamps[pose_id]) + " ")
                for i in t.tolist():
                    f.write(str(i) + " ")
                for i in q.as_quat().tolist():
                    f.write(str(i) + " ")
                f.write("\n")

    def save_orb_traj_tum(self, save_file):
        if self.use_orb_backend:
            poses, stamps = convert_poses(self.orb_backend.get_trajectory_points())
            with open(save_file, "w") as f:
                for pose_id, pose_es_ in enumerate(poses):
                    t = pose_es_[:3, 3]
                    q = R.from_matrix(pose_es_[:3, :3])
                    f.write(str(stamps[pose_id]) + " ")
                    for i in t.tolist():
                        f.write(str(i) + " ")
                    for i in q.as_quat().tolist():
                        f.write(str(i) + " ")
                    f.write("\n")

    def save_traj(self, save_path):
        save_traj_path = os.path.join(save_path, "save_traj")
        pose_es_txt = None
        if not self.use_gt_pose and self.use_orb_backend:
            traj_history = self.orb_backend.get_trajectory_points()
            self.pose_es, _ = convert_poses(traj_history)

        pose_es = np.stack(self.pose_es, axis=0)
        pose_gt = np.stack(self.pose_gt, axis=0)

        ates_ba = self.eval_total_ate(pose_es, pose_gt)
        print("ate: ", ates_ba[-1])
        np.save(os.path.join(save_traj_path, "pose_gt.npy"), pose_gt)
        np.save(os.path.join(save_traj_path, "pose_es.npy"), pose_es)


        save_path = os.path.join(save_traj_path, "poses.txt")
        pose_es_txt = record_poses_txt(save_path,self.pose_txt)  # 记录位姿的txt给物体slam使用

        self.save_ate_fig(ates_ba, save_traj_path, "ate")

        plt.figure()
        plt.plot(pose_es[:, 0, 3], pose_es[:, 1, 3])
        plt.plot(pose_gt[:, 0, 3], pose_gt[:, 1, 3])
        plt.legend(["es", "gt"])
        plt.savefig(os.path.join(save_traj_path, "traj_xy.jpg"))
        
        if self.use_orb_backend:
            self.orb_backend.shutdown()
        
    def eval_ate(self, pose_es, pose_gt, frame_id=-1):
        pose_es = np.stack(pose_es, axis=0)[:frame_id, :3, 3]
        pose_gt = np.stack(pose_gt, axis=0)[:frame_id, :3, 3]
        ate = eval_ate(pose_gt, pose_es)
        return ate


class TrackingProcess(Tracker):
    def __init__(self, slam, args):
        args.icp_use_model_depth = False
        super().__init__(args)

        self.args = args
        # online scanner
        self.use_online_scanner = args.use_online_scanner
        self.scanner_finish = False

        # sync mode
        self.sync_tracker2mapper_method = slam.sync_tracker2mapper_method
        self.sync_tracker2mapper_frames = slam.sync_tracker2mapper_frames

        # tracker2mapper
        self._tracker2mapper_call = slam._tracker2mapper_call
        self._tracker2mapper_frame_queue = slam._tracker2mapper_frame_queue

        self.mapper_running = True

        # mapper2tracker
        self._mapper2tracker_call = slam._mapper2tracker_call
        self._mapper2tracker_map_queue = slam._mapper2tracker_map_queue

        self.dataset_cameras = slam.dataset.scene_info.train_cameras
        self.map_process = slam.map_process
        self._end = slam._end
        self.max_fps = args.tracker_max_fps
        self.frame_time = 1.0 / self.max_fps
        self.frame_id = 0
        self.last_mapper_frame_id = 0

        self.last_frame = None
        self.last_global_params = None

        self.track_renderer = Renderer(args)
        self.save_path = args.save_path

    def map_preprocess_mp(self, frame, frame_id):
        self.map_input = super().map_preprocess(frame, frame_id)

    def send_frame_to_mapper(self):
        print("tracker send frame {} to mapper".format(self.map_input["time"]))
        self._tracker2mapper_call.acquire()
        self._tracker2mapper_frame_queue.put(self.map_input)
        self.map_process._requests[0] = True
        self._tracker2mapper_call.notify()
        self._tracker2mapper_call.release()

    def finish_(self):
        if self.use_online_scanner:
            return self.scanner_finish
        else:
            return self.frame_id >= len(self.dataset_cameras)

    def getNextFrame(self):
        frame_info = self.dataset_cameras[self.frame_id]
        frame = loadCam(self.args, self.frame_id, frame_info, 1)
        print("get frame: {}".format(self.frame_id))
        self.frame_id += 1
        return frame


    def run(self):
        self.time = 0
        self.initialize_orb()

        while not self.finish_():
            frame = self.getNextFrame()
            if frame is None:
                break
            frame_id = frame.uid
            print("current tracker frame = %d" % self.time)
            # update current map
            move_to_gpu(frame)

            self.map_preprocess_mp(frame, frame_id)
            self.tracking(frame, self.map_input)
            self.map_input["frame"] = copy.deepcopy(frame)
            self.map_input["frame"] = frame

            self.map_input["poses_new"] = self.get_new_poses()
            # send message to mapper

            self.send_frame_to_mapper()

            wait_begin = time.time()
            if not self.finish_() and self.mapper_running:
                if self.sync_tracker2mapper_method == "strict":
                    if (frame_id + 1) % self.sync_tracker2mapper_frames == 0:
                        with self._mapper2tracker_call:
                            print("wait mapper to wakeup")
                            print(
                                "tracker buffer size: {}".format(
                                    self._tracker2mapper_frame_queue.qsize()
                                )
                            )
                            self._mapper2tracker_call.wait()
                elif self.sync_tracker2mapper_method == "loose":
                    if (
                        frame_id - self.last_mapper_frame_id
                    ) > self.sync_tracker2mapper_frames:
                        with self._mapper2tracker_call:
                            print("wait mapper to wakeup")
                            self._mapper2tracker_call.wait()
                else:
                    pass
            wait_end = time.time()

            self.unpack_map_to_tracker()
            self.update_last_mapper_render(frame)
            self.update_viewer(frame)

            move_to_cpu(frame)

            self.time += 1
        # send a invalid time stamp as end signal
        self.map_input["time"] = -1
        self.send_frame_to_mapper()
        self.save_traj(self.save_path)
        self._end[0] = 1
        with self.finish:
            print("tracker wating finish")
            self.finish.wait()
        print("track finish")

    def stop(self):
        with self.finish:
            self.finish.notify()

    def unpack_map_to_tracker(self):
        self._mapper2tracker_call.acquire()
        while not self._mapper2tracker_map_queue.empty():
            map_info = self._mapper2tracker_map_queue.get()
            self.last_frame = map_info["frame"]
            self.last_global_params = map_info["global_params"]
            self.last_mapper_frame_id = map_info["frame_id"]
            print("tracker unpack map {}".format(self.last_mapper_frame_id))
        self._mapper2tracker_call.notify()
        self._mapper2tracker_call.release()

    def update_last_mapper_render(self, frame):
        pose_t0_w = frame.get_c2w.cpu().numpy()
        if self.last_frame is not None:
            pose_w_t0 = np.linalg.inv(pose_t0_w)
            self.last_frame.update(pose_w_t0[:3, :3].transpose(), pose_w_t0[:3, 3])
            render_output = self.track_renderer.render(
                self.last_frame,
                self.last_global_params,
                None
            )
            self.update_last_status(
                self.last_frame,
                render_output["depth"].permute(1, 2, 0),
                self.map_input["depth_map"],
                render_output["normal"].permute(1, 2, 0),
                self.map_input["normal_map_w"],
            )
