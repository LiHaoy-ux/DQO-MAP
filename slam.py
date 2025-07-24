import os
from argparse import ArgumentParser
import torch.multiprocessing as mp
from utils.config_utils import read_config
parser = ArgumentParser(description="Training script parameters")
#parser.add_argument("--config", type=str, default="configs/replica/office0.yaml")
#parser.add_argument("--config", type=str, default="configs/aithor/aithor1.yaml")
#parser.add_argument("--config", type=str, default="configs/real/real.yaml")

#parser.add_argument("--config", type=str, default="configs/aithor/aithor2.yaml")
#parser.add_argument("--config", type=str, default="configs/tum/fr1_desk.yaml")
parser.add_argument("--config", type=str, default="configs/replica/room0.yaml")
#parser.add_argument("--config", type=str, default="configs/RO-MAP/room.yaml")
#parser.add_argument("--config", type=str, default="configs/RO-MAP/scene1.yaml")
#parser.add_argument("--config", type=str, default="configs/RO-MAP/scene2.yaml")
args = parser.parse_args()
config_path = args.config #'configs/tum/fr1_desk.yaml'
args = read_config(config_path)
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(device) for device in args.device_list)
import torch
import json
from utils.camera_utils import loadCam
from arguments import DatasetParams, MapParams, OptimizationParams
from scene import Dataset
from SLAM.multiprocess.mapper import Mapping
from SLAM.multiprocess.tracker import Tracker
from SLAM.utils import *
from SLAM.eval import eval_frame
from utils.general_utils import safe_state
from utils.monitor import Recorder


from gui.multiprocessing_utils import FakeQueue
from gui import  gui_utils, slam_gui
import gc
import time

DEBUG_OBJ=False #Debug objects

torch.set_printoptions(4, sci_mode=False)
def main():
    # set visible devices
    time_recorder = Recorder(args.device_list[0])
    optimization_params = OptimizationParams(parser)
    dataset_params = DatasetParams(parser, sentinel=True)
    map_params = MapParams(parser)#


    safe_state(args.quiet)


    optimization_params = optimization_params.extract(args)
    dataset_params = dataset_params.extract(args)
    map_params = map_params.extract(args)

    if args.use_gui:
        mp.set_start_method("spawn")#Start multiple processes

        q_main2vis = mp.Queue() if args.use_gui else FakeQueue()
        q_vis2main = mp.Queue() if args.use_gui else FakeQueue()
        params_gui = gui_utils.ParamsGUI(
            pipe=None,
            background=torch.tensor([0,0,0], dtype=torch.float32, device="cuda"),
            gaussians=Mapping(args),
            q_main2vis=q_main2vis,
            q_vis2main=q_vis2main,
            render_args=args,
        )
        # If using GUI, create a GUI process gui_decess and start executing it
        gui_process = mp.Process(target=slam_gui.run, args=(params_gui,))
        gui_process.start()  # 
        time.sleep(10)  # 
        #win = slam_gui.run(params_gui)

    # Initialize dataset
    #Read images and poses from the dataset
    dataset = Dataset(
        dataset_params,
        shuffle=False,
        resolution_scales=dataset_params.resolution_scales,
    )

    record_mem = args.record_mem #

    gaussian_map = Mapping(args, time_recorder)
 
    gaussian_map.create_workspace()
    gaussian_tracker = Tracker(args)
    # save config file
    prepare_cfg(args)
    # set time log
    tracker_time_sum = 0
    mapper_time_sum = 0

    if DEBUG_OBJ:
        target_frames = list(range(400, 700))  #Perform debug_obj
    # start SLAM
    for frame_id, frame_info in enumerate(dataset.scene_info.train_cameras):
        if DEBUG_OBJ:
            if frame_id not in target_frames: continue
        #2024-11-19 Obtain the object detection result for this frame
        if dataset_params.use_object:
            curr_dect = dataset.detect_results[frame_id]
        else:
            curr_dect = None

        curr_frame = loadCam(
            dataset_params, frame_id, frame_info, dataset_params.resolution_scales[0],curr_dect
        )

        print("\n========== curr frame is: %d ==========\n" % frame_id)
        move_to_gpu(curr_frame)#Put the depth map and color map into GPU
        start_time = time.time()
        # tracker process
        frame_map = gaussian_tracker.map_preprocess(curr_frame, frame_id)
        gaussian_tracker.tracking(curr_frame, frame_map)
        tracker_time = time.time()
        tracker_consume_time = tracker_time - start_time
        time_recorder.update_mean("tracking", tracker_consume_time, 1)

        tracker_time_sum += tracker_consume_time
        print(f"[LOG] tracker cost time: {tracker_time - start_time}")

        mapper_start_time = time.time()

        new_poses = gaussian_tracker.get_new_poses()#Get the latest pose from tracking
        gaussian_map.update_poses(new_poses)#Update the pose in the map
        # mapper process
        gaussian_map.mapping                              (curr_frame, frame_map, frame_id, optimization_params)

        gaussian_map.get_render_output(curr_frame)#得到渲染后的颜色 深度

        gaussian_tracker.update_last_status(
            curr_frame,
            gaussian_map.model_map["render_depth"],
            gaussian_map.frame_map["depth_map"],
            gaussian_map.model_map["render_normal"],
            gaussian_map.frame_map["normal_map_w"],
        )#更新mask和深度误差
        mapper_time = time.time()
        mapper_consume_time = mapper_time - mapper_start_time
        time_recorder.update_mean("mapping", mapper_consume_time, 1)#更新建图时间

        mapper_time_sum += mapper_consume_time
        print(f"[LOG] mapper cost time: {mapper_time - tracker_time}")#建图时间
        if record_mem:
            time_recorder.watch_gpu()
        if args.use_gui:
            gaussian_map.get_gaussian_for_gui(curr_frame, frame_id,q_main2vis)
        # win._update_thread()
        # report eval loss 一段时间保存和是输出评估误差
        if ((gaussian_map.time + 1) % gaussian_map.save_step == 0) or (
            gaussian_map.time == 0
        ):
            eval_frame(
                gaussian_map,
                curr_frame,
                os.path.join(gaussian_map.save_path, "eval_render"),
                min_depth=gaussian_map.min_depth,
                max_depth=gaussian_map.max_depth,
                save_picture=True,
                run_pcd=False
            )#评估误差，输出图到eval_render，输出误差到终端
            gaussian_map.save_model(save_data=True)#保存.ply点云数据
            gaussian_map.save_obj(save_data=True)#保存.obj模型数据

        gaussian_map.time += 1
        move_to_cpu(curr_frame)
        torch.cuda.empty_cache()
    print("\n========== main loop finish ==========\n")
    print(
        "[LOG] stable num: {:d}, unstable num: {:d}".format(
            gaussian_map.get_stable_num, gaussian_map.get_unstable_num
        )
    )
    print("[LOG] processed frame: ", gaussian_map.optimize_frames_ids)
    print("[LOG] keyframes: ", gaussian_map.keyframe_ids)
    print("[LOG] mean tracker process time: ", tracker_time_sum / (frame_id + 1))
    print("[LOG] mean mapper process time: ", mapper_time_sum / (frame_id + 1))
    
    new_poses = gaussian_tracker.get_new_poses()
    gaussian_map.update_poses(new_poses)
    gaussian_map.global_optimization(optimization_params, is_end=True)
    eval_frame(
        gaussian_map,
        gaussian_map.keyframe_list[-1],
        os.path.join(gaussian_map.save_path, "eval_render"),
        min_depth=gaussian_map.min_depth,
        max_depth=gaussian_map.max_depth,
        save_picture=True,
        run_pcd=False
    )
 
    gaussian_map.save_model(save_data=True)
    gaussian_map.save_obj(save_data=True)
    gaussian_map.record_iou(gaussian_map.keyframe_list[-1])
    gaussian_tracker.save_traj(args.save_path)
    time_recorder.cal_fps()
    time_recorder.save(args.save_path)
    gaussian_map.time += 1
    
    if args.pcd_densify:    
        densify_pcd = gaussian_map.stable_pointcloud.densify(1, 30, 5)
        o3d.io.write_point_cloud(
            os.path.join(args.save_path, "save_model", "pcd_densify.ply"), densify_pcd
        )

    if args.use_gui:
        for i in range(10):
            gaussian_map.get_gaussian_for_gui(curr_frame, frame_id,q_main2vis)
        gaussian_map.get_gaussian_for_gui(curr_frame, frame_id, q_main2vis, finish=True)
        gui_process.join()
        input("GUI Stopped and joined the main thread")
    input("Press Enter to exit...")



if __name__ == "__main__":
    main()
