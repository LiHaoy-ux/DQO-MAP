#!/usr/bin/env python3
import sys
import os.path
import orbslam2
import time
import cv2
from tqdm import tqdm
from scipy.spatial.transform import Rotation
import numpy as np

def main(vocab_path, settings_path, sequence_path, association_path):

    rgb_filenames, depth_filenames, timestamps = load_images(association_path)
    num_images = len(timestamps)

    slam = orbslam2.System(vocab_path, settings_path, orbslam2.Sensor.RGBD)
    slam.set_use_viewer(False)
    slam.initialize(False)

    times_track = [0 for _ in range(num_images)]
    print('-----')
    print('Start processing sequence ...')
    print('Images in the sequence: {0}'.format(num_images))

    for idx in tqdm(range(num_images)):
        rgb_image = cv2.imread(os.path.join(sequence_path, rgb_filenames[idx]), cv2.IMREAD_UNCHANGED)
        depth_image = cv2.imread(os.path.join(sequence_path, depth_filenames[idx]), cv2.IMREAD_UNCHANGED)
        tframe = timestamps[idx]

        if rgb_image is None:
            print("failed to load image at {0}".format(rgb_filenames[idx]))
            return 1
        if depth_image is None:
            print("failed to depth at {0}".format(depth_filenames[idx]))
            return 1
        t1 = time.time()
        slam.process_image_rgbd(rgb_image, depth_image, tframe)
        t2 = time.time()

        ttrack = t2 - t1
        times_track[idx] = ttrack

        t = 0
        if idx < num_images - 1:
            t = timestamps[idx + 1] - tframe
        elif idx > 0:
            t = tframe - timestamps[idx - 1]

        if ttrack < t:
            time.sleep(t - ttrack)

    save_trajectory(slam.get_trajectory_points(), 'trajectory.txt')

    slam.shutdown()

    times_track = sorted(times_track)
    total_time = sum(times_track)
    print('-----')
    print('median tracking time: {0}'.format(times_track[num_images // 2]))
    print('mean tracking time: {0}'.format(total_time / num_images))

    return 0


def load_images(path_to_association):
    rgb_filenames = []
    depth_filenames = []
    timestamps = []
    with open(path_to_association) as times_file:
        for line in times_file:
            if len(line) > 0 and not line.startswith('#'):
                t, rgb, _, depth = line.rstrip().split(' ')[0:4]
                rgb_filenames.append(rgb)
                depth_filenames.append(depth)
                timestamps.append(float(t))
    return rgb_filenames, depth_filenames, timestamps

def rotation_matrix_to_tum_format(rotation_matrix):
    rotation = Rotation.from_matrix(rotation_matrix)
    quaternion = rotation.as_quat()
    return quaternion

def convert_to_tum_format(poses, timestamps):
    tum_poses = []
    for i in range(poses.shape[0]):
        pose = poses[i]
        quaternion = rotation_matrix_to_tum_format(pose[:3, :3])
        tum_timestamp = timestamps[i]  # Scaling factor of 0.1 to convert timestamps to seconds
        tum_pose = f"{tum_timestamp:.6f} {' '.join(map(str, pose[:3, 3]))} {' '.join(map(str, quaternion))}" 
        tum_poses.append(tum_pose)
    return tum_poses

def write_tum_poses_to_file(file_path, tum_poses):
    with open(file_path, 'w') as f:
        for pose in tum_poses:
            f.write(pose + '\n')
def convert_and_write_tum_poses(c2w_variable, output_filename, timestamps):
    tum_poses = convert_to_tum_format(c2w_variable, timestamps)
    write_tum_poses_to_file(output_filename, tum_poses)

def save_trajectory(trajectory, filename):
    timestampes = []
    poses = []
    for i in range(len(trajectory)):
        stamp, r00, r01, r02, t0, r10, r11, r12, t1, r20, r21, r22, t2 = trajectory[i]
        timestampes.append(stamp)
        poses.append(np.array([[r00, r01, r02, t0],
                               [r10, r11, r12, t1],
                               [r20, r21, r22, t2],
                               [0, 0, 0, 1]]))
    poses = np.stack(poses, axis=0)
    convert_and_write_tum_poses(poses, filename, timestampes)

if __name__ == '__main__':
    if len(sys.argv) != 5:
        print('Usage: ./orbslam_rgbd_tum path_to_vocabulary path_to_settings path_to_sequence path_to_association')
    vocab_path = "ORBvoc.txt"
    setting_path = "tum1.yaml"
    sequence_path = "/data/TUM_RGBD/rgbd_dataset_freiburg1_desk"
    assocation_path = "assocations.txt"
    main(vocab_path, setting_path, sequence_path, assocation_path)
