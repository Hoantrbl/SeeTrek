import os
import tqdm
import glob
import plyfile
import numpy as np
import json


def main():
    root_dir = '/xxx/datasets/scannet/raw/scans'
    pc_means_path = 'pc_means_new.json'
    scene_dir_list = glob.glob(os.path.join(root_dir, "*"))

    pc_means_dict = json.load(open(pc_means_path, 'r'))
    for scene_dir in tqdm.tqdm(scene_dir_list, total=len(scene_dir_list)):
        # 24 fps
        scene_pose_dir = os.path.join(scene_dir, 'pose')
        
        scene_pose_path_list = glob.glob(os.path.join(scene_pose_dir, "*.txt"))
        scene_pose_path_list = sorted(scene_pose_path_list)
        
        # obtain the mean value of the point clouds in the scene
        scene_name = scene_dir.split('/')[-1]
        # points_mean = np.array(pc_means_dict[scene_name])
        points_mean = np.array([0, 0, 0])
        
        scene_pose_list = []
        for scene_pose_path in scene_pose_path_list:
            last_column = []
            with open(scene_pose_path, 'r') as scene_pose_file:
                for i, line in enumerate(scene_pose_file):
                    if i >= 3:
                        break
                    columns = line.strip().split()
                    if columns:
                        last_column.append(columns[-1])

            # minus the mean value of the point clouds to the last column
            last_column = np.array(list(map(float, last_column))) - points_mean
            last_column = list(map(str, last_column))

            scene_pose_list.append(last_column)

        # output_file_path = os.path.join(scene_dir, 'all_pose_normalized.txt')
        output_file_path = os.path.join(scene_dir, 'all_pose.txt')
        # write scene_pose_list to file
        with open(output_file_path, 'w') as output_file:
            for scene_pose in scene_pose_list:
                output_file.write(' '.join(scene_pose) + '\n')


if __name__ == "__main__":
    main()
