import os
import tqdm
import json
import glob

import numpy as np


def main():
    root_dir = "/xxx/datasets/scannet++/data/data"

    scene_name_list = []
    with open(os.path.join(root_dir, '../splits/nvs_sem_train.txt'), 'r') as f:
        train_list = f.readlines()
        for scene_name in train_list:
            scene_name_list.append(scene_name.strip())
    
    with open(os.path.join(root_dir, '../splits/nvs_sem_val.txt'), 'r') as f:
        val_list = f.readlines()
        for scene_name in val_list:
            scene_name_list.append(scene_name.strip())

    for scene_name in tqdm.tqdm(scene_name_list, total=len(scene_name_list)):
        # pose_path = os.path.join(root_dir, '0a5c013435', 'iphone', 'pose_intrinsic_imu.json')
        pose_path = os.path.join(root_dir, scene_name, 'iphone', 'pose_intrinsic_imu.json')
        
        with open(pose_path, "r") as f:
            pose = json.load(f)
        
        pose_list = []
        for frame in pose:
            position = [str(pose[frame]['aligned_pose'][0][3]),
                        str(pose[frame]['aligned_pose'][1][3]),
                        str(pose[frame]['aligned_pose'][2][3])
                        ]
            pose_list.append(position)
        
        # write scene_pose_list to file
        output_file_path = os.path.join(root_dir, scene_name, 'iphone', 'pose.txt')
        with open(output_file_path, 'w') as output_file:
            for frame_pose in pose_list:
                output_file.write(' '.join(frame_pose) + '\n')
        

if __name__ == '__main__':
    main()
