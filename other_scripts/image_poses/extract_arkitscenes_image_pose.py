import os
import tqdm
import json
import glob

import numpy as np


def main():
    root_dir = "/xxx/ARKitScenes/3dod_downloaded/raw/Validation"
    
    scene_path_list = glob.glob(os.path.join(root_dir, "*"))
    
    for scene_path in tqdm.tqdm(scene_path_list, total=len(scene_path_list)):
        pose_path = os.path.join(scene_path, 'lowres_wide.traj')
        
        if not os.path.exists(pose_path):
            continue
        
        new_lines = []
        with open(pose_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                line_list = line.strip().split(' ')
                # new_line_list = [line_list[0]] + line_list[4:7]
                new_line_list = line_list

                new_line = ' '.join(new_line_list) + '\n'
                new_lines.append(new_line)
        
        # save to .txt
        # timestamp, x, y, z
        with open(os.path.join(scene_path, 'pose_10fps.txt'), "w") as f:
            f.writelines(new_lines)


if __name__ == '__main__':
    main()
