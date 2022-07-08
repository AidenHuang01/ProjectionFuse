import numpy as np 
import open3d as o3d
import os
from tqdm import tqdm
from pathlib import Path

DATA_PATH = "/yucheng/experiment/"
OUTPUT_PATH = "/yucheng/output/"

test_list = os.listdir(DATA_PATH)
print(f"start converting pcd to npy for data under {DATA_PATH}")
for test_name in tqdm(test_list):
    points_path = DATA_PATH + test_name + "/mtlb-pcdFromBag/"
    points_list = os.listdir(points_path)
    print(f"currently converting {test_name}")
    for file in tqdm(points_list):
        if file[-4:] != '.pcd':
            continue
        pcd_path = points_path + file
        pcd = o3d.io.read_point_cloud(pcd_path)
        out_arr = np.asarray(pcd.points)  
        num_points = out_arr.shape[0]
        intensity = np.zeros((num_points, 1))
        out_arr = np.hstack((out_arr, intensity))
        out_arr[:,0] = -out_arr[:,0]
        output_file = file[:-4] + '.npy'
        output_file_name = "/yucheng/output/" + test_name + "/points/" + output_file
        output_file_path = Path(output_file_name)
        output_file_path.parent.mkdir(exist_ok=True, parents=True)
        np.save(output_file_name, out_arr) 
        # print('successfully converted ', output_file)