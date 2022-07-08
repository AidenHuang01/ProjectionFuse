# import necessary libs
import torch
import pickle
import numpy as np
from dis import dis
import matplotlib.pyplot as plt
from PIL import Image
from projection_utils import *
from tqdm import tqdm
from pathlib import Path
import os

# Define constants
DATA_DIR = "/yucheng/experiment/"
PRED_DICT_DIR = "/yucheng/output/"

test_list = os.listdir(DATA_DIR)

for test_name in tqdm(test_list):
    images_dir = DATA_DIR + test_name + "/mtlb-pngFromBag"
    # Getting filename for image, pcd, box2d and box3d
    file_list_image=os.listdir(images_dir)
    file_list_no_ex = [x.split('.')[0] for x in file_list_image]
    print(f"filtering test {test_name}")
    for file in tqdm(file_list_no_ex):
        pred_dict_2D_path = PRED_DICT_DIR + test_name + "/2D_boxes/" + file + ".pickle"
        pred_dict_3D_path = PRED_DICT_DIR + test_name + "/3D_boxes/" + file + "_pred_dict.pickle"
        with open(pred_dict_2D_path, 'rb') as handle:
            pred_dict_2D = pickle.load(handle)
            boxes_2D = pred_dict_2D["boxes"]
            scores_2D = pred_dict_2D["scores"]
            classes_2D = pred_dict_2D["classes"]
        with open(pred_dict_3D_path, 'rb') as handle:
            pred_dict_3D = pickle.load(handle)
            boxes_3D = pred_dict_3D["pred_boxes"]
            scores_3D = pred_dict_3D["pred_scores"]
            labels_3D = pred_dict_3D["pred_labels"]
        boxes_3D_filtered = []
        scores_3D_filtered = []
        labels_3D_filtered = []
        boxes_3D_to_img = lidar2CameraOurs(boxes_3D[:, [1, 2, 0]])
        boxes_3D_to_img = boxes_3D_to_img[:,[1,0]]
        result = detect_inlier(boxes_2D, boxes_3D_to_img)
        pred_dict_3D_filtered = {}
        for i in range(len(result)):
            if result[i]:
                if scores_3D[i] > 0.2:
                    boxes_3D_filtered.append(boxes_3D[i])
                    scores_3D_filtered.append(scores_3D[i])
                    labels_3D_filtered.append(labels_3D[i])
            elif scores_3D[i] > 0.5:
                    boxes_3D_filtered.append(boxes_3D[i])
                    scores_3D_filtered.append(scores_3D[i])
                    labels_3D_filtered.append(labels_3D[i])
        pred_dict_3D_filtered["pred_boxes"] = boxes_3D_filtered
        pred_dict_3D_filtered["pred_scores"] = scores_3D_filtered
        pred_dict_3D_filtered["pred_labels"] = labels_3D_filtered
        output_file_name = PRED_DICT_DIR + test_name + "/3D_boxes_filtered/" + file + ".pickle"
        output_file_path = Path(output_file_name)
        output_file_path.parent.mkdir(exist_ok=True, parents=True)
        with open(output_file_name, 'wb') as handle:
            pickle.dump(pred_dict_3D_filtered, handle)