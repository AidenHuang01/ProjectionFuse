#!/bin/bash
echo "./pcd2np.py : converting pointcloud from pcd to npy"
python pcd2np.py

echo "3D object detection by OpenPCDet and PointPillar"
python ./OpenPCDet/tools/demo.py --cfg_file ./OpenPCDet/tools/cfgs/kitti_models/pointpillar.yaml --ckpt ./OpenPCDet/tools/pointpillar_7728.pth --ext .npy
cd ./OpenPCDet/tools/
sh ./run.sh
cd ../../

echo "2D object detection by Detectron2"
python ./detectron2/workspace/detect.py

echo "ProjectFuse by combining 2D and 3D bouning boxes together"
python ./fusion.py
