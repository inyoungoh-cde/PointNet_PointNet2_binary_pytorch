# Pytorch Implementation of PointNet and PointNet++ for binary problem
This repo is implementation for PointNet and PointNet++ for binary (0 or 1) in pytorch.

# Update
03/05/2025:

(1) Our code is heavily based on https://github.com/yanx27/Pointnet_Pointnet2_pytorch

(2) We modified codes for semantic segmentation for binary problem using the ShapeNet dataset. Especially, the semantic segmentation code is designed for the S3DIS dataset, but we modified it to work on the ShapeNet dataset.

(3) It was confirmed to work on Windows 10, and the dataset used is a point cloud similar to the ShapeNet dataset. It is a custom dataset with 6 channels: x, y, z, feature1, feature2, and label (0 or 1).

# Requirements
- Windows 10 Home with NVIDIA GeForece RTX 3080 Ti (single GPU)
- CUDA 11.0.221
- CUDNN 8.9.2.26
- Python 3.7.1
- PyTorch 1.7.0

# Semantic Segmentation (ShapeNet)

All five input channels (x, y, z, feature1, feature2) are used. When the --normal flag is used, all five channels are used as inputs to the network. Otherwise, only the three-channel 3D position (x, y, z) is used as input.

- PointNet++

  (1) Train: python train_semseg_bnry_working.py --npoint 500 --batch_size 32 --epoch 50 --gpu 0 --model pointnet2_sem_seg_msg_binary --log_dir pointnet2_sem_seg_msg_binary_XYZ

  (2) Test: python test_semseg_bnry_working.py --num_point 500 --batch_size 160 --gpu 0 --log_dir pointnet2_sem_seg_msg_binary_XYZ
