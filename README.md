# Pytorch Implementation of PointNet and PointNet++ for binary problem
This repo is implementation for PointNet and PointNet++ for binary (0 or 1) in pytorch.

# Update
02/14/2025:

(1) Our code is heavily based on https://github.com/yanx27/Pointnet_Pointnet2_pytorch

(2) We modified two codes for part segmentation and semantic segmentation for binary problem using the ShapeNet dataset. Especially, the semantic segmentation code is designed for the S3DIS dataset, but we modified it to work on the ShapeNet dataset.

(3) It was confirmed to work on Windows 10, and the dataset used is a point cloud similar to the ShapeNet dataset. It is a custom dataset with 6 channels: x, y, z, feature1, feature2, and label (0 or 1).

# Requirements
- Windows 10 Home with NVIDIA GeForece RTX 3080 Ti (single GPU)
- CUDA 11.0.221
- CUDNN 8.9.2.26
- Python 3.7.1
- PyTorch 1.7.0

# Part Segmentation for binary (ShapeNet)

If the --normal flag is provided, the custom dataset's feature1 and feature2 are used; otherwise, the code operates with only the three channels: x, y, and z. When testing, you must manually create a folder named "predicted_results" in the directory specified by log_dir. In the future, we plan to modify the code to automatically create this folder.

- PointNet
  
  (1) Train: python train_partseg_bnry.py --normal --npoint 500 --batch_size 10 --epoch 50 --gpu 0 --model pointnet_part_seg_binary --log_dir pointnet_part_seg_binary

  (2) Test: python test_partseg_bnry.py --normal --num_point 500 --batch_size 160 --gpu 0 --log_dir pointnet_part_seg_binary

- PointNet++

  (1) Train: python train_partseg_bnry.py --normal --npoint 500 --batch_size 10 --epoch 50 --gpu 0 --model pointnet2_part_seg_msg_binary --log_dir pointnet2_part_seg_msg_binary

  (2) Test: python test_partseg_bnry.py --normal --num_point 500 --batch_size 160 --gpu 0 --log_dir pointnet2_part_seg_msg_binary

# Semantic Segmentation (ShapeNet)

All five input channels (x, y, z, feature1, feature2) are used.

- PointNet++

  (1) Train: python train_semseg_bnry.py --npoint 500 --batch_size 10 --epoch 50 --gpu 0 --model pointnet2_sem_seg_msg_binary --log_dir pointnet2_sem_seg_msg_binary

  (2) Test: python test_semseg_bnry.py --num_point 500 --batch_size 160 --gpu 0 --log_dir pointnet2_sem_seg_msg_binary
