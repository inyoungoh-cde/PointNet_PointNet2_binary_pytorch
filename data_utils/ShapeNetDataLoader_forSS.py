# -*- coding:utf-8 -*-
import os
import json
import warnings
import numpy as np
from torch.utils.data import Dataset
warnings.filterwarnings('ignore')

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc

class SemanticDataset(Dataset):
    def __init__(self, root='./data/shapenetcore_partanno_segmentation_benchmark_v0_normal', npoints=2500, split='train', normal_channel=False):
        self.npoints = npoints
        self.root = root
        self.normal_channel = normal_channel
        
        if split == 'train':
            json_file = os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list.json')
        elif split == 'val':
            json_file = os.path.join(self.root, 'train_test_split', 'shuffled_val_file_list.json')
        elif split == 'test':
            json_file = os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list.json')
        else:
            print('Unknown split: %s. Exiting..' % (split))
            exit(-1)
        
        with open(json_file, 'r') as f:
            file_list = json.load(f)
        self.datapath = [os.path.join(self.root, f + '.txt') for f in file_list]
        
        self.cache = {}
        self.cache_size = 200000

    def __getitem__(self, index):
        if index in self.cache:
            point_set, seg = self.cache[index]
        else:
            fn = self.datapath[index]
            data = np.loadtxt(fn).astype(np.float32)
            if not self.normal_channel:
                point_set = data[:, 0:3]
            else:
                point_set = data[:, 0:5]
            seg = data[:, -1].astype(np.int32)  # label: 0 or 1
            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, seg)
        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
        
        if point_set.shape[0] < self.npoints:
            point_set_ = np.zeros((self.npoints, point_set.shape[1]), dtype=np.float32)
            seg_ = np.zeros((self.npoints), dtype=np.int32)
            for i in range(0, point_set.shape[0]):
                point_set_[i, :] = point_set[i, :]
                seg_[i] = seg[i]
            iter = 0
            for i in range(point_set.shape[0], self.npoints):
                point_set_[i, :] = point_set[iter, :]
                seg_[i] = seg[iter]
                iter = iter + 1
                if iter == point_set.shape[0]:
                    iter = 0
        else:
            point_set_ = point_set[:self.npoints, :]
            seg_ = seg[:self.npoints]
            
        return point_set_, seg_

    def __len__(self):
        return len(self.datapath)

if __name__ == '__main__':
    dataset = SemanticDataset(root='./data/shapenetcore_partanno_segmentation_benchmark_v0_normal', npoints=2500, split='train', normal_channel=True)
    print("Dataset length:", len(dataset))
    point_set, seg = dataset[0]
    print("Point set shape:", point_set.shape)
    print("Segmentation shape:", seg.shape)
