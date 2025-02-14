# *_*coding:utf-8 *_*
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

class PartNormalDataset(Dataset):
    def __init__(self,root = './data/shapenetcore_partanno_segmentation_benchmark_v0_normal', npoints=2500, split='train', class_choice=None, normal_channel=False):
        self.npoints = npoints
        self.root = root
        self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')
        self.cat = {}
        self.normal_channel = normal_channel

        i = 0
        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                if i ==1:
                    break                    
                self.cat[ls[0]]=ls[1]  #self.cat[ls[0]] = ls[1]
                i = i+1
        self.cat = {k: v for k, v in self.cat.items()}
        self.classes_original = dict(zip(self.cat, range(len(self.cat))))
        
        if not class_choice is  None:
            self.cat = {k:v for k,v in self.cat.items() if k in class_choice}      

        self.meta = {}
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list.json'), 'r') as f:
            train_ids = set([str(d.split('/')[2]) for d in json.load(f)]) 
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_val_file_list.json'), 'r') as f:
            val_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list.json'), 'r') as f:
            test_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        for item in self.cat:
            self.meta[item] = []
            dir_point = os.path.join(self.root, self.cat[item])
            fns = sorted(os.listdir(dir_point))
            if split == 'trainval':
                fns = [fn for fn in fns if ((fn[0:-4] in train_ids) or (fn[0:-4] in val_ids))]
            elif split == 'train':
                fns = [fn for fn in fns if fn[0:-4] in train_ids]
            elif split == 'val':
                fns = [fn for fn in fns if fn[0:-4] in val_ids]
            elif split == 'test':
                fns = [fn for fn in fns if fn[0:-4] in test_ids]
            else:
                print('Unknown split: %s. Exiting..' % (split))
                exit(-1)

            for fn in fns:
                token = (os.path.splitext(os.path.basename(fn))[0])
                self.meta[item].append(os.path.join(dir_point, token + '.txt'))

        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                self.datapath.append((item, fn))

        self.classes = {}
        for i in self.cat.keys():
            self.classes[i] = self.classes_original[i]

        self.seg_classes = {'OnlyOne': [0, 1]}

        self.cache = {}
        self.cache_size = 200000

    def __getitem__(self, index):
        if index in self.cache:
            point_set, cls, seg = self.cache[index]
        else:
            fn = self.datapath[index]
            cat = self.datapath[index][0]
            cls = self.classes[cat]
            cls = np.array([cls]).astype(np.int32)
            data = np.loadtxt(fn[1]).astype(np.float32)
          
            if not self.normal_channel:
                point_set = data[:, 0:3]
            else:
                point_set = data[:, 0:5]
            seg = data[:, -1].astype(np.int32)
            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, cls, seg)
        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])

        if not self.normal_channel:
            point_set_= np.zeros((self.npoints,3), dtype=np.float32)
            seg_= np.zeros((self.npoints), dtype=np.int32)
            if len(point_set) < self.npoints:
                
                    for i in range(0,len(point_set)):
                        point_set_[i][0]= point_set[i][0]
                        point_set_[i][1]= point_set[i][1]
                        point_set_[i][2]= point_set[i][2]
                        seg_[i]= seg[i]
                
                    iter= 0
                    for i in range(len(point_set),self.npoints):
                        point_set_[i][0]= point_set[iter][0]
                        point_set_[i][1]= point_set[iter][1]
                        point_set_[i][2]= point_set[iter][2]
    
                        iter=iter+1
                        if iter == (len(point_set)-1):
                           iter = 0
            else :
                    
                    point_set_ = point_set
        else:

            npt = 5
            point_set_ = np.zeros((self.npoints, npt), dtype=np.float32)
            seg_ = np.zeros((self.npoints), dtype=np.int32)
            if len(point_set) < self.npoints:

                for i in range(0, len(point_set)):
                    for j in range(0, npt):
                        point_set_[i][j] = point_set[i][j]
                    seg_[i] = seg[i]

                iter = 0
                for i in range(len(point_set), self.npoints):

                    for j in range(0, npt):
                        point_set_[i][j] = point_set[iter][j]
                    seg_[i] = seg[iter]

                    iter = iter + 1
                    if iter == (len(point_set) - 1):
                        iter = 0

            else:

                point_set_ = point_set


        point_set = point_set_
        seg = seg_

        return point_set, cls, seg #point_set_ point_set

    def __len__(self):
        return len(self.datapath)



