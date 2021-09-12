import os
import glob
import math
import os.path
import sys
sys.path.append('..')


import cv2
import numpy as np
import paddle.io.Dataset as Dataset

from data.preprocess import process_pointcloud
from data.data_aug   import aug_data
from config import get_config

cfg = get_config()

class KittiDataset(Dataset):
    def __init__(self, data_dir, shuffle=False, aug=False, is_testset=False):
        super().__init__()
        self.data_dir= data_dir
        self.f_rgb   = glob.glob(os.path.join(data_dir, 'image_2', '*.jpg')).sort() 
        self.f_lidar = glob.glob(os.path.join(data_dir, 'velodyne', '*.bin')).sort()
        self.f_label = glob.glob(os.path.join(data_dir, 'label_2', '*.txt')).sort()
        self.data_tag= [name.split('/')[-1].split('.')[-2] for name in f_rgb]

        assert len(data_tag) != 0, 'dataset folder is not correct'
        assert len(data_tag) == len(f_rgb) == len(f_lidar), 'dataset folder is not correct'

        self.nums = len(f_rgb)
        self.indices = list(range(nums))
        self.num_batches = int(math.floor(nums / float(cfg.DATA.BATCH_SIZE)))
        if shuffle:
            np.random.shuffle(indices)
        self.aug     = aug
        self.is_testset = is_testset
        
        

    def __getitem__(self, load_index):
        if self.aug:
            ret = aug_data(self.data_tag[load_index], self.data_dir)
        else:
            rgb = cv2.resize(cv2.imread(self.f_rgb[load_index]), (cfg.INPUT_HIGHT, cfg.INPUT_WIDTH))
            raw_lidar = np.fromfile(self.f_lidar[load_index], dtype=np.float32).reshape((-1, 4))
            if not self.is_testset:
                labels = [line for line in open(self.f_label[load_index], 'r').readlines()]
            else:
                labels = ['']
            tag   = self.data_tag[load_index]
            voxel = process_pointcloud(raw_lidar)
            feature = voxel['feature_buffer']
            num_list= voxel['number_buffer']
            coordinate = voxel['coordinate_buffer']
            coordinate = np.pad(coordinate, ((0,0), (1,0)), mode='constant', constant_values=0)

        return tag, labels, feature, num_list, coordinate, rgb, raw_lidar


    def __len__(self):
        return len(self.f_rgb)




