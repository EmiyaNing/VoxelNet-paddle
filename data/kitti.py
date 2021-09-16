import os
import glob
import math
import os.path
import sys
sys.path.append('..')


import cv2
import numpy as np

from paddle.io import Dataset
from data.preprocess import process_pointcloud
from data.data_aug   import aug_data
from config import get_config

cfg = get_config()

class KittiDataset(Dataset):
    def __init__(self, data_dir, shuffle=False, aug=False, is_testset=False):
        super().__init__()
        self.data_dir= data_dir
        self.f_rgb   = glob.glob(os.path.join(data_dir, 'image_2', '*.png'))
        self.f_rgb.sort()
        self.f_lidar = glob.glob(os.path.join(data_dir, 'velodyne', '*.bin'))
        self.f_lidar.sort()
        if is_testset:
            self.f_label = glob.glob(os.path.join(data_dir, 'label_2', '*.txt'))
        
        self.data_tag= [name.split('/')[-1].split('.')[-2] for name in self.f_rgb]

        assert len(self.data_tag) != 0, 'dataset folder is not correct'
        assert len(self.data_tag) == len(self.f_rgb) == len(self.f_lidar), 'dataset folder is not correct'

        self.nums = len(self.f_rgb)
        self.indices = list(range(self.nums))
        self.num_batches = int(math.floor(self.nums / float(cfg.DATA.BATCH_SIZE)))
        if shuffle:
            np.random.shuffle(self.indices)
        self.aug     = aug
        self.is_testset = is_testset
        
        

    def __getitem__(self, load_index):
        if self.aug:
            ret = aug_data(self.data_tag[load_index], self.data_dir)
        else:
            rgb = cv2.resize(cv2.imread(self.f_rgb[load_index]), (cfg.INPUT_HIGHT, cfg.INPUT_WIDTH))
            raw_lidar = np.fromfile(self.f_lidar[load_index], dtype=np.float32).reshape((-1, 4))
            if not self.is_testset:
                label_list = np.array([line for line in open(self.f_label[load_index], 'r').readlines()])
                label      = []
                for line in label_list:
                    line_list = line.split(' ')
                    if line_list[0] == 'Car':
                        temp_label = []
                        for i in range(len(line_list) - 1):
                            temp_label.append(float(line_list[i+1]))
                        label.append(temp_label)
                label      = np.array(label)               
            tag   = self.data_tag[load_index]
            voxel = process_pointcloud(raw_lidar)
            feature = voxel['feature_buffer']
            num_list= voxel['number_buffer']
            coordinate = voxel['coordinate_buffer']
        if not self.is_testset:
            return tag, label, feature, num_list, coordinate
        else:
            return tag, feature, num_list, coordinate


    def __len__(self):
        return len(self.f_rgb)




