import os
import os.path

import cv2
import numpy as np
import paddle.io.Dataset as Dataset

from utils.utils import box3d_corner_to_center_batch, anchors_center_to_corner, corner_to_standup_box2d_batch
from utils.box_overlaps import bbox_overlaps

class KittiDataset(Dataset):
    def __init__(self, cfg, set='train', type='velodyne_train'):
        super().__init__()
        

    def cal_target(self, gt_box3d):
        pass

    def preprocess(self, lidar):
        pass

    def __getitem__(self, i):
        pass

    def __len__(self):
        pass

