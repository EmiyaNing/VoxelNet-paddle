import sys
import paddle
import paddle.nn as nn
sys.path.append('..')
#from utils.utils import *
from models.voxel_encoder import VoxelFeatureExtractor
from models.rpn import MiddleAndRPN
from config import get_config

cfg = get_config()

class VoxelNet(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.classes = config.MODEL.DETECT_OBJ
        rpn_hid = config.MODEL.RPN_HIDDEN_DIM
        rpn_layer = config.MODEL.RPN_LAYER_NUM
        rpn_output= config.MODEL.RPN_OUTPUT_DIM
        vfe_filter= config.MODEL.VFE_FILTER_NUM
        self.vfe_layer = VoxelFeatureExtractor(vfe_filter)
        self.middle_rpn= MiddleAndRPN(rpn_layer, rpn_hid, rpn_output)


    def forward(self, inputs, num_voxel, coor):
        vfe_feature = self.vfe_layer(inputs, num_voxel, coor)
        print(vfe_feature.shape)
        p_map,r_map = self.middle_rpn(vfe_feature)
        return p_map, r_map
