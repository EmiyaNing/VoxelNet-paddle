"""This file implement the voxel encoder's some layer class"""
import sys
import paddle
import paddle.nn as nn
import numpy as np
sys.path.append('..')
from config import get_config

cfg = get_config()

def get_paddings_indicator(actual_num, max_num, axis=0):
    '''
        Create boolean mask by actuall number of a padded tensor
        Args:
            actual_num: A tensor, whose each element is area's voxel number. shape=[area_count]
            max_num:    integar value, which indicate the max voxel number in a area.
        Returns:
            paddings_indicator: A boolean tensor with shape [area_count, max_num], which show
            how much voxel in this area.
    '''
    actual_num    = paddle.unsqueeze(actual_num, axis=axis+1)
    max_num_shape = [1] * len(actual_num.shape)
    max_num_shape[axis + 1] = -1
    max_num       = paddle.arange(end=max_num, dtype='int32')
    paddings_indicator = paddle.cast(actual_num, dtype='int32') > max_num
    return paddings_indicator


class VFELayer(nn.Layer):
    '''
        VFELayer is the basic layer of Voxel Feature Encoder class.
        This layer is used to extract the voxel-inter featrure from the random
        sampling result.
        In VoxelNet, this layer is used to process the input cloud point information.
    '''
    def __init__(self, in_channels, out_channels, name='vfe'):
        '''
        Args:
            in_channels:     The input tensor's channels.
            out_channels:    The output tensor's channels.
            name:            The layer's name.
        '''
        super().__init__()
        self.name  = name
        self.units = out_channels // 2
        self.linear= nn.Linear(in_channels, self.units, bias_attr=False)
        self.norm  = nn.BatchNorm1D(self.units, momentum=0.01, epsilon=1e-3)
        self.relu  = nn.ReLU()

    def forward(self, inputs):
        '''
        Args:
            inputs: A tensor with shape [batch_size, random_sample_num, 7]
        Returns:
            concated: A tensor with shape [batch_size, random_sample_num, 2 * units]
        '''
        batch_sz, voxel_count, _ = inputs.shape
        x = self.linear(inputs)
        x = paddle.transpose(self.norm(paddle.transpose(x, perm=[0, 2, 1])), perm=[0, 2, 1])
        pointwise   = self.relu(x)

        aggregated  = paddle.max(pointwise, axis=1, keepdim=True)# K, 1, units
        repeated    = paddle.broadcast_to(aggregated, shape=[batch_sz, voxel_count, self.units])

        concated    = paddle.concat([pointwise, repeated], axis=2)# K, T, 2*units
        return concated



class VoxelFeatureExtractor(nn.Layer):
    '''
    VoxelFeatureExtractor is a module which contain two VFELayer.
    '''
    def __init__(self,
                 num_filters=[32, 128],
                 name = 'VoxelFeatureExtractor'):
        '''
        Args:
            num_filters: a list, which indicate the filters number of each VFE layer.
            name:        a string value, indicate this module's name.
        '''
        super().__init__()
        self.name       = name
        num_filters     = [7] + num_filters
        self.vfe_number = len(num_filters)
        self.VFE_layerlist = nn.LayerList()
        for i in range(self.vfe_number-1):
            self.VFE_layerlist.append(VFELayer(num_filters[i], num_filters[i+1]))
        self.linear     = nn.Linear(num_filters[-1], num_filters[-1], bias_attr=False)
        self.norm       = nn.BatchNorm1D(num_filters[-1], momentum=0.01, epsilon=1e-3)
        self.relu       = nn.ReLU()

    def voxel_index(self, features, coor):
        dim           = features.shape[-1]
        features      = features.numpy()
        coor          = coor.numpy()
        dense_feature = np.zeros([dim, cfg.DATA.BATCH_SIZE, 10, cfg.INPUT_HIGHT, cfg.INPUT_WIDTH])
        dense_feature[:, coor[:, 0], coor[:, 1], coor[:, 2], coor[:, 3]] = features.T
        result        = paddle.to_tensor(dense_feature, dtype='float32')
        result        = paddle.transpose(result, perm=[1, 0, 2, 3, 4])
        return result



    def forward(self, features, num_voxel, coor):
        '''
        Args:
            features: the input tensor with shape [area_count, num_voxel_size, 4]
            num_voxel: the input tensor with shape [area_count], which indicate each 
                       area actually contain how much voxel.
            coor:      the input tensor with shape [area_count, 4], whose each element
                       store as [batch, d, h, w]
        Returns:
            voxelwise: A tensor with shape [areas_count, feature_channels]
        '''
        new_shape  = num_voxel.shape + [1, 1] #[area_count, 1, 1]
        num_voxel_t= paddle.reshape(paddle.cast(num_voxel, dtype=features.dtype), shape=new_shape)

        point_mean = paddle.sum(features[:, :, :3]) / num_voxel_t 
        features_relative = features[:, :, :3] - point_mean #x-vx, y-vy, z-vz 
        features = paddle.concat([features, features_relative], axis=-1)#[x,y,z,r,x-vx,y-vy,z-vz]

        voxel_count= features.shape[1]
        mask       = get_paddings_indicator(num_voxel, voxel_count, axis=0)
        mask       = paddle.unsqueeze(mask, axis=-1)# used to filter empty point.
        for layer in self.VFE_layerlist:
            features = layer(features)
            features = features * mask
        x = self.linear(features)
        x = paddle.transpose(self.norm(paddle.transpose(x, perm=[0, 2, 1])), perm=[0, 2, 1])
        x = self.relu(x)
        x = x * mask
        voxelwise = paddle.max(x, axis=1)
        voxelwise = self.voxel_index(voxelwise, coor)
        return voxelwise


