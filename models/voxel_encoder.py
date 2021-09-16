"""This file implement the voxel encoder's some layer class"""
import sys
import paddle
import paddle.nn as nn
import numpy as np
sys.path.append('..')
from config import get_config

cfg = get_config()


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
        self.linear= nn.Linear(in_channels, self.units)
        self.norm  = nn.BatchNorm1D(self.units, momentum=0.01, epsilon=1e-3)
        self.relu  = nn.ReLU()

    def forward(self, inputs):
        '''
        Args:
            inputs: A tensor with shape [area_count, random_sample_num, 7]
        Returns:
            concated: A tensor with shape [area_count, random_sample_num, 2 * units]
        '''
        batch_sz, area_count, voxel_count, _ = inputs.shape
        x = self.linear(inputs)
        x_batch_list = []
        for i in range(batch_sz):
            single_feature = paddle.transpose(self.norm(paddle.transpose(x[i, :, :, :], perm=[0, 2, 1])), perm=[0, 2, 1])
            single_feature = paddle.unsqueeze(single_feature, axis=0)
            x_batch_list.append(single_feature)
        x           = paddle.concat(x_batch_list, axis=0)
        pointwise   = self.relu(x)

        aggregated  = paddle.max(pointwise, axis=2, keepdim=True)# B, K, 1, units
        repeated    = paddle.broadcast_to(aggregated, shape=[batch_sz, area_count, voxel_count, self.units])

        concated    = paddle.concat([pointwise, repeated], axis=3)# B, K, T, 2*units
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


    def voxel_index(self, features, coor):
        batch_sz, _, dim  = features.shape
        features          = features.numpy()
        coor              = coor.numpy()
        dense_list        = []
        for i in range(batch_sz):
            dense_feature = np.zeros([dim, 10, cfg.INPUT_HIGHT, cfg.INPUT_WIDTH])
            dense_feature[:, coor[i, :, 0], coor[i, :, 1], coor[i, :, 2]] = features[i, :, :].T
            dense_feature = dense_feature[np.newaxis, :, :, :, :]
            dense_list.append(dense_feature)
        dense_feature = np.concatenate(dense_list, axis=0)
        result        = paddle.to_tensor(dense_feature, dtype='float32')
        return result



    def forward(self, features, num_voxel, coor):
        '''
        Args:
            features: the input tensor with shape [batch_sz, area_count, num_voxel_size, 4]
            num_voxel: the input tensor with shape [batch_sz,area_count], which indicate each 
                       area actually contain how much voxel.
            coor:      the input tensor with shape [batch_sz, area_count, 3], whose each element
                       store as [d, h, w]
        Returns:
            voxelwise: A tensor with shape [batch_sz,areas_count, feature_channels]
        '''

        batch_sz, area_count, voxel_count, _= features.shape
        mask        = paddle.max(features, axis=3, keepdim=True) != 0
        mask        = paddle.cast(mask, dtype='float32')
        for layer in self.VFE_layerlist:
            features = layer(features)
            features = features * mask
        
        x = features * mask
        voxelwise = paddle.max(x, axis=2)
        voxelwise = self.voxel_index(voxelwise, coor)
        return voxelwise


