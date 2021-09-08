import sys
import paddle
sys.path.append('..')
from models.voxel_encoder import *

def test_VFE_layer():
    vfe  = VFELayer(in_channels = 7, out_channels = 64)
    data = paddle.rand([4, 32, 7])
    result = vfe(data)
    print(result)

def test_VoxelFeatureExtractor():
    model = VoxelFeatureExtractor([32, 128])
    data  = paddle.rand([32, 35, 4])
    num_voxel = paddle.randint(0, 35, [32])
    coor  = paddle.randint(0, 35, [32, 4])
    result= model(data, num_voxel, coor)
    print(result)
    print(result.shape)

if __name__ == '__main__':
    #test_VFE_layer()
    test_VoxelFeatureExtractor()