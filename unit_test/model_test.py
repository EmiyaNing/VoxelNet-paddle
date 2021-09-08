import sys
import paddle
sys.path.append('..')
from models.model import VoxelNet
from config import get_config

cfg = get_config()

def test_model():
    data   = paddle.rand([32, 35, 4])
    num    = paddle.randint(0, 35, [32])
    coor   = paddle.randint(0, 4, [32, 4])
    model  = VoxelNet(cfg)
    p_map, r_map = model(data, num, coor)
    print('p_map.shape = ', p_map.shape)
    print('r_map.shape = ', r_map.shape)

if __name__ == '__main__':
    test_model()