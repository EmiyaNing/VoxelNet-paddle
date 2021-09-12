import sys
import paddle
sys.path.append('..')
from models.model import VoxelNet
from config import get_config

cfg = get_config()

def print_model_named_params(model):
    print('----------------------------------')
    for name, param in model.named_parameters():
        print(name, param.shape)
    print('----------------------------------')


def print_model_named_buffers(model):
    print('----------------------------------')
    for name, param in model.named_buffers():
        print(name, param.shape)
    print('----------------------------------')

def test_model():
    data   = paddle.rand([32, 35, 4])
    num    = paddle.randint(0, 35, [32])
    coor   = paddle.randint(0, 4, [32, 4])
    model  = VoxelNet(cfg)
    p_map, r_map = model(data, num, coor)
    print('p_map.shape = ', p_map.shape)
    print('r_map.shape = ', r_map.shape)

def show_model():
    model = VoxelNet(cfg)
    print_model_named_params(model)

if __name__ == '__main__':
    test_model()
    #show_model()