import paddle
import numpy as np
from models.model import VoxelNet
from config import get_config

cfg = get_config()

def print_model_named_params(model):
    for name, param in model.named_parameters():
        print(name, param.shape)

def print_model_named_buffers(model):
    for name, buff in model.named_buffers():
        print(name, buff.shape)


if __name__ == '__main__':
    model = VoxelNet(cfg)
    print_model_named_params(model)