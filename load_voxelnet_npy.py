import paddle
import numpy as np
from models.model import VoxelNet
from config import get_config

cfg = get_config()
static_dict = np.load('voxelnet.npy', allow_pickle=True).item()



def print_model_named_params(model):
    for name, param in model.named_parameters():
        print(name, param.shape)

def print_model_named_buffers(model):
    for name, buff in model.named_buffers():
        print(name, buff.shape)




mapping = [
    ('VFE-1/kernel','vfe_layer.VFE_layerlist.0.linear.weight'),
    ('VFE-1/bias','vfe_layer.VFE_layerlist.0.linear.bias'),
    ('VFE-1/gamma','vfe_layer.VFE_layerlist.0.norm.weight'),
    ('VFE-1/beta','vfe_layer.VFE_layerlist.0.norm.bias'),
    ('VFE-1/moving_mean','vfe_layer.VFE_layerlist.0.norm._mean'),
    ('VFE-1/moving_variance','vfe_layer.VFE_layerlist.0.norm._variance'),
    ('VFE-2/kernel','vfe_layer.VFE_layerlist.1.linear.weight'),
    ('VFE-2/bias','vfe_layer.VFE_layerlist.1.linear.bias'),
    ('VFE-2/gamma','vfe_layer.VFE_layerlist.1.norm.weight'),
    ('VFE-2/beta','vfe_layer.VFE_layerlist.1.norm.bias'),
    ('VFE-2/moving_mean','vfe_layer.VFE_layerlist.1.norm._mean'),
    ('VFE-2/moving_variance','vfe_layer.VFE_layerlist.1.norm._variance'),
]

def generate_middle_mapping():
    global mapping
    tf_front_name     = 'MiddleAndRPN_/conv'
    paddle_front_name = 'middle_rpn.Mconv'
    tf_back_list      = ['kernel', 'bias', 'gamma', 'beta', 'moving_mean', 'moving_variance']
    paddle_back_list  = ['conv.weight','conv.bias', 'norm.weight', 'norm.bias', 'norm._mean', 'norm._variance']
    for i in range(3):
        for j in range(6):
            tf_name     = tf_front_name + str(i + 1) + '/'
            paddle_name = paddle_front_name + str(i + 1) + '.'
            tf_name     += tf_back_list[j]
            paddle_name += paddle_back_list[j]
            #mapping[tf_name] = paddle_name
            mapping.append((tf_name, paddle_name))

def generate_rpn_mapping():
    global mapping
    tf_front_name     = 'MiddleAndRPN_/'
    paddle_front_name = 'middle_rpn.block'
    tf_back_name      = ['kernel', 'bias', 'gamma', 'beta', 'moving_mean', 'moving_variance']
    paddle_back_name  = ['conv.weight', 'conv.bias', 'norm.weight', 'norm.bias', 'norm._mean', 'norm._variance']
    for i in range(3):
        if i == 0:
            for j in range(4):
                if j == 0:
                    for k in range(6):
                        paddle_front  = paddle_front_name + str(i + 1) + '.conv.'
                        paddle_name = paddle_front + paddle_back_name[k]
                        tf_name     = tf_front_name + 'conv' + str((j + 1) + 3) + '/' + tf_back_name[k]
                        #mapping[tf_name] = paddle_name
                        mapping.append((tf_name, paddle_name))
                else:
                    for k in range(6):
                        paddle_front  = paddle_front_name + str(i + 1) + '.'
                        paddle_name = paddle_front + 'layers.' + str(j-1) + '.' + paddle_back_name[k]
                        tf_name     = tf_front_name + 'conv' + str((j + 1) + 3) + '/' + tf_back_name[k]
                        #mapping[tf_name] = paddle_name
                        mapping.append((tf_name, paddle_name))
            for k in range(6):
                paddle_front  = paddle_front_name + str(i + 1) + '.'
                if k < 2:
                    paddle_name = paddle_front + 'deconv.de' + paddle_back_name[k]
                else:
                    paddle_name = paddle_front + 'deconv.' + paddle_back_name[k]
                tf_name = tf_front_name + 'deconv' + str(i+1) + '/' + tf_back_name[k]
                #mapping[tf_name] = paddle_name
                mapping.append((tf_name, paddle_name))
        else:
            tf_index = [7, 13]
            for j in range(6):
                if j == 0:
                    for k in range(6):
                        paddle_front  = paddle_front_name + str(i + 1) + '.conv.'
                        paddle_name = paddle_front + paddle_back_name[k]
                        tf_name     = tf_front_name + 'conv' + str((j + 1) + tf_index[i-1]) + '/' + tf_back_name[k]
                        #mapping[tf_name] = paddle_name
                        mapping.append((tf_name, paddle_name))
                else:
                    for k in range(6):
                        paddle_front  = paddle_front_name + str(i + 1) + '.'
                        paddle_name = paddle_front + 'layers.' + str(j-1) + '.' + paddle_back_name[k]
                        tf_name     = tf_front_name + 'conv' + str((j + 1) + tf_index[i-1]) + '/' + tf_back_name[k]
                        #mapping[tf_name] = paddle_name
                        mapping.append((tf_name, paddle_name))
            for k in range(6):
                paddle_front  = paddle_front_name + str(i + 1) + '.'
                if k < 2:
                    paddle_name = paddle_front + 'deconv.de' + paddle_back_name[k]
                else:
                    paddle_name = paddle_front + 'deconv.' + paddle_back_name[k]
                tf_name = tf_front_name + 'deconv' + str(i+1) + '/' + tf_back_name[k]
                #mapping[tf_name] = paddle_name
                mapping.append((tf_name, paddle_name))

def generate_pr_map():
    global mapping
    tf_front_name     = 'MiddleAndRPN_/conv'
    paddle_front_name = ['middle_rpn.pconv', 'middle_rpn.rconv']
    tf_back_list      = ['kernel', 'bias']
    paddle_back_list  = ['weight', 'bias']
    for i in range(2):
        tf_front = tf_front_name + str(i + 20) + '/'
        paddle_front  = paddle_front_name[i] + '.'
        for j in range(2):
            tf_name     = tf_front + tf_back_list[j]
            paddle_name = paddle_front + paddle_back_list[j]
            #mapping[tf_name] = paddle_name
            mapping.append((tf_name, paddle_name))

def convert(tf_static, paddle_model):
    def _set_value(tf_name, pd_name):
        tf_shape = tf_static[tf_name].shape
        pd_shape = tuple(pd_params[pd_name].shape)
        print(f'set {tf_name} {tf_shape} to {pd_name} {pd_shape}')
        value    = tf_static[tf_name]
        if len(value.shape) == 4:
            value = value.transpose((3, 2, 0, 1))
        if len(value.shape) == 5:
            value = value.transpose((4, 3, 0, 1, 2))
        pd_params[pd_name].set_value(value)

    global mapping
    pd_params = {}
    for name, param in paddle_model.named_parameters():
        pd_params[name] = param
    
    for name, param in paddle_model.named_buffers():
        pd_params[name] = param

    for tf_name, pd_name in mapping:
        _set_value(tf_name, pd_name)
    return paddle_model

if __name__ == '__main__':
    model = VoxelNet(cfg)
    print_model_named_params(model)
    '''for key in static_dict.keys():
        print(key)
        print(static_dict[key].shape)'''
    generate_middle_mapping()
    generate_rpn_mapping()
    generate_pr_map()
    model = convert(static_dict, model)
    paddle.save(model.state_dict(), './voxelnet.pdparams')