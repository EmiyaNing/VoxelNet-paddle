import paddle
import paddle.nn as nn
import numpy as np

from config import get_config



class BNConv3D(nn.Layer):
    def __init__(self,
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding):
        super().__init__()
        self.conv = nn.Conv3D(in_channels, out_channels, kernel_size, stride, padding)
        self.norm = nn.BatchNorm3D(out_channels)
        self.active = nn.ReLU()
    
    def forward(self,inputs):
        result = self.norm(self.conv(inputs))
        result = self.active(result)
        return result



class BNConv2D(nn.Layer):
    def __init__(self, in_channels,
                out_channels,
                kernel_size,
                stride=1,
                padding=0):
        super().__init__()
        self.conv = nn.Conv2D(in_channels, out_channels, kernel_size, stride, padding)
        self.norm = nn.BatchNorm2D(out_channels)
        self.active = nn.ReLU()

    def forward(self, inputs):
        result = self.norm(self.conv(inputs))
        result = self.active(result)
        return result


class DevConv2D(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding):
        super().__init__()
        self.deconv = nn.Conv2DTranspose(in_channels, out_channels, kernel_size, stride, padding)
        self.norm   = nn.BatchNorm2D(out_channels)
        self.active = nn.ReLU()

    def forward(self, inputs):
        result = self.norm(self.deconv(inputs))
        result = self.active(result)
        return result

class RPNBlock(nn.Layer):
    def __init__(self,
                 layer_num,
                 hidden_dim=128,
                 out_dim=256,
                 dev_k = 3,
                 dev_s = 1,
                 dev_p = 0):
        super().__init__()
        self.conv   = BNConv2D(128, hidden_dim, 3, 2, 1)
        self.deconv = DevConv2D(hidden_dim, out_dim, dev_k, dev_s, dev_p)
        self.layers = nn.LayerList()
        for i in range(layer_num - 1):
            self.layers.append(BNConv2D(hidden_dim, hidden_dim, 3, 1, 1))


    def forward(self, inputs):
        result = self.conv(inputs)
        for layer in self.layers:
            result = layer(result)
        deconv = self.deconv(result)
        return result,deconv


class MiddleAndRPN(nn.Layer):
    def __init__(self, 
                 RPN_layer_num = [4, 6, 6],
                 RPN_hiddendim = [128, 128, 256],
                 RPN_outputdim = [256, 256, 256],
                 name = ''):
        super().__init__()
        self.Mconv1 = BNConv3D(128, 64, 3, (2, 1, 1), (1, 1, 1))
        self.Mconv2 = BNConv3D(64, 64, 3, (1, 1, 1), (0, 1, 1))
        self.Mconv3 = BNConv3D(64, 64, 3, (2, 1, 1), (1, 1, 1))

        self.block1 = RPNBlock(RPN_layer_num[0], RPN_hiddendim[0], RPN_outputdim[0], 3, 1, 1)
        self.block2 = RPNBlock(RPN_layer_num[1], RPN_hiddendim[1], RPN_outputdim[1], 2, 2, 0)
        self.block3 = RPNBlock(RPN_layer_num[2], RPN_hiddendim[2], RPN_outputdim[2], 4, 4, 0)

        self.pconv  = nn.Conv2D(768, 2, 1, 1, 0)
        self.rconv  = nn.Conv2D(768, 14, 1, 1, 0)

    def forward(self,inputs):
        middle = self.Mconv1(inputs)
        middle = self.Mconv2(middle)
        middle = self.Mconv3(middle)
        #middle = paddle.transpose(middle, perm=[0, 3, 4, 1, 2])
        middle = paddle.reshape(middle,shape=[-1, middle.shape[1]*middle.shape[2],
                                              middle.shape[3] , middle.shape[4]])

        result,deconv1 = self.block1(middle)
        result,deconv2 = self.block2(result)
        _,deconv3      = self.block3(result)

        final_conv     = paddle.concat([deconv3, deconv2, deconv1], axis=1)
        p_map          = self.pconv(final_conv)
        r_map          = self.rconv(final_conv)
        return p_map, r_map
