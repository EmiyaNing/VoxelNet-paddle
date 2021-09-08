import sys
import paddle
sys.path.append('..')
from models.rpn import MiddleAndRPN

def test_MiddleAndRPN():
    data = paddle.rand([4, 128, 10, 200, 240])
    model= MiddleAndRPN()
    p_map, r_map = model(data)
    print('p_map shape = ', p_map.shape)
    print('r_map shape = ', r_map.shape)

if __name__ == '__main__':
    test_MiddleAndRPN()