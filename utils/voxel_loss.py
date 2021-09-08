import sys
import paddle
import paddle.nn as nn
sys.path.append('..')
from config import get_config

cfg = get_config()
small_addon_for_BCE = 1e-6


def smooth_l1(deltas, targets, sigma=3.0):
    sigma2 = sigma * sigma
    diffs  = deltas - targets
    temp_re= paddle.abs(diffs) < (1.0 / sigma2)
    smooth_l1_signs   = paddle.cast(temp_re, dtype='float32')
    smooth_l1_option1 = paddle.multiply(diffs, diffs) * 0.5 * sigma2
    smooth_l1_option2 = paddle.abs(diffs) - 0.5 / sigma2
    smooth_l1         = paddle.multiply(smooth_l1_option1, smooth_l1_signs) + \
                        paddle.multiply(smooth_l1_option2, 1 - smooth_l1_signs)
    return smooth_l1

class VoxelLoss(nn.Layer):
    def __init__(self, alpha=1.5, beta=1, sigma=3):
        super().__init__()
        self.alpha = alpha
        self.beta  = beta
        self.sigma = sigma
        self.sigmoid = nn.Sigmoid()


    def forward(self,
                p_maps,
                r_maps,
                targets,
                pos_equal_one,
                pos_equal_one_sum,
                pos_equal_one_for_reg,
                neg_equal_one,
                neg_equal_one_sum):
        p_pos = self.sigmoid(p_maps)
        output_shape = [cfg.MODEL.FEATURE_HIGHT, cfg.MODEL.FEATURE_WIDTH]

        cls_pos_loss = (- pos_equal_one * paddle.log(p_pos + small_addon_for_BCE)) / pos_equal_one_sum
        cls_neg_loss = (- neg_equal_one * paddle.log(1 - p_pos + small_addon_for_BCE)) / neg_equal_one_sum

        cls_loss     = paddle.sum(self.alpha * cls_pos_loss + self.beta * cls_neg_loss)
        cls_pos_loss_rec = paddle.sum(cls_pos_loss)
        cls_neg_loss_rec = paddle.sum(cls_neg_loss)

        reg_loss     = smooth_l1(r_map * pos_equal_one_for_reg, targets * 
                                 pos_equal_one_for_reg, self.sigma) / self.pos_equal_one_sum
        reg_loss     = paddle.sum(reg_loss)

        loss         = paddle.sum(cls_loss + reg_loss)
        delta_output = r_maps
        prob_output  = p_pos

        return loss, delta_output, prob_output