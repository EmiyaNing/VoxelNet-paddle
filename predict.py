import paddle
import paddle.nn.functional as F
import numpy as np
from paddle.io import DataLoader

from utils.utils import *
from config import get_config
from models.model import VoxelNet
from data.kitti import KittiDataset


cfg = get_config()
anchors = cal_anchors()


def predict_step(model, data, vis=False):
    tag         = data[0]
    vox_feature = data[1]
    vox_number  = data[2]
    vox_coor    = data[3]
    batch_sz,_, _, _ = vox_feature.shape
    print('predict', tag)
    print('\n\n')
    p_map, r_map = model(vox_feature, vox_number, vox_coor)
    print("Model forward over\n\n")
    deltas = r_map.numpy()
    probs  = F.sigmoid(p_map).numpy()
    print('Now deltas shape = ', deltas.shape)
    print('Now probs shape = ', probs.shape)
    batch_boxes3d  = delta_to_boxes3d(deltas, anchors, coordinate='lidar')
    batch_boxes2d  = batch_boxes3d[:, :, [0, 1, 4, 5, 6]]
    batch_probs    = probs.reshape((cfg.DATA.BATCH_SIZE, -1))
    print('Now batch_boxes2d shape = ',batch_boxes2d.shape)
    print('Now batch_probs shape = ', batch_probs.shape)

    ret_box3d = []
    ret_score = []
    for batch_id in range(batch_sz):
            # remove box with low score
            ind = np.where(batch_probs[batch_id, :] >= cfg.MODEL.RPN_SCORE_THRESH)[0]
            print('Now in batch_id ', str(batch_id), ' ind shape ', ind.shape)
            tmp_boxes3d = batch_boxes3d[batch_id, ind, ...]
            tmp_boxes2d = batch_boxes2d[batch_id, ind, ...]
            tmp_scores = batch_probs[batch_id, ind]
            print('Now in batch_id ', str(batch_id), ' tmp_boxes3d shape ', tmp_boxes3d.shape)
            print('Now in batch_id ', str(batch_id), ' tmp_boxes2d shape ', tmp_boxes2d.shape)
            print('Now in batch_id ', str(batch_id), ' tmp_scores shape ', tmp_scores.shape)
            # TODO: if possible, use rotate NMS
            boxes2d = corner_to_standup_box2d(
                center_to_corner_box2d(tmp_boxes2d, coordinate='lidar'))
            print('Now in batch_id ', str(batch_id), ' boxes2d shape ', boxes2d.shape)
            keep    = py_cpu_nms(boxes2d, tmp_scores, cfg.MODEL.RPN_SCORE_THRESH)
            tmp_boxes3d = tmp_boxes3d[keep, ...]
            tmp_scores = tmp_scores[keep]
            ret_box3d.append(tmp_boxes3d)
            ret_score.append(tmp_scores)

    ret_box3d_score = []
    for box3d, scores in zip(ret_box3d, ret_score):
        ret_box3d_score.append(np.concatenate([np.tile(cfg.MODEL.DETECT_OBJ, len(box3d))[:, np.newaxis], box3d, scores[:, np.newaxis]], axis=-1))



    return tag,ret_box3d_score

def main():
    model = VoxelNet(cfg)
    model_state = paddle.load('voxelnet.pdparams')
    model.set_dict(model_state)
    model.eval()
    dataset = KittiDataset(data_dir=cfg.DATA.DATA_PATH,
                           shuffle=False,
                           aug=False,
                           is_testset=True)
    dataloader = DataLoader(dataset, 
                            batch_size=cfg.DATA.BATCH_SIZE,
                            num_workers=cfg.DATA.NUM_WORKERS,
                            shuffle=False)

    for batch_id, data in enumerate(dataloader):
        tags, results = predict_step(model, data)
        for tag, reuslt in zip(tags, results):
            of_path = './output/data/' + tag + '.txt'
            with open(of_path, 'w+') as f:
                labels = box3d_to_label([result[:, 1:8]], [result[:, 0]], [result[:, -1]], coordinate='lidar')[0]
                for line in lines:
                    f.write(line)
                print('write out {} objects to {}'.format(len(labels), tag))


if __name__ == '__main__':
    main()

    
