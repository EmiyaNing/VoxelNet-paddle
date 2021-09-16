import sys
sys.path.append('..')
import numpy as np

from config import get_config
cfg = get_config()

data_dir = 'velodyne'

def process_pointcloud(point_cloud, cls=cfg.MODEL.DETECT_OBJ):
    '''
        总结一下这个函数的大致功能。
        首先输入时(N, 4)的点云数据，N表示一张图中一共有N个点，每个点按x,y,z,r的格式存储。
        这个函数首先按照论文的需求，将每张图按照网格进行划分，然后从每个网格中随机选取T个点，
        最后输出的数据格式为[Area_number, T, 4]
    '''
    # Input:
    #   (N, 4)
    # Output:
    #   voxel_dict
    if cls == 'Car':
        scene_size = np.array([4, 80, 70.4], dtype=np.float32)
        # 假设每个小网格的尺寸为[0.4, 0.2, 0.2]
        voxel_size = np.array([0.4, 0.2, 0.2], dtype=np.float32)
        grid_size = np.array([10, 400, 352], dtype=np.int64)
        lidar_coord = np.array([0, 40, 3], dtype=np.float32)
        max_point_number = 35
        max_area_number  = 15000
    else:
        scene_size = np.array([4, 40, 48], dtype=np.float32)
        voxel_size = np.array([0.4, 0.2, 0.2], dtype=np.float32)
        grid_size = np.array([10, 200, 240], dtype=np.int64)
        lidar_coord = np.array([0, 20, 3], dtype=np.float32)
        max_point_number = 45
        max_area_number  = 15000

        np.random.shuffle(point_cloud)

    shifted_coord = point_cloud[:, :3] + lidar_coord

    # reverse the point cloud coordinate (X, Y, Z) -> (Z, Y, X)
    # 通过shifted_coord/voxel_size操作，将整张图中的所有点云进行放大。
    # 在完成放大操作之后，也相当完成了网格的划分操作。
    # 以类别car为例子，每个网格的大小应该为[2.5, 5, 5]
    voxel_index = np.floor(
        shifted_coord[:, ::-1] / voxel_size).astype(np.int)


    # filter the point which out of range.
    # 下面的代码对整张图中超出场景范围的点云进行过滤。
    bound_x = np.logical_and(
        voxel_index[:, 2] >= 0, voxel_index[:, 2] < grid_size[2])
    bound_y = np.logical_and(
        voxel_index[:, 1] >= 0, voxel_index[:, 1] < grid_size[1])
    bound_z = np.logical_and(
        voxel_index[:, 0] >= 0, voxel_index[:, 0] < grid_size[0])

    bound_box = np.logical_and(np.logical_and(bound_x, bound_y), bound_z)


    point_cloud = point_cloud[bound_box]
    voxel_index = voxel_index[bound_box]


    # [K, 3] coordinate buffer as described in the paper
    # 通过上述操作后，每个点都被划分到了一个voxel当中。
    # 然后根据我们的需求,我们需要找到K个含有point的voxel。
    coordinate_buffer = np.unique(voxel_index, axis=0)


    K = len(coordinate_buffer)
    T = max_point_number
    
    empty_buffer      = np.zeros(shape=(max_area_number - K, 3), dtype=np.int) 
    
    # 干脆点，我把这个每张图中的最大voxel_number 设置为15000.
    # 过滤完了之后，这一张图中一共K个区域包含point。
    # [max_area_number, 1] store number of points in each voxel grid
    number_buffer = np.zeros(shape=(max_area_number), dtype=np.int64)

    # [max_area_number, T, 7] feature buffer as described in the paper
    feature_buffer = np.zeros(shape=(max_area_number, T, 7), dtype=np.float32)

    # build a reverse index for coordinate buffer
    index_buffer = {}
    for i in range(K):
        index_buffer[tuple(coordinate_buffer[i])] = i

    for voxel, point in zip(voxel_index, point_cloud):
        index = index_buffer[tuple(voxel)]
        number = number_buffer[index]
        if number < T:
            feature_buffer[index, number, :4] = point
            number_buffer[index] += 1

    feature_buffer[:, :, -3:] = feature_buffer[:, :, :3] - \
        feature_buffer[:, :, :3].sum(axis=1, keepdims=True)/number_buffer.reshape(max_area_number, 1, 1)

    # splicing the empty_buffer with coordinate_buffer
    coordinate_buffer = np.concatenate([coordinate_buffer, empty_buffer], axis=0)
   
    voxel_dict = {'feature_buffer': feature_buffer,
                  'coordinate_buffer': coordinate_buffer,
                  'number_buffer': number_buffer}
    return voxel_dict