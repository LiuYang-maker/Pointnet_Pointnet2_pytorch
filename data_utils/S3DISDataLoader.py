import os
import numpy as np

from tqdm import tqdm
from torch.utils.data import Dataset


class S3DISDataset(Dataset):
    def __init__(self, split='train', data_root='../data/frame_dataset_npy/', num_point=4096, test_area=4, block_size=1.0,
                 sample_rate=1.0, transform=None):
        super().__init__()  # 调用父类的构造函数，初始化基类 Dataset。
        self.num_point = num_point  # 设置实例变量 num_point，表示每个样本点云的点数。
        self.block_size = block_size  # 表示数据块的大小。
        self.transform = transform  # 设置实例变量 transform，表示用于对数据进行转换的函数。
        frames = sorted(os.listdir(data_root))  # 获取指定路径下的所有目录，并对其进行排序，将目录名存储在列表 rooms 中。
        frames = [frame for frame in frames if 'frame-' in frame]  # 筛选出列表 rooms 中包含字符串 'Area_' 的目录。
        # 根据 split 参数的值选择训练集或测试集的房间列表。
        if split == 'train':
            rooms_split = [frame for frame in frames if not '_{}.npy'.format(test_area) in frame]
        else:
            rooms_split = [frame for frame in frames if '_{}.npy'.format(test_area) in frame]
            # rooms_split = [room for room in rooms if 'Area_{}'.format(test_area) in room]
        # 初始化存储点云和标签的列表。
        self.room_points, self.room_labels = [], []
        # 初始化存储每个房间的坐标最小值和最大值的列表。
        self.room_coord_min, self.room_coord_max = [], []
        # 初始化存储每个房间点数的列表。
        num_point_all = []
        # 初始化长度为 10 的零数组，用于统计每个类别的点数。
        labelweights = np.zeros(10)
        # 遍历选定的房间列表，并使用 tqdm 进行迭代可视化。
        for room_name in tqdm(rooms_split, total=len(rooms_split)):
            room_path = os.path.join(data_root, room_name)
            room_data = np.load(room_path)  # xyzrgbl, N*7 从房间numpy文件加载点云数据，room_data 是一个 N*7 的数组，包含 xyzrgbl 信息。
            points, labels = room_data[:, 0:6], room_data[:, 6]  # xyzrgb, N*6; l, N
            tmp, _ = np.histogram(labels, range(11))  # 使用直方图统计每个类别的点数，将结果存储在 tmp 中。
            labelweights += tmp
            # 计算当前房间点云的坐标最小值和最大值。
            coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]
            # 将点云和标签添加到对应的列表中。
            self.room_points.append(points), self.room_labels.append(labels)
            # 将坐标最小值和最大值添加到对应的列表中。
            self.room_coord_min.append(coord_min), self.room_coord_max.append(coord_max)
            num_point_all.append(labels.size)  # 将当前房间的点数添加到 num_point_all 中。
        labelweights = labelweights.astype(np.float32)
        labelweights = labelweights / np.sum(labelweights)  # 归一化，得到每个类别的权重。
        self.labelweights = np.power(np.amax(labelweights) / labelweights, 1 / 3.0)  # 计算每个类别的权重，并进行平方根操作。
        print(self.labelweights)  # 打印类别权重信息。
        sample_prob = num_point_all / np.sum(num_point_all)  # 计算每个房间被采样的概率。
        num_iter = int(np.sum(num_point_all) * sample_rate / num_point)  # 计算总采样次数
        room_idxs = []  # 初始化房间索引列表。
        for index in range(len(rooms_split)):
            # temp1 = sample_prob[index] * num_iter  # index对应的房间要采样的次数
            # temp = [index] * int(round(sample_prob[index] * num_iter))  # 根据采样概率将房间索引重复添加到列表中。
            room_idxs.extend([index] * int(round(sample_prob[index] * num_iter)))  # 根据采样概率将房间索引重复添加到列表中。
        self.room_idxs = np.array(room_idxs)
        print("Totally {} samples in {} set.".format(len(self.room_idxs), split))  # 打印数据集中的样本总数。

    def __getitem__(self, idx):
        """
        idx : 采样索引
        """
        room_idx = self.room_idxs[idx]
        points = self.room_points[room_idx]  # N * 6
        labels = self.room_labels[room_idx]  # N
        N_points = points.shape[0]

        while (True):
            center = points[np.random.choice(N_points)][:3]
            block_min = center - [self.block_size / 2.0, self.block_size / 2.0, 0]
            block_max = center + [self.block_size / 2.0, self.block_size / 2.0, 0]
            point_idxs = np.where(
                (points[:, 0] >= block_min[0]) & (points[:, 0] <= block_max[0]) & (points[:, 1] >= block_min[1]) & (
                        points[:, 1] <= block_max[1]))[0]
            if point_idxs.size > 1024:
                break

        if point_idxs.size >= self.num_point:
            selected_point_idxs = np.random.choice(point_idxs, self.num_point, replace=False)
        else:
            selected_point_idxs = np.random.choice(point_idxs, self.num_point, replace=True)

        # normalize
        selected_points = points[selected_point_idxs, :]  # num_point * 6
        current_points = np.zeros((self.num_point, 9))  # num_point * 9
        current_points[:, 6] = selected_points[:, 0] / self.room_coord_max[room_idx][0]
        current_points[:, 7] = selected_points[:, 1] / self.room_coord_max[room_idx][1]
        current_points[:, 8] = selected_points[:, 2] / self.room_coord_max[room_idx][2]
        selected_points[:, 0] = selected_points[:, 0] - center[0]
        selected_points[:, 1] = selected_points[:, 1] - center[1]
        selected_points[:, 3:6] /= 255.0
        current_points[:, 0:6] = selected_points
        current_labels = labels[selected_point_idxs]
        if self.transform is not None:
            current_points, current_labels = self.transform(current_points, current_labels)
        return current_points, current_labels

    def __len__(self):
        """
        返回数据集中样本的总数。
        """
        return len(self.room_idxs)


class ScannetDatasetWholeScene():
    # prepare to give prediction on each points
    def __init__(self, root, block_points=4096, split='test', test_area=5, stride=0.5, block_size=1.0, padding=0.001):
        self.block_points = block_points
        self.block_size = block_size
        self.padding = padding
        self.root = root
        self.split = split
        self.stride = stride
        self.scene_points_num = []
        assert split in ['train', 'test']
        if self.split == 'train':
            self.file_list = [d for d in os.listdir(root) if d.find('Area_%d' % test_area) is -1]
        else:
            self.file_list = [d for d in os.listdir(root) if d.find('Area_%d' % test_area) is not -1]
        self.scene_points_list = []
        self.semantic_labels_list = []
        self.room_coord_min, self.room_coord_max = [], []
        for file in self.file_list:
            data = np.load(root + file)
            points = data[:, :3]
            self.scene_points_list.append(data[:, :6])
            self.semantic_labels_list.append(data[:, 6])
            coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]
            self.room_coord_min.append(coord_min), self.room_coord_max.append(coord_max)
        assert len(self.scene_points_list) == len(self.semantic_labels_list)

        labelweights = np.zeros(13)
        for seg in self.semantic_labels_list:
            tmp, _ = np.histogram(seg, range(14))
            self.scene_points_num.append(seg.shape[0])
            labelweights += tmp
        labelweights = labelweights.astype(np.float32)
        labelweights = labelweights / np.sum(labelweights)
        self.labelweights = np.power(np.amax(labelweights) / labelweights, 1 / 3.0)

    def __getitem__(self, index):
        point_set_ini = self.scene_points_list[index]
        points = point_set_ini[:, :6]
        labels = self.semantic_labels_list[index]
        coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]
        grid_x = int(np.ceil(float(coord_max[0] - coord_min[0] - self.block_size) / self.stride) + 1)
        grid_y = int(np.ceil(float(coord_max[1] - coord_min[1] - self.block_size) / self.stride) + 1)
        data_room, label_room, sample_weight, index_room = np.array([]), np.array([]), np.array([]), np.array([])
        for index_y in range(0, grid_y):
            for index_x in range(0, grid_x):
                s_x = coord_min[0] + index_x * self.stride
                e_x = min(s_x + self.block_size, coord_max[0])
                s_x = e_x - self.block_size
                s_y = coord_min[1] + index_y * self.stride
                e_y = min(s_y + self.block_size, coord_max[1])
                s_y = e_y - self.block_size
                point_idxs = np.where(
                    (points[:, 0] >= s_x - self.padding) & (points[:, 0] <= e_x + self.padding) & (
                            points[:, 1] >= s_y - self.padding) & (
                            points[:, 1] <= e_y + self.padding))[0]
                if point_idxs.size == 0:
                    continue
                num_batch = int(np.ceil(point_idxs.size / self.block_points))
                point_size = int(num_batch * self.block_points)
                replace = False if (point_size - point_idxs.size <= point_idxs.size) else True
                point_idxs_repeat = np.random.choice(point_idxs, point_size - point_idxs.size, replace=replace)
                point_idxs = np.concatenate((point_idxs, point_idxs_repeat))
                np.random.shuffle(point_idxs)
                data_batch = points[point_idxs, :]
                normlized_xyz = np.zeros((point_size, 3))
                normlized_xyz[:, 0] = data_batch[:, 0] / coord_max[0]
                normlized_xyz[:, 1] = data_batch[:, 1] / coord_max[1]
                normlized_xyz[:, 2] = data_batch[:, 2] / coord_max[2]
                data_batch[:, 0] = data_batch[:, 0] - (s_x + self.block_size / 2.0)
                data_batch[:, 1] = data_batch[:, 1] - (s_y + self.block_size / 2.0)
                data_batch[:, 3:6] /= 255.0
                data_batch = np.concatenate((data_batch, normlized_xyz), axis=1)
                label_batch = labels[point_idxs].astype(int)
                batch_weight = self.labelweights[label_batch]

                data_room = np.vstack([data_room, data_batch]) if data_room.size else data_batch
                label_room = np.hstack([label_room, label_batch]) if label_room.size else label_batch
                sample_weight = np.hstack([sample_weight, batch_weight]) if label_room.size else batch_weight
                index_room = np.hstack([index_room, point_idxs]) if index_room.size else point_idxs
        data_room = data_room.reshape((-1, self.block_points, data_room.shape[1]))
        label_room = label_room.reshape((-1, self.block_points))
        sample_weight = sample_weight.reshape((-1, self.block_points))
        index_room = index_room.reshape((-1, self.block_points))
        return data_room, label_room, sample_weight, index_room

    def __len__(self):
        return len(self.scene_points_list)


if __name__ == '__main__':
    # data_root = '/data/yxu/PointNonLocal/data/stanford_indoor3d/'
    data_root = '../data/frame_dataset_npy/'
    num_point, test_area, block_size, sample_rate = 4096, 4, 1.0, 1.0

    point_data = S3DISDataset(split='train', data_root=data_root, num_point=num_point, test_area=test_area,
                              block_size=block_size, sample_rate=sample_rate, transform=None)
    print('point data size:', point_data.__len__())
    print('point data 0 shape:', point_data.__getitem__(0)[0].shape)
    print('point label 0 shape:', point_data.__getitem__(0)[1].shape)
    import torch, time, random

    manual_seed = 123
    random.seed(manual_seed)
    np.random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    torch.cuda.manual_seed_all(manual_seed)


    def worker_init_fn(worker_id):
        random.seed(manual_seed + worker_id)


    train_loader = torch.utils.data.DataLoader(point_data, batch_size=16, shuffle=True, num_workers=16, pin_memory=True,
                                               worker_init_fn=worker_init_fn)
    for idx in range(4):
        end = time.time()
        for i, (input, target) in enumerate(train_loader):
            print('time: {}/{}--{}'.format(i + 1, len(train_loader), time.time() - end))
            end = time.time()
