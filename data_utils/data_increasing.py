import numpy as np
import os
import glob

'''
通过点云三维旋转、平移，放缩、随机点丢失、index重排,生成新的点云

'''


# TODO 考虑一下是否要归一化
# 先不归一化，因为不是分类而是要分割，需要保证原始的尺寸

def shuffle_points(point_cloud):
    """ Shuffle orders of points in each point cloud -- changes FPS behavior.
        Use the same shuffling idx for the entire batch.
        这个函数用于对每个点云中的点的顺序进行洗牌操作，从而改变最远点采样（FPS）的行为。
        它接受一个形状为NxC的数据作为输入，其中N表示每个点云中的点数，C表示每个点的特征数。
        Input:
            NxC array
        Output:
            NxC array
    """
    idx = np.arange(point_cloud.shape[0])
    np.random.shuffle(idx)
    return point_cloud[idx, :]


def rotate_point_cloud(point_cloud, angle_sigma=0.06, angle_clip=0.18):
    """ Randomly rotate point cloud.
        Input:
          point_cloud: Nx6 array, original point cloud (x, y, z, r, g, b, L)
          angle_range: maximum rotation angle in degrees
        Return:
          Nx6 array, rotated point cloud
    """
    angles = np.clip(angle_sigma * np.random.randn(3), -angle_clip, angle_clip)
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(angles[0]), -np.sin(angles[0])],
                   [0, np.sin(angles[0]), np.cos(angles[0])]])
    Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                   [0, 1, 0],
                   [-np.sin(angles[1]), 0, np.cos(angles[1])]])
    Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                   [np.sin(angles[2]), np.cos(angles[2]), 0],
                   [0, 0, 1]])
    rotation_matrix = np.dot(Rz, np.dot(Ry, Rx))
    rotated_point_cloud = np.dot(point_cloud[:, :3], rotation_matrix)
    rotated_point_cloud = np.concatenate((rotated_point_cloud, point_cloud[:, 3:]), axis=1)
    return rotated_point_cloud


def translate_point_cloud(point_cloud, translation_range=0.1):
    """ Randomly translate point cloud.
        Input:
          point_cloud: Nx6 array, original point cloud (x, y, z, r, g, b, L)
          translation_range: maximum translation range
        Return:
          Nx6 array, translated point cloud
    """
    translation = np.random.uniform(-translation_range, translation_range, size=(3,))
    translated_point_cloud = point_cloud[:, :3] + translation
    translated_point_cloud = np.concatenate((translated_point_cloud, point_cloud[:, 3:]), axis=1)
    return translated_point_cloud


def drop_random_points(point_cloud, max_dropout_ratio=0.2):
    """ Drops random points from the point cloud.随机丢弃点
        Input:
          point_cloud: Nx7 array, original point cloud (x, y, z, r, g, b, L)
          drop_ratio: float, ratio of points to drop (between 0 and 1)
        Return:
          Nx7 array, point cloud with random points dropped
    """
    num_points = point_cloud.shape[0]
    dropout_ratio = np.random.random() * max_dropout_ratio  # 0~0.875
    num_points_to_drop = int(num_points * dropout_ratio)
    if num_points_to_drop > 0:
        drop_indices = np.random.choice(num_points, num_points_to_drop, replace=False)
        remaining_indices = np.delete(np.arange(num_points), drop_indices)
        dropped_point_cloud = point_cloud[remaining_indices]
    else:
        dropped_point_cloud = point_cloud
    return dropped_point_cloud


def scale_point_cloud(point_cloud, scale_range=(0.8, 5.0)):
    """ Randomly scale point cloud.
        Input:
          point_cloud: Nx6 array, original point cloud (x, y, z, r, g, b, L)
          scale_range: tuple (min_scale, max_scale) specifying the range of scaling factors
        Return:
          Nx6 array, scaled point cloud
    """
    scale_factor = np.random.uniform(scale_range[0], scale_range[1])
    scaled_point_cloud = point_cloud[:, :3] * scale_factor
    scaled_point_cloud = np.concatenate((scaled_point_cloud, point_cloud[:, 3:]), axis=1)
    return scaled_point_cloud


def read_point_cloud_file(file_path):
    """ Read point cloud data from txt file.
        Input:
          file_path: path to the txt file
        Return:
          Nx7 array, point cloud data (x, y, z, r, g, b, L)
    """
    point_cloud_data = np.loadtxt(file_path)
    return point_cloud_data


def save_point_cloud_file(point_cloud, file_path):
    fout = open(file_path, 'w')
    for i in range(point_cloud.shape[0]):
        fout.write('%f %f %f %d %d %d %d\n' % \
                   (point_cloud[i, 0], point_cloud[i, 1], point_cloud[i, 2],
                    point_cloud[i, 3], point_cloud[i, 4], point_cloud[i, 5],
                    point_cloud[i, 6]))
    fout.close()
    # np.savetxt(file_path, point_cloud, fmt='%.6f')


def process_point_cloud_folder(folder_path, increse_times=4):
    # 读取文件夹中的所有文件
    subfolders = []
    with os.scandir(folder_path) as entries:
        for entry in entries:
            if entry.is_dir():
                subfolders.append(entry.path)
    for subfolder in subfolders:
        file_paths = glob.glob(os.path.join(subfolder, "*.txt"))
        for file_path in file_paths:
            # 处理每个文件
            point_cloud_data = read_point_cloud_file(file_path)
            for i in range(increse_times):
                # 默认每个文件随机调整4次
                # 进行其他处理操作，例如旋转、平移、缩放、丢弃等
                # shuffle point cloud
                adjust_point_cloud_data = shuffle_points(point_cloud_data)
                # Randomly scale the point cloud
                adjust_point_cloud_data = scale_point_cloud(adjust_point_cloud_data)
                # Randomly rotate the point cloud
                adjust_point_cloud_data = rotate_point_cloud(adjust_point_cloud_data)
                # Randomly translate the point cloud
                adjust_point_cloud_data = translate_point_cloud(adjust_point_cloud_data)
                adjust_point_cloud_data = drop_random_points(adjust_point_cloud_data)
                # 保存处理后的点云数据到新的文件
                output_file_path = file_path.split('.')[0] + 'processed_' + str(i) + '.txt'
                save_point_cloud_file(adjust_point_cloud_data, output_file_path)
                print("Processed point cloud saved to:", output_file_path)


# # Example usage
# file_path = r"E:\BaiduSyncdisk\deeplearning-dataset\cc-4-3-phone-video-20240202\video-4-3-400w_with_labels.txt"
# point_cloud_data = read_point_cloud_file(file_path)
# # shuffle point cloud
# point_cloud_data = shuffle_points(point_cloud_data)
# # Randomly scale the point cloud
# point_cloud_data = scale_point_cloud(point_cloud_data)
# # Randomly rotate the point cloud
# point_cloud_data = rotate_point_cloud(point_cloud_data)
#
# # Randomly translate the point cloud
# point_cloud_data = translate_point_cloud(point_cloud_data)
#
# point_cloud_data = drop_random_points(point_cloud_data)
#
# # Save the transformed point cloud to a new file in the original folder
# output_file_path = os.path.join(os.path.dirname(file_path), "transformed_point_cloud.txt")
#
# save_point_cloud_file(point_cloud_data, output_file_path)
#
# print("Transformed point cloud saved to:", output_file_path)
if __name__ == '__main__':
    folder_path = r"E:\cloud-point-process\deeplearning-dataset"
    process_point_cloud_folder(folder_path, increse_times=10)
