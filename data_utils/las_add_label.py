import laspy
import json
import numpy as np
import open3d as o3d


def point_cloud_add_label(point_cloud_filename, label_json):
    # 读取JSON文件
    # json_file = r"E:\point_cloud_process_by_software\las-str\phone-1-2-20240108-withGCP-0-50-1-100mb.json"
    with open(label_json) as json_file:
        data = json.load(json_file)
    # 提取数据
    index_category_dict = data['index_category_dict']
    categorys = data['categorys']

    point_cloud_type = point_cloud_filename.split('.')[-1]
    xyz = []
    other_fields = []
    if point_cloud_type == 'las':
        point_cloud = laspy.read(point_cloud_filename)
        # 获取原始的 XYZ 坐标和其他字段
        xyz = np.vstack([point_cloud.x, point_cloud.y, point_cloud.z]).transpose()
        other_fields = np.vstack([point_cloud.red, point_cloud.green, point_cloud.blue]).transpose()  # 例子：RGB颜色
    elif point_cloud_type == 'pcd':
        point_cloud = o3d.io.read_point_cloud(point_cloud_filename)
        # 将点云转换为NumPy数组
        points = np.asarray(point_cloud.points)
        xyz = points[:, :3]
        # 获取RGB信息（如果有的话）
        if point_cloud.has_colors():
            other_fields = np.asarray(point_cloud.colors)
        else:
            other_fields = None
    elif point_cloud_type == 'ply':
        point_cloud = o3d.io.read_point_cloud(point_cloud_filename)
        # 获取点云的XYZ坐标
        xyz = np.asarray(point_cloud.points)
        # 获取点云的RGB颜色信息（如果有的话）
        other_fields = np.asarray(point_cloud.colors)
        # 将颜色值从float转换为8位整数
        if other_fields is not None:
            other_fields = (other_fields * 255).astype(int)
    elif point_cloud_type == 'txt':
        with open(point_cloud_filename, 'r') as f:
            for line in f:
                coordinates = line.strip().split(" ")
                xyz1 = [float(coord) for coord in coordinates[:3]]
                other_field = [int(coord) for coord in coordinates[3:6]]
                xyz.append(xyz1)
                other_fields.append(other_field)
            xyz = np.asarray(xyz)
            other_fields = np.asarray(other_fields)
    # 在数据中添加一个示例标签列（可以根据需要修改）
    labels = np.asarray(categorys)
    labels = labels[:, np.newaxis]
    # labels = np.transpose(labels)

    # 将标签列添加到原始数据
    new_fields = np.hstack([other_fields, labels])

    # 用于存储每个标注文件中的点和标签

    # 创建一个新的 LAS 文件
    # out_las = laspy.create(point_format=2)
    # out_las.x = xyz[:, 0]
    # out_las.y = xyz[:, 1]
    # out_las.z = xyz[:, 2]
    # out_las.red = new_fields[:, 0]
    # out_las.green = new_fields[:, 1]
    # out_las.blue = new_fields[:, 2]
    # out_las.label = new_fields[:, 3]  # 新的标签列
    #
    # # 保存新的 LAS 文件
    # out_las_file_path = las_name.split('.')[0] + '_with_labels.las'
    # out_las.write(out_las_file_path)

    out_txt_file_path = point_cloud_filename.split('.')[0] + '_with_labels.txt'
    fout = open(out_txt_file_path, 'w')
    for i in range(xyz.shape[0]):
        fout.write('%f %f %f %d %d %d %d\n' % \
                   (xyz[i, 0], xyz[i, 1], xyz[i, 2],
                    new_fields[i, 0], new_fields[i, 1], new_fields[i, 2],
                    new_fields[i, 3]))
    fout.close()


def las_to_txt(input_las):
    # 读取LAS文件
    las_file = laspy.read(input_las)


def read_las(file_path):
    las_file = laspy.read(file_path)
    # 提取位置信息
    xyz = list(zip(las_file.x, las_file.y, las_file.z))
    # 提取颜色信息并进行归一化
    rgb = list(zip(las_file.red / 65535.0, las_file.green / 65535.0, las_file.blue / 65535.0))
    return xyz, rgb


if __name__ == '__main__':
    # 读取JSON文件
    json_file = r"E:\BaiduSyncdisk\deeplearning-dataset\cc-4-3-phone-video-20240202\video-4-3-400w.json"
    point_cloud_name = r"E:\BaiduSyncdisk\deeplearning-dataset\cc-4-3-phone-video-20240202\video-4-3-400w.txt"

    point_cloud_add_label(point_cloud_name, json_file)
