import glob
import os
import numpy as np


def main():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    subfolder = os.path.join(BASE_DIR, 'data/frame_dataset_txt')
    file_paths = glob.glob(os.path.join(subfolder, "*.txt"))
    for file_path in file_paths:
        point_cloud_data = np.loadtxt(file_path)
        temp_names = file_path.split('\\')[-1].split('-')
        new_name = 'frame-' + temp_names[1] + '-' + temp_names[2] + '-' + temp_names[-2] + '-' + \
                   temp_names[-1].split('.')[0] + '.npy'
        new_path = os.path.join(BASE_DIR, 'data/frame_dataset_npy/' + new_name)
        np.save(new_path, point_cloud_data)
        print(new_path)


def main3():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    subfolder = os.path.join(BASE_DIR, 'data/temp')
    file_paths = glob.glob(os.path.join(subfolder, "*.npy"))
    for file_path in file_paths:
        room_data = np.load(file_path)
        room_data[:, -1] -= 1
        np.save(file_path, room_data)
        print(file_path + ':success!')


def main2():
    # 定义一个类别名称的列表。
    classes = ['ground', 'wall', 'column', 'beam', 'weight', 'bolt', 'floor', 'foundation', 'sundries', 'joint']
    # 创建一个将类别名称映射到整数标签的字典。
    class2label = {cls: i + 1 for i, cls in enumerate(classes)}
    seg_classes = class2label
    # 创建一个将整数标签映射到类别名称的字典。
    seg_label_to_cat = {}
    for i, cat in enumerate(seg_classes.keys()):
        seg_label_to_cat[i + 1] = cat


def check_npy():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    subfolder = os.path.join(BASE_DIR, 'data/frame_dataset_npy')
    file_paths = glob.glob(os.path.join(subfolder, "*.npy"))
    for file_path in file_paths:
        room_data = np.load(file_path)
        for e in room_data[:, -1]:
            if e < 0 or e > 9:
                print(file_path + ':exist error! e='+str(e))

def get_data():
    file_path = r"C:\Users\Administrator\Desktop\tempdata.txt"
    new_file_path = r"C:\Users\Administrator\Desktop\new_tempdata.txt"
    data = np.loadtxt(file_path)
    new_data = []
    # j = 0
    for i in range(int(len(data)/10)):
        j = i * 10
        tempdata = [data[j],data[j+1],data[j+2],data[j+3],data[j+4],data[j+5],data[j+6],data[j+7],data[j+8],data[j+9]]
        new_data.append(tempdata)
        # new_data[j] = tempdata
        # j = j+1
    new_data = np.asarray(new_data)
    np.savetxt(new_file_path, new_data)


if __name__ == '__main__':
    # main3()
    # check_npy()
    get_data()
