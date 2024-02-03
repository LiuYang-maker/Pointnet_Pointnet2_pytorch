import os
import sys
from indoor3d_util import DATA_PATH, collect_point_label
'''
这个文件是只针对原作者针对S3DIS的数据处理程序
'''
# 获取当前脚本的绝对路径以及根目录路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
# 将当前脚本所在目录添加到系统路径中
sys.path.append(BASE_DIR)

# 从文件中读取标注文件的路径列表
anno_paths = [line.rstrip() for line in open(os.path.join(BASE_DIR, 'meta/anno_paths.txt'))]
# 将相对路径转换为绝对路径
anno_paths = [os.path.join(DATA_PATH, p) for p in anno_paths]

# 设置输出文件夹的路径，并确保该文件夹存在，如果不存在则创建
output_folder = os.path.join(ROOT_DIR, 'data/stanford_indoor3d')
if not os.path.exists(output_folder):
    os.mkdir(output_folder)

# Note: there is an extra character in the v1.2 data in Area_5/hallway_6. It's fixed manually.
# 注意：在 v1.2 数据中的 Area_5/hallway_6 中存在额外的字符，需要手动修复。
# 遍历标注文件的路径列表，处理每一个标注文件
for anno_path in anno_paths:
    # 打印当前正在处理的标注文件路径
    print(anno_path)
    try:
        # 解析标注文件路径，提取出区域名和走廊名，并拼接成输出文件的文件名
        elements = anno_path.split('/')
        out_filename = elements[-3].split('\\')[-1]+'_'+elements[-2]+'.npy' # Area_1_hallway_1.npy
        # 调用 collect_point_label 函数，将标注文件转换为 numpy 格式并保存
        collect_point_label(anno_path, os.path.join(output_folder, out_filename), 'numpy')
    except:
        print(anno_path, 'ERROR!!')
