import glob
import os
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
subfolder = os.path.join(BASE_DIR, 'data/frame_dataset_txt')
file_paths = glob.glob(os.path.join(subfolder, "*.txt"))
for file_path in file_paths:
    point_cloud_data = np.loadtxt(file_path)
    temp_names = file_path.split('\\')[-1].split('-')
    new_name = 'frame-'+temp_names[1]+'-'+temp_names[2]+'-'+temp_names[-2]+'-'+temp_names[-1].split('.')[0]+'.npy'
    new_path = os.path.join(BASE_DIR, 'data/frame_dataset_npy/'+new_name)
    np.save(new_path, point_cloud_data)
    print(new_path)
