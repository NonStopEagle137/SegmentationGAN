### Generating segmentation maps (trimaps) for further semantic segmentation.

import numpy as np
import tensorflow as tf
import cv2
import glob

trimap_base_path = r'C:\Users\Athrva Pandhare\Downloads\Compressed\Oxford_iiit_dataset\annotations\trimaps'
trimap_target_path = r'C:\Users\Athrva Pandhare\Desktop\New folder (3)\trimaps'

trimap_files = glob.glob(trimap_base_path + '\*.png')


count = 0
for file in trimap_files:
    trimap = cv2.imread(file) * 125
    file_name = file.split('\\')[-1].split('.')[0]
    count += 1
    cv2.imwrite(trimap_target_path + '\\' + file_name + '.png', trimap)