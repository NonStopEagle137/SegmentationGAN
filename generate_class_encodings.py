### Generate Class Encodings ###
import numpy as np
import tensorflow as tf
import glob

trimap_image_path = r'C:\Users\Athrva Pandhare\Desktop\New folder (3)\combined_images'

unique_names = list(set(glob.glob(trimap_image_path + '\*.png')))
encoding = list()

for i, _ in enumerate(unique_names):
    unique_names[i] = unique_names[i].split('\\')[-1].split('.')[0]
    encoding.append([i,unique_names[i]])

with open('encodings.csv', 'a+') as file_:
    for entry in encoding:
        file_.write(str(entry[0]) + ',' + str(entry[1]))
        file_.write('\n')
