## Combined image_trimap generation ##
import cv2
import tensorflow as tf
import glob
import numpy as np
from tqdm import tqdm

images_base_path = r'C:\Users\Athrva Pandhare\Desktop\New folder (3)\images'
trimap_base_path = r'C:\Users\Athrva Pandhare\Desktop\New folder (3)\trimaps'

combined_target_path = r'C:\Users\Athrva Pandhare\Desktop\New folder (3)\combined_images'

trimap_files = glob.glob(trimap_base_path + '\*.png')
image_files = glob.glob(images_base_path + '\*.jpg')
print("Length : ", len(trimap_files))
print("Length : ", len(image_files))

for i, file in tqdm(enumerate(trimap_files)):
    trimap = cv2.imread(file)
    image = cv2.imread(image_files[i])
    if image is None:
        continue
    
    t_file_name = file.split('\\')[-1].split('.')[0]
    i_file_name = image_files[i].split('\\')[-1].split('.')[0]
    #print(i_file_name)
    assert i_file_name == t_file_name
    assert trimap.shape[-1] == 3
    assert image.shape[-1] == 3
    if image.shape[0] >= 512 and image.shape[1] >= 1024:
        alt_shape = (image.shape[0] - image.shape[0]% 512, image.shape[1] - image.shape[1]%1024)
    elif image.shape[0] < 512 and image.shape[1] >= 1024:
        alt_shape = (512, image.shape[1] - image.shape[1]%1024)
    elif image.shape[1] < 1024 and image.shape[0] >= 512:
        alt_shape = (image.shape[0] - image.shape[0]%512, 1024)
    else:
        alt_shape = (512,1024)
    #print(f"Shape = {alt_shape[0], alt_shape[1]}")
    image = cv2.resize(image, alt_shape, cv2.INTER_NEAREST) # we don't want to interpolate.
    trimap = cv2.resize(trimap, alt_shape, cv2.INTER_NEAREST)
    combined = np.hstack([trimap, image])
    cv2.imwrite(combined_target_path + '\\' + i_file_name + '.png', combined)