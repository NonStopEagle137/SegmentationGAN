import tensorflow as tf
import os
import time
from matplotlib import pyplot as plt
from IPython import display
import numpy as np
from tqdm import tqdm
import cv2

sample_num = 0


def initilaize_parameters(transform_path, processed_path, buffer_size,
                            batch_size, img_width, img_height,
                            channels, lambda_,
                            print_defaults = False):
    global BATCH_SIZE, BUFFER_SIZE, IMG_HEIGHT, IMG_WIDTH, OUTPUT_CHANNELS, LAMBDA, PATH_PROCESSED, PATH_TRANSFORM
    PATH_TRANSFORM = transform_path
    PATH_PROCESSED = processed_path

    BUFFER_SIZE = buffer_size
    BATCH_SIZE = batch_size
    IMG_WIDTH = img_width
    IMG_HEIGHT = img_height
    OUTPUT_CHANNELS = channels
    LAMBDA = lambda_
    return
    


def load_processed(image_file):
  image = tf.io.read_file(image_file)
  image = tf.image.decode_jpeg(image)

  real_image = image.copy()
  

  
  real_image = tf.cast(real_image, tf.float32)

  return real_image
def load_transform(image_file):
  image = tf.io.read_file(image_file)
  image = tf.image.decode_jpeg(image)

  input_image = image.copy()

  input_image = tf.cast(input_image, tf.float32)

  return input_image

def resize(input_image, height, width):
  input_image = tf.image.resize(input_image, [height, width],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  #real_image = tf.image.resize(real_image, [height, width],
  #                             method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

  return input_image

def random_crop(input_image, real_image):
  stacked_image = tf.stack([input_image, real_image], axis=0)
  cropped_image = tf.image.random_crop(
      stacked_image, size=[2, IMG_HEIGHT, IMG_WIDTH, 3])

  return cropped_image[0], cropped_image[1]

def normalize(input_image):
  input_image = tf.cast(input_image, tf.float32)
  #real_image = tf.cast(real_image, tf.float32)
  input_image = (input_image / 127.5) - 1
  #real_image = (real_image / 127.5) - 1

  return input_image
  
@tf.function()
def random_jitter(input_image, real_image):
  # resizing to 286 x 286 x 3
  #input_image, real_image = resize(input_image, real_image, 286, 286)

  # randomly cropping to 256 x 256 x 3
  input_image, real_image = random_crop(input_image, real_image)

  if tf.random.uniform(()) > 0.5:
    # random mirroring
    input_image = tf.image.flip_left_right(input_image)
    real_image = tf.image.flip_left_right(real_image)

  return input_image, real_image
  
def load_image_train(image_file_transform, image_file_processed):
  input_image = load_processed(image_file_processed)
  real_image = load_transform(image_file_transform)
  input_image, real_image = random_jitter(input_image, real_image)
  input_image, real_image = normalize(input_image, real_image)

  return input_image, real_image
  
def load_image_test(image_file_transform, image_file_processed):
  input_image = load_processed(image_file_processed)
  real_image = load_transform(image_file_transform)
  input_image, real_image = resize(input_image, real_image,
                                   IMG_HEIGHT, IMG_WIDTH)
  input_image, real_image = normalize(input_image, real_image)

  return input_image, real_image


def check_entry(index):
  fig, ax = plt.subplots(2)
  ax[0].imshow(cv2.imread(test_dataset_processed[index]))
  ax[1].imshow(cv2.imread(test_dataset_transform[index]))
  return


def grab_train_img_from_names(train_dataset_processed):
  train_dataset = []
  for i in tqdm(range(len(train_dataset_processed))):
    input = cv2.imread(train_dataset_transform[i])
    real = cv2.imread(train_dataset_processed[i])
    input, real = resize(input, real, 512, 512)
    input, real = normalize(input, real)
    train_dataset.append(np.array([input, real]))
  train_dataset = np.array(train_dataset)
  return train_dataset


def grab_img_from_names(test_dataset_processed):
  test_dataset = []
  for i in range(len(test_dataset_processed)):
    real = cv2.imread(test_dataset_processed[i])
    #real= resize(real, 512, 512)
    real = normalize(real)
    test_dataset.append(real)
  # test_dataset = np.array(test_dataset)
  return test_dataset

def generate_images(model, test_input, tar):
  global sample_num
  sample_num += 1
  prediction = model(np.expand_dims(test_input,0), training=True)
  class_out = prediction[0]
  prediction = prediction[-1]
  if class_out.numpy()[0,0] < class_out.numpy()[0,1]:
    class_name = 'Dog'
  elif class_out.numpy()[0,0] > class_out.numpy()[0,1]:
    class_name = 'Cat'
  plt.figure(figsize=(15, 15))

  display_list = [test_input, tar, prediction[0]]
  title = ['Input Image', 'Ground Truth', 'Predicted Image']
  display_list[0] = display_list[0] * 0.5 + 0.5
  display_list[1] = display_list[1] * 0.5 + 0.5
  display_list[2] = display_list[2] * 0.5 + 0.5
  img = np.hstack([display_list[0], display_list[1], display_list[2]])
  try:
      img = img.numpy()
  except:
      pass
  plt.imsave(r'C:\Users\Athrva Pandhare\Desktop\New folder (4)\Samples\{}_{}.jpg'.format(class_name, sample_num),img) 
  plt.axis('off')
  plt.close()