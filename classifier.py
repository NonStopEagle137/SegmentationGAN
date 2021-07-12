### Classification model ###
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import numpy as np
import cv2
import glob

class discriminator_:
    def __init__(self):
        inp = tf.keras.layers.Input(shape=[None, None, 3], name='input_image')
        tar = tf.keras.layers.Input(shape=[None, None, 3], name='target_image')
        x = tf.keras.layers.concatenate([inp, tar], axis = 1)
        discriminator_loaded = tf.keras.applications.DenseNet121(
                        include_top=False,
                        weights="imagenet",
                        input_tensor=None,
                        pooling=None,
                        classes=1000,
                        )
        out = discriminator_loaded(x)
        self.discriminator = tf.keras.Model(inputs = [inp, tar], outputs = out)
        # self.discriminator.trainable = False
        
    def get_model(self):

        return self.discriminator
    
    
    
    
class seg_cls:
    def __init__(self, num_classes): # notice that there is no input shape here.
        self.num_classes = num_classes;
    
    def _upsample(self, filters, size, apply_dropout=False):
        initializer = tf.random_normal_initializer(0., 0.02)

        result = tf.keras.Sequential()
        result.add(
        tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                        padding='same',
                                        kernel_initializer=initializer,
                                        use_bias=False))

        result.add(tf.keras.layers.BatchNormalization())

        if apply_dropout:
          result.add(tf.keras.layers.Dropout(0.5))

        result.add(tf.keras.layers.ReLU())

        return result
    
    def _downsample(self, filters, size, apply_batchnorm=True):
        initializer = tf.random_normal_initializer(0., 0.02)

        result = tf.keras.Sequential()
        result.add(
          tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                                 kernel_initializer=initializer, use_bias=False))

        if apply_batchnorm:
            result.add(tf.keras.layers.BatchNormalization())

            result.add(tf.keras.layers.LeakyReLU())

        return result

    def generate_model(self):
        
        base = tf.keras.applications.DenseNet121(include_top=False,
                                         weights='imagenet')
        # base.trainable = False # freezing the pre-triained layers
        x1 = tf.keras.layers.Conv2D(64, (5,5), strides=1, padding='valid')(base.output)
        x2 = tf.keras.layers.Conv2D(64, (5,5), strides=1, padding='valid')(x1)
        x3 = tf.keras.layers.Conv2D(64, (3,3), strides=1, padding='valid')(x2)
        x4 = tf.keras.layers.Conv2D(self.num_classes, (1,1), strides=1, padding='valid')(x3)
        class_out = tf.keras.layers.GlobalMaxPooling2D()(x4)
        class_out = tf.keras.layers.Activation(activation = 'sigmoid')(class_out)
        
        
        y1 = self._upsample(128, 4, apply_dropout = True)(base.output)
        y2 = self._upsample(64, 4, apply_dropout = True)(y1)
        y3 = self._upsample(32, 4, apply_dropout = True)(y2)
        y4 = self._upsample(16, 4, apply_dropout = True)(y3)
        trimap_out = self._upsample(3, 4, apply_dropout = True)(y4)
        trimap_out = tf.keras.layers.Activation(activation='sigmoid')(trimap_out)
        self.model = tf.keras.Model(inputs=base.input, outputs=[class_out, trimap_out])
        
        return self.model

"""
clf = seg_cls(num_classes = 2)
shape_ = (224,224,3)
model = clf.generate_model()
pred = model(np.expand_dims(np.random.uniform(size = shape_),axis = 0))
print(pred[0].shape, pred[1].shape)
"""
	