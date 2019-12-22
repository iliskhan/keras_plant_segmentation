import os
import cv2
import keras
import numpy as np 

from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Conv2D, UpSampling2D


train_dir = "../train"
test_dir = "../test"



def data_reader(dir_name, width, height):

    imgs_dir_name = os.path.join(dir_name, 'imgs')
    masks_dir_name = os.path.join(dir_name, 'masks')

    imgs_names = [os.path.join(imgs_dir_name, name) for name in os.listdir(imgs_dir_name)]
    masks_names = [os.path.join(masks_dir_name, name) for name in os.listdir(masks_dir_name)]

    imgs = np.array([cv2.resize(cv2.imread(name), (width, height)) for name in imgs_names])
    masks = np.array([cv2.resize(cv2.imread(name), (width, height)) for name in masks_names])

    return imgs, masks


width, height = 256, 256

train_x, train_y = data_reader(train_dir, width, height)
test_x, test_y = data_reader(test_dir, width, height)

best_w = keras.callbacks.ModelCheckpoint('models/fcn_best.h5', 
                                        monitor='val_loss',
                                        verbose=0,
                                        save_best_only=True,
                                        save_weights_only=True,
                                        mode='auto',
                                        period=1)

last_w = keras.callbacks.ModelCheckpoint('models/fcn_last.h5', 
                                        monitor='val_loss',
                                        verbose=0,
                                        save_best_only=False,
                                        save_weights_only=True,
                                        mode='auto',
                                        period=1)

callbacks = [best_w, last_w]

base_model = VGG16(weights='imagenet', input_shape=(width, height, 3), include_top=False)

base_out = base_model.output

up = UpSampling2D(32)(base_out)




