# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %%
import os
import cv2
import keras
import imutils

import numpy as np 
import matplotlib.pyplot as plt
import segmentation_models as sm

import imgaug.augmenters as iaa

from sklearn.utils import shuffle
from keras.applications.vgg16 import VGG16
from keras.models import Model


# %%
# seq = iaa.Sequential([
#     iaa.Flipud(1), # crop images from each side by 0 to 16px (randomly chosen)
#     iaa.Fliplr(1), # horizontally flip 50% of the images
# ])


# %%
train_dir = "../data/train"
test_dir = "../data/test"


# %%
def thresholder(img):
    ret,thresh = cv2.threshold(img,127,255,cv2.THRESH_BINARY)

    thresh[thresh != 0] = 1
    
    return thresh
    


# %%
def data_reader(dir_name):

    imgs_dir_name = os.path.join(dir_name, 'imgs')
    masks_dir_name = os.path.join(dir_name, 'masks')

    imgs_names = [os.path.join(imgs_dir_name, name) for name in os.listdir(imgs_dir_name)]
    masks_names = [os.path.join(masks_dir_name, name) for name in os.listdir(masks_dir_name)]

    imgs = np.array([cv2.imread(name) for name in imgs_names])
    masks = np.array([thresholder(cv2.imread(name, 0)) for name in masks_names])

    return imgs, masks.reshape((*masks.shape, 1))


# %%
# width, height = 640, 480

train_x, train_y = data_reader(train_dir)
test_x, test_y = data_reader(test_dir)


# %%
def rotator(x, y):
    print("shape before", x.shape)
    aug_x, aug_y = [], []
    
    for angle in (0, 45, 90, 135, 180, 225, 270, 315):
        
        images_aug, masks_aug = iaa.Affine(rotate=(angle))(images=x, segmentation_maps=y)
        aug_x.append(images_aug), aug_y.append(masks_aug)
        
#     print(len(aug_x))
    
    x, y = np.array(aug_x), np.array(aug_y)
    print("shape after", x.shape)
    
    x = x.reshape(-1, *x.shape[2:])
    y = y.reshape(-1, *y.shape[2:])
    
    print("after reshape", x.shape)
    
    return x, y


# %%
def blurring(x, y):    

    images_aug, masks_aug = iaa.GaussianBlur(sigma=0.2)(images=x, segmentation_maps=y)
    
    return np.concatenate([x, images_aug]), np.concatenate([y, masks_aug])


# %%
def turner(x, y):    
    
    images_aug, masks_aug = iaa.Fliplr(1.0)(images=x, segmentation_maps=y)
    
    return np.concatenate([x, images_aug]), np.concatenate([y, masks_aug])


# %%
def brightness_changer(x, y):
    
    dark_imgs, dark_masks = iaa.Multiply(0.7)(images=x, segmentation_maps=y)
    light_imgs, light_masks = iaa.Multiply(1.5)(images=x, segmentation_maps=y)

    return np.concatenate([x, light_imgs, dark_imgs]), np.concatenate([y, light_masks, dark_masks])


# %%
def augmentation(x, y):
    x, y = turner(x, y)
    x, y = brightness_changer(x, y)
    x, y = rotator(x, y)
#     x, y = blurring(x, y)
    
    print("output data shape =", x.shape)
    return x, y


# %%
def scaler(x, y):
    aug_x, aug_y = [], []
    

    for i in range(len(x)):
        img = x[i]
        mask = y[i]
        
        img_height = img.shape[1]
        mask_height = mask.shape[1]
        
        for scale in (0.75, 0.5):
            aug_x.append(imutils.resize(img, height=int(img_height * scale)))
            aug_y.append(imutils.resize(mask, height=int(mask_height * scale)))
                    
    x, y = np.array(aug_x), np.array(aug_y)
    
    x = x.reshape(-1, *x.shape[2:])
    y = y.reshape(-1, *y.shape[2:])
        
    return x, y


# %%
train_x, train_y = augmentation(train_x, train_y)
test_x, test_y = augmentation(test_x, test_y)


# %%
# train_x, train_y = scaler(train_x, train_y)


# %%
# fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(25,25)) 

# idx = 22

# a = cv2.cvtColor(train_y[idx], cv2.COLOR_GRAY2BGR)
# for_plot = a * 255

# axes[0].imshow(train_x[idx][:,:,::-1])
# axes[1].imshow(for_plot)
# cv2.imshow('img', train_y[-1])
# cv2.waitKey(0)

# plt.show()


# %%
BACKBONE = 'resnet34'
preprocess_input = sm.get_preprocessing(BACKBONE)

train_x = preprocess_input(train_x)


# %%
# test_x = preprocess_input(test_x)


# %%
# base_model = VGG16(weights='imagenet', input_shape=(width, height, 3), include_top=False)

# base_out = base_model.output

# up = UpSampling2D(32, interpolation='bilinear')(base_out)
# conv = Conv2D(1, (1, 1))(up)
# conv = Activation('sigmoid')(conv)

# model = Model(input=base_model.input, output=conv)

model = sm.Unet(BACKBONE, encoder_weights='imagenet')


# %%

best_w = keras.callbacks.ModelCheckpoint('../models/fcn_best.h5', 
                                        monitor='val_loss',
                                        verbose=0,
                                        save_best_only=True,
                                        save_weights_only=True,
                                        mode='auto',
                                        period=1)

last_w = keras.callbacks.ModelCheckpoint('../models/fcn_last.h5', 
                                        monitor='val_loss',
                                        verbose=0,
                                        save_best_only=False,
                                        save_weights_only=True,
                                        mode='auto',
                                        period=1)

callbacks = [best_w, last_w]

adam = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999)

model.compile(optimizer=adam, 
              loss="binary_crossentropy", 
              metrics=["acc"])


# %%
def data_generator(x, y, batch_size):
    
    samples_per_epoch = x.shape[0]
    number_of_batches = samples_per_epoch/batch_size
    counter=0

    while 1:

        X_batch = np.array(x[batch_size*counter:batch_size*(counter+1)]).astype('float32')
        y_batch = np.array(y[batch_size*counter:batch_size*(counter+1)]).astype('float32')
        counter += 1
        yield X_batch,y_batch

        #restart counter to yeild data in the next epoch as well
        if counter >= number_of_batches:
            counter = 0


# %%
def split_data(x, y, val_percent):
    m = x.shape[0]
    val_chunk_size = int(m * val_percent)
    
#     shuffle_idxs = np.random.permutation(x.shape[0])
#     x = x[shuffle_idxs, :]
#     y = y[shuffle_idxs, :]
#     x_1, y_1 = shuffle(x[:m//2], y[:m//2])
#     x_2, y_2 = shuffle(x[m//2:], y[m//2:])
    
#     np.concatenate([x_1, x_2], out=x)
#     np.concatenate([y_1, y_2], out=y)
    train_x, train_y = x[:val_chunk_size], y[:val_chunk_size]
    val_x, val_y = x[val_chunk_size:], y[val_chunk_size:]
    
    return train_x, train_y, val_x, val_y


# %%
# history = model.fit(train_x, train_y, epochs=50, verbose=1, batch_size=1, callbacks=callbacks, validation_split=0.3)

batch_size = 1
samples_per_epoch = train_x.shape[0]

train_x, train_y, val_x, val_y = split_data(train_x, train_y, val_percent= 0.2)

history = model.fit_generator(data_generator(train_x, train_y, batch_size),
    epochs=30,
    steps_per_epoch = samples_per_epoch/batch_size,
    validation_data = data_generator(val_x, val_y, batch_size*5),
    validation_steps = samples_per_epoch/batch_size*5
)


# %%
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


# %%
model.evaluate(test_x, test_y, batch_size=1)


# %%
pred = model.predict(test_x, batch_size=1)


# %%
axes = plt.subplots(nrows=3, ncols=2, figsize=(25,25))[1]
for idx, axis in enumerate(axes):
    axis[0].imshow(test_x[idx][:,:,::-1])
    axis[1].imshow(pred[idx, ..., 0] > 0.5)

plt.show()


# %%
best_model = sm.Unet(BACKBONE)

best_model.load_weights("../models/fcn_best.h5")
best_model.compile(optimizer=adam, loss="binary_crossentropy", metrics=["acc"])


# %%
best_model.evaluate(test_x, test_y, batch_size=1)


# %%
pred = best_model.predict(test_x, batch_size=1)


# %%
axes = plt.subplots(nrows=3, ncols=2, figsize=(25,25))[1]
for idx, axis in enumerate(axes):
    axis[0].imshow(test_x[idx][:,:,::-1])
    axis[1].imshow(pred[idx, ..., 0] > 0.5)

plt.show()


# %%



