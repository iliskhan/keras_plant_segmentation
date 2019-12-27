import os
import cv2
import imutils

import numpy as np 
from get_segment import get_liquid_pixels
import segmentation_models as sm 

from time import time


def model_loader(model_name, path_to_weigth):
    model = sm.Unet(model_name)

    model.load_weights(path_to_weigth)

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["acc"])

    return model

def predict_mask(img, model):

    img_for_net = np.reshape(img, (1,*img.shape))
    
    pred = model.predict(img_for_net)
    mask = pred[0]
    mask[mask > 0.5] = 0
    mask[mask != 0] = 1

    return mask.astype(np.uint8)

def main():

    model_name = "resnet34"

    model = model_loader(model_name, "../models/fcn_best.h5")

    img = cv2.imread("img.jpg")

    start = time()
    liquid_img = get_liquid_pixels(img)
    plant_mask = predict_mask(img, model)
    print(time() - start)
    
    res = cv2.bitwise_or(liquid_img, liquid_img,  mask=plant_mask)

    res = imutils.resize(res, height=800)
    img = imutils.resize(img, height=800)
    cv2.imshow('liquid', img)
    cv2.imshow('img', res)
    cv2.waitKey(0)


main()