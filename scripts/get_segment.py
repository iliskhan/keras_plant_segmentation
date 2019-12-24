import os
import cv2
import torch
import NET_FCN

import numpy as np

liquid_segmentation_model = "../models/LiquidSolidPhaseRecognitionNetWeights.torch"

NUM_CLASSES = 4

def net_load(model_name):

    Net=NET_FCN.Net(NumClasses=NUM_CLASSES)
    Net.load_state_dict(torch.load(liquid_segmentation_model))
    Net.eval()
    Net.half()

    return Net

def get_max_area_contouts(img):
    label = np.array(img, dtype=np.uint8)

    _,thresh = cv2.threshold(label,127,255,0)
    contours,_ = cv2.findContours(thresh, 1, 2)
        
    cnt = sorted(contours, key=cv2.contourArea)[-1]

    return cnt

def get_liquid_pixels(img):

    Net = net_load(liquid_segmentation_model)

    x, y = img.shape[:2]

    original = img.copy()

    if x > 1000:

        new_x, new_y = int(x/2), int(y/2)

        img = cv2.resize(img, (new_y, new_x))

    img_for_net = np.reshape(img, (1, *img.shape))

    with torch.autograd.no_grad():
        _, Pred = Net.forward(img_for_net,EvalMode=True)  # Predict annotation using net
    LabelPred = Pred.data.cpu().numpy()

    LabelPred[LabelPred == 1] = 0
    LabelPred[LabelPred == 3] = 1
    LabelPred[LabelPred == 2] = 1

    label = LabelPred[0] * 255.

    cnt = get_max_area_contouts(label)

    mask = np.zeros(label.shape, np.uint8)

    cv2.fillPoly(mask, pts =[cnt], color=(255,255,255))

    img[mask == 0] = 0

    # temp = plant_masking(img)

    # cv2.imshow('masked', temp * 255.)

    mask = cv2.resize(mask, (y, x))


    original[mask == 0] = 0 

    return original

if __name__ == "__main__":
    img = cv2.imread('../data/test/imgs/photo_2019-12-20_18-03-14.jpg')

    img = get_liquid_pixels(img)

    cv2.imshow('img', img)
    cv2.waitKey(0)
