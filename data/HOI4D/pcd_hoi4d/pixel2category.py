from .utils import category2label_map
from .palette import pal_color_map
import cv2
import numpy as np
import os


def get_mask_and_label(path):
    '''/mnt/sas-raid5-7.2T/HOI4D_data/datatang/ZY20210800001/H1/C1/N15/S99/s4/T1/2Dseg/mask'''
    l = path.split('/')
    for i in l:
        if 'C' in i:
            category = i
    #for img in os.listdir(path):
        #f(img, category)
    return f(path, category)


def f(img_path, category):
    image = cv2.imread(img_path)[:, :, ::-1]
    assert image.shape == (1080, 1920, 3)
    # image = shift_mask(image, img_path)
    color_map = pal_color_map()

    arrs = []
    labels_motion = []  # motion segmentation
    for i in range(10):
        color = color_map[i + 1]
        valid = (image[..., 0] == color[0]) & (image[..., 1] == color[1]) & (image[..., 2] == color[2])
        if np.sum(valid) == 0:
            continue
        if i >= len(category2label_map[category]):
            continue
        arrs.append(valid)
        labels_motion.append(i + 1)
    
    # arrs: LIST( ndarray[] ), 其中满足条件的像素点为1, 否则为0 <motion segmentation>
    return arrs, np.array(labels_motion)

    

if __name__ == "__main__":
    pass
