import numpy as np
import cv2 as cv
import imageio

im_name = '000000'
im_path = '../data/images/im_' + im_name + '.png'
ann_path = '../data/coor_2d/ann_' + im_name + '.npy'

coors = np.load(ann_path)
im = imageio.imread(im_path)
im = np.asarray(im)
im = np.stack([im for _ in range(3)], axis = 2)

red = [0,0,255]
green = [0,255,0]

coors = coors.astype(int)

im[coors[1,:10], coors[0,:10],:] = green
im[coors[1,10:], coors[0,10:],:] = red

cv.imwrite('test.png', im)

