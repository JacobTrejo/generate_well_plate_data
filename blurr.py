import numpy as np
import cv2 as cv
from PIL import Image, ImageFilter
import tqdm
from multiprocessing import Pool
import os

images_folder = 'data/images/'
outputs_folder = 'data/images/'
coor_folder = 'data/coor_2d/'

def blurr_image(frame_idx):
    
    strIdxInFormat = format(frame_idx, '06d')
    im_name = 'im_' + strIdxInFormat + '.png'
    im_path = images_folder + im_name
    OriImage = Image.open(im_path)

    gaussImage = OriImage.filter(ImageFilter.GaussianBlur( 1 + (np.random.rand() - .5) * .6  ))
    gaussImage = np.asarray( gaussImage )

    cv.imwrite( outputs_folder + im_name, gaussImage)



p = Pool()
r = list(tqdm.tqdm(p.imap(blurr_image, range(0,500000)), total=500000))



