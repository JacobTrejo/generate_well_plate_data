from programs.Config import Config
import random
import numpy as np
import cv2 as cv
import imageio
from scipy.io import loadmat
from programs.construct_model import f_x_to_model, f_x_to_model_centered, f_x_to_model_bigger
from skimage.util import random_noise
from PIL import ImageFilter
from PIL import Image
from skimage import draw

# Auxilary Functions
def roundHalfUp(a):
    """
    Function that rounds the way that matlab would. Necessary for the program to run like the matlab version
    :param a: numpy array or float
    :return: input rounded
    """
    return (np.floor(a) + np.round(a - (np.floor(a) - 1)) - 1)

def uint8(a):
    """
    This function is necessary to turn back arrays and floats into uint8.
    arr.astype(np.uint8) could be used, but it rounds differently than the
    matlab version.
    :param a: numpy array or float
    :return: numpy array or float as an uint8
    """

    a = roundHalfUp(a)
    if np.ndim(a) == 0:
        if a <0:
            a = 0
        if a > 255:
            a = 255
    else:
        a[a>255]=255
        a[a<0]=0
    return a

def imGaussNoise(image,mean,var):
    """
       Function used to make image have static noise

       Args:
           image (numpy array): image
           mean (float): mean
           var (numpy array): var

       Returns:
            noisy (numpy array): image with noise applied
       """
    row,col= image.shape
    sigma = var**0.5
    gauss = np.random.normal(mean,sigma,(row,col))
    gauss = gauss.reshape(row,col)
    print(np.min(gauss))
    noisy = image + gauss
    return noisy

def imGaussNoiseClipped(image,mean,var, maxValue = .5):
    """
       Function used to make image have static noise

       Args:
           image (numpy array): image
           mean (float): mean
           var (numpy array): var

       Returns:
            noisy (numpy array): image with noise applied
       """
    row,col= image.shape
    sigma = var**0.5
    gauss = np.random.normal(mean,sigma,(row,col))
    gauss = gauss.reshape(row,col)
    gauss = np.clip(gauss, 0, maxValue)
    noisy = image + gauss
    return noisy

def get_valid_points_mask(pts, img_shape):
    imSizeY, imSizeX = img_shape[:2]
    valid_x = (np.floor(pts[0,:]) >= 0) * (np.ceil(pts[0,:]) < imSizeX)
    valid_y = (np.floor(pts[1,:]) >= 0) * (np.ceil(pts[1,:]) < imSizeY)
    valid = valid_y * valid_x
    return valid

# Rotate along x axis. Angles are accepted in radians
def rotx(angle):
    M = np.array([[1, 0, 0],  [0, np.cos(angle), -np.sin(angle)],  [0, np.sin(angle), np.cos(angle)]])
    return M

# Rotate along y axis. Angles are accepted in radians
def roty(angle):
    M = np.array([[np.cos(angle), 0, np.sin(angle)],  [0, 1, 0],  [-np.sin(angle), 0, np.cos(angle)]])
    return M

# Rotate along z axis. Angles are accepted in radians
def rotz(angle):
    M = np.array([[np.cos(angle), -np.sin(angle), 0],  [np.sin(angle), np.cos(angle), 0],  [0, 0, 1]])
    return M

def x_seglen_to_3d_points(x, seglen):
    """
        Function that turns the x vector into the 3d points of the fish
    :param x:
    :param seglen:
    :return:
    """
    hp = x[0: 2]
    dt = x[2: 11]
    pt = np.zeros((2, 10))
    theta = np.zeros((9, 1))
    theta[0] = dt[0]
    pt[:, 0] = hp

    for n in range(0, 9):
        R = np.array([[np.cos(dt[n]), -np.sin(dt[n])], [np.sin(dt[n]), np.cos(dt[n])]])
        if n == 0:
            vec = np.matmul(R, np.array([seglen, 0], dtype=R.dtype))
        else:
            vec = np.matmul(R, vec)
            theta[n] = theta[n - 1] + dt[n]
        pt[:, n + 1] = pt[:, n] + vec

    # Now calculating the eyes
    size_lut = 49
    size_half = (size_lut + 1) / 2

    dh1 = pt[0, 1] - np.floor(pt[0, 1])
    dh2 = pt[1, 1] - np.floor(pt[1, 1])

    d_eye = seglen

    XX = size_lut
    YY = size_lut
    ZZ = size_lut

    c_eyes = 1.9
    c_belly = 0.98
    c_head = 1.04
    canvas = np.zeros((XX, YY, ZZ))

    theta, gamma, phi = x[2], 0, 0

    R = rotz(theta) @ roty(phi) @ rotx(gamma)

    # Initialize points of the ball and stick model in the canvas
    pt_original = np.zeros((3, 3))
    # pt_original_1 is the mid-point in Python's indexing format
    pt_original[:, 1] = np.array([np.floor(XX / 2) + dh1, np.floor(YY / 2) + dh2, np.floor(ZZ / 2)])
    pt_original[:, 0] = pt_original[:, 1] - np.array([seglen, 0, 0], dtype=pt_original.dtype)
    pt_original[:, 2] = pt_original[:, 1] + np.array([seglen, 0, 0], dtype=pt_original.dtype)

    eye1_c = np.array([[c_eyes * pt_original[0, 0] + (1 - c_eyes) * pt_original[0, 1]],
                       [c_eyes * pt_original[1, 0] + (1 - c_eyes) * pt_original[1, 1] + d_eye / 2],
                       [pt_original[2, 1] - seglen / 8]], dtype=pt_original.dtype)
    eye1_c = eye1_c - pt_original[:, 1, None]
    eye1_c = np.matmul(R, eye1_c) + pt_original[:, 1, None]

    eye2_c = np.array([[c_eyes * pt_original[0, 0] + (1 - c_eyes) * pt_original[0, 1]],
                       [c_eyes * pt_original[1, 0] + (1 - c_eyes) * pt_original[1, 1] - d_eye / 2],
                       [pt_original[2, 1] - seglen / 8]], dtype=pt_original.dtype)
    eye2_c = eye2_c - pt_original[:, 1, None]
    eye2_c = np.matmul(R, eye2_c) + pt_original[:, 1, None]

    eye1_c[0] = eye1_c[0] - (size_half - 1) + pt[0, 1]
    eye1_c[1] = eye1_c[1] - (size_half - 1) + pt[1, 1]
    eye2_c[0] = eye2_c[0] - (size_half - 1) + pt[0, 1]
    eye2_c[1] = eye2_c[1] - (size_half - 1) + pt[1, 1]

    pt = np.concatenate([pt, eye1_c[0: 2], eye2_c[0: 2]], axis=1)

    return pt

def addBoxes(pt, padding= .7):
    """
        Function that adds boxes to the fish given its backbone points
    :param pt:
    :param padding:
    :return:
    """
    number_of_eyes = 2
    threshold = 3
    # minus one assuming the boxes are in the center
    amount_of_boxes = pt.shape[1] - number_of_eyes - 1

    dimensions_of_center = 2
    dimensions_of_box = 2
    pointsAndBoxes = np.zeros(( dimensions_of_box + dimensions_of_center, amount_of_boxes))
    # TODO:vectorize
    for pointIdx in range(amount_of_boxes):
        fPX, fPY = pt[:, pointIdx]
        sPX, sPY = pt[:, pointIdx + 1]

        c = np.array( [(fPX + sPX)/ 2, (fPY + sPY)/2] )
        l = np.array( [np.abs(fPX - sPX), np.abs(fPY - sPY)] )

        # Even if is close to zero there should still be some length/width to the box
        l[l < threshold] = threshold

        # Adding the padding
        l = (padding + 1) * l

        cX, cY = c
        lX, lY = l

        pointsAndBoxes[:, pointIdx] = [cX, cY, lX, lY]

    # Now the eyes
    for eyeIdx in range(number_of_eyes):
        lenghtOfEyes = 9 # ?, maybe make it depend on seglen
        cX, cY = pt[:, -number_of_eyes + eyeIdx]
        lX, lY = lenghtOfEyes, lenghtOfEyes

        box_for_eyes = np.array([ [ cX], [cY], [lX], [lY]])
        pointsAndBoxes = np.concatenate( (pointsAndBoxes, box_for_eyes), axis = 1)
    return pointsAndBoxes

def draw_circle(circle, canvas, color = None):
    rr, cc = draw.circle_perimeter(int(circle.centerY), int(circle.centerX), radius=circle.radius, shape=canvas.shape)
    if color is None:
        canvas[rr, cc] = 255
    else:
        canvas[rr, cc, :] = color


    return canvas

def crop_image(left_shift, right_shift, top_shift, bottom_shift, im):

    frame_size_y, frame_size_x = im.shape[:2]

    newFrameX = frame_size_x + left_shift + right_shift
    newFrameY = frame_size_y + top_shift + bottom_shift

    new_im = np.zeros((newFrameY, newFrameX))

    if left_shift >= 0:
        smallX = 0
    else:
        smallX = -1 * left_shift
    if right_shift >= 0:
        bigX = im.shape[1]
    else:
        bigX = right_shift

    if top_shift >= 0:
        smallY = 0
    else:
        smallY = -1 * top_shift
    if bottom_shift >= 0:
        bigY = im.shape[0]
    else:
        bigY = bottom_shift

    cropped_im = im[smallY:bigY, smallX:bigX]

    # Now getting the indices for the canvas the cropped will be place on
    bigY, bigX = cropped_im.shape
    if left_shift >= 0:
        smallX = left_shift
    else:
        smallX = 0
    if top_shift >= 0:
        smallY = top_shift
    else:
        smallY = 0

    bigX += smallX
    bigY += smallY

    new_im[smallY:bigY, smallX:bigX] = cropped_im

    return new_im

def resize(im, fish):
    # resizing to get the same size as the fish in the videos
    should_resize = True
    if should_resize:
        ogFrameX = 1966 + np.random.randint(-20, 20)
        ogFrameY = 1480 + np.random.randint(-20, 20)
        xRatio = 960 / ogFrameX
        yRatio = 720 / ogFrameY

        new_width = int(np.ceil(im.shape[1] * xRatio))
        new_height = int(np.ceil(im.shape[0] * yRatio))
        # updating the ratio to get the right accuracy
        xRatio = new_width / im.shape[1]
        yRatio = new_height / im.shape[0]
        # updating the points
        pts = fish.pts
        pts[0, :] *= xRatio
        pts[1, :] *= yRatio
        fish.pts = pts
        # updating the bounding box
        boundingBox = fish.boundingBox
        boundingBox.smallX = boundingBox.smallX * xRatio
        boundingBox.bigX = boundingBox.bigX * xRatio
        boundingBox.smallY = boundingBox.smallY * yRatio
        boundingBox.bigY = boundingBox.bigY * yRatio
        fish.boundingBox = boundingBox

        new_dimension = (new_width, new_height)
        im = cv.resize(im, new_dimension, interpolation=cv.INTER_LINEAR)

    return im, fish

def randomly_crop(im, fish):
    amount_off_top, amount_off_bottom, amount_off_left, amount_off_right = 0, 0, 0, 0
    # cropping it randomly
    height, width = im.shape[:2]
    if height > 101:
        amount_to_cut_off = height - 101
        amount_off_top = np.random.randint(0, amount_to_cut_off + 1)
        amount_off_top = np.clip(amount_off_top, 0, int(amount_to_cut_off / 2))
        amount_off_bottom = amount_to_cut_off - amount_off_top
        amount_off_bottom *= -1
        amount_off_top *= -1
    if width > 101:
        amount_to_cut_off = width - 101
        amount_off_left = np.random.randint(0, amount_to_cut_off + 1)
        amount_off_left = np.clip(amount_off_left, 0, int(amount_to_cut_off / 2))
        amount_off_right = amount_to_cut_off - amount_off_left
        amount_off_right *= -1
        amount_off_left *= -1

    im = crop_image(amount_off_left, amount_off_right, amount_off_top, amount_off_bottom, im)
    # updating the points
    pts = fish.pts
    pts[0, :] += amount_off_left
    pts[1, :] += amount_off_top
    fish.pts = pts
    # # Updating the bounding box
    boundingBox = fish.boundingBox
    boundingBox.smallX += amount_off_left
    boundingBox.bigX += amount_off_left
    boundingBox.smallY += amount_off_top
    boundingBox.bigY += amount_off_top
    fish.boundingBox = boundingBox

    return im, fish

def add_blur(im):
    # adding blur
    OriImage = Image.fromarray(im.astype(np.uint8))
    gaussImage2 = OriImage.filter(ImageFilter.GaussianBlur(1 + ((np.random.rand() - .5) * .2)))
    im = np.asarray(gaussImage2).astype(float)
    return im

def add_background_noise(im):
    # adding noise
    filter_size = 2 * roundHalfUp(np.random.rand()) + 3
    sigma = np.random.rand() + 0.5
    kernel = cv.getGaussianKernel(int(filter_size), sigma)
    im = cv.filter2D(im, -1, kernel)
    maxGray = max(im.flatten())
    if maxGray != 0:
        # im = im / max(im.flatten())
        im /= 255
    else:
        im[0, 0] = 1
    im = imGaussNoise(im, (np.random.rand() * np.random.normal(50, 10)) / 255,
                      (np.random.rand() * 50 + 20) / 255 ** 2)
    # Converting Back
    if maxGray != 0:
        # im = im * (255 / max(im.flatten()))
        im *= 255
        # im *= maxGray
    else:
        im[0, 0] = 0
        im = im * 255

    return im


def add_background_noise_flipped(im):
    # adding noise
    maxGray = max(im.flatten())
    if maxGray != 0:
        # im = im / max(im.flatten())
        im /= 255
    else:
        im[0, 0] = 1
    im = imGaussNoiseClipped(im, (np.random.rand() * np.random.normal(50, 10)) / 255,
                      (np.random.rand() * 50 + 20) / 255 ** 2)
    # Converting Back
    if maxGray != 0:
        # im = im * (255 / max(im.flatten()))
        im *= 255
        # im *= maxGray
    else:
        im[0, 0] = 0
        im = im * 255
    
    filter_size = 2 * roundHalfUp(np.random.rand()) + 3
    sigma = np.random.rand() + 0.5
    kernel = cv.getGaussianKernel(int(filter_size), sigma)
    im = cv.filter2D(im, -1, kernel)

    return im

def applyGaussianKernel(im):
    filter_size = 2 * roundHalfUp(np.random.rand()) + 3
    sigma = np.random.rand() + 0.5
    kernel = cv.getGaussianKernel(int(filter_size), sigma)
    im = cv.filter2D(im, -1, kernel)

    return im



def add_patchy_noise(im, fish):
    frameSizeY, frameSizeX = im.shape[:2]

    # bounding box noise
    pvar = np.random.poisson(Config.averageAmountOfPatchyNoise)
    if (pvar > 0):
        for i in range(1, int(np.floor(pvar + 1))):
            # No really necessary, but just to ensure we do not lose too many
            # patches to fishes barely visible or fishes that do not appear in the view

            amountOfFishes = 1
            # idxListOfPatchebleFishes = [idx for idx, fish in enumerate(fishList) if np.random.rand() > .8]
            idxListOfPatchebleFishes = [0]

            # idxListOfPatchebleFishes = [idx for idx, fish in enumerate(fishVectList + overlappingFishVectList) if fish.is_valid_fish]
            amountOfPossibleCenters = len(idxListOfPatchebleFishes)
            finalVar_mat = np.zeros((frameSizeY, frameSizeX))
            amountOfCenters = np.random.randint(1, high=(amountOfPossibleCenters + 1))
            for centerIdx in range(amountOfCenters):
                # y, x
                center = np.zeros((2))
                shouldItGoOnAFish = True if np.random.rand() > .5 else False
                if shouldItGoOnAFish:

                    # fish = (fishVectList + overlappingFishVectList)[ idxListOfPatchebleFishes[centerIdx] ]

                    boundingBox = fish.boundingBox

                    # boundingBox = boundingBoxList[idxListOfPatchebleFishes[centerIdx]]

                    center[0] = (boundingBox.getHeight() * (np.random.rand() - .5)) + boundingBox.getCenterY()
                    center[1] = (boundingBox.getWidth() * (np.random.rand() - .5)) + boundingBox.getCenterX()
                    center = center.astype(int)
                    # clip just in case we went slightly out of bounds
                    center[0] = np.clip(center[0], 0, frameSizeY - 1)
                    center[1] = np.clip(center[1], 0, frameSizeX - 1)
                else:
                    center[0] = np.random.randint(0, high=frameSizeY)
                    center[1] = np.random.randint(0, high=frameSizeX)

                zeros_mat = np.zeros((frameSizeY, frameSizeX))
                zeros_mat[int(center[0]) - 1, int(center[1]) - 1] = 1
                randi = (2 * np.random.randint(5, high=35)) + 1
                se = cv.getStructuringElement(cv.MORPH_ELLIPSE, (randi, randi))
                zeros_mat = cv.dilate(zeros_mat, se)
                finalVar_mat += zeros_mat

            maxG = max(im.flatten())
            im = im / maxG
            # gray_b = imnoise(gray_b, 'localvar', var_mat * 3 * (np.random.rand() * 60 + 20) / 255 ** 2)
            im = random_noise(im, mode='localvar', local_vars=(finalVar_mat * 3 * (
                    np.random.rand() * 60 + 20) / 255 ** 2) + .00000000000000001)
            # im = im * (maxG / max(im.flatten()))
            # np.clip(im, 0, 1)
            im = im * (255 / max(im.flatten()))

    return im

# class Fish:
#     class BoundingBox:
#         BoundingBoxThreshold = 2
#         def __init__(self, smallY, bigY, smallX, bigX):
#             self.smallY = smallY
#             self.bigY = bigY
#             self.smallX = smallX
#             self.bigX = bigX
#
#         def getHeight(self):
#             return ( self.bigY - self.smallY)
#
#         def getWidth(self):
#             return ( self.bigX - self.smallX)
#
#         def getCenterX(self):
#             return (( self.bigX + self.smallX) / 2)
#
#         def getCenterY(self):
#             return (( self.bigY + self.smallY) / 2)
#
#         def isValidBox(self):
#             height = self.getHeight()
#             width = self.getWidth()
#
#             if (height <= Fish.BoundingBox.BoundingBoxThreshold) or (width <= Fish.BoundingBox.BoundingBoxThreshold):
#                 return False
#             else:
#                 return True
#
#     def __init__(self, pts, graymodel):
#         self.pts = pts
#         self.graymodel = graymodel
#         self.vis = np.ones((pts.shape[1]))
#         # self.vis = np.zeros((pts.shape[1]))
#         # self.vis[ self.valid_points_masks ] = 1
#
#         # Creating the bounding box
#         nonzero_coors = np.array( np.where(graymodel > 0) )
#         try:
#             smallY = np.min(nonzero_coors[0,:])
#             bigY = np.max( nonzero_coors[0,:])
#             smallX = np.min( nonzero_coors[1,:])
#             bigX = np.max( nonzero_coors[1,:])
#         except:
#             smallY = 0
#             bigY = 0
#             smallX = 0
#             bigX = 0
#         self.boundingBox = Fish.BoundingBox(smallY, bigY, smallX, bigX)
#     @property
#     def xs(self):
#         return self.pts[0,:]
#     @property
#     def ys(self):
#         return self.pts[1,:]
#     @property
#     def intXs(self):
#         return np.ceil( self.pts[0,:] ).astype(int)
#     @property
#     def intYs(self):
#         return np.ceil( self.pts[1,:]).astype(int)
#     # @property
#     # def valid_points_masks(self):
#     #     xs = self.intXs
#     #     ys = self.intYs
#     #     xs_in_bounds = (xs < imageSizeX) * (xs >= 0)
#     #     ys_in_bounds = (ys < imageSizeY) * (ys >= 0)
#     #     return xs_in_bounds * ys_in_bounds
#
#     # def amount_of_vis_points(self):
#     #     val_xs = self.pts[0,:][self.valid_points_masks]
#     #     return val_xs.shape[0]
#
#     @property
#     def is_valid_fish(self):
#         # if (self.amount_of_vis_points() >= 1) and self.boundingBox.isValidBox():
#         #     return True
#         # else:
#         #     return False
#         return True

def place_cropped_image_to_canvas(image, canvas, crops):
    """
        function that places a small cropped image onto a bigger image, the canvas, based on the crops
    :param image: numpy array representing the small image
    :param canvas: numpy array representing the canvas
    :param crops: numpy array of the crops
    :return: numpy array of the canvas with the cropped image place inside it
    """

    imageSizeY, imageSizeX = canvas.shape
    imageSizeYForSmallerPicture, imageSizeXForSmallerPicture = image.shape

    crops = crops.astype(int)

    smallY, bigY, smallX, bigX = crops
    canvas_indices = np.copy(crops)
    gray_indices = np.array([0, imageSizeYForSmallerPicture - 1, 0, imageSizeXForSmallerPicture - 1])

    # Adjusting if gray_b is horizontally out of bounds
    isGrayBCompletelyOutOfBounds = False
    if smallX < 0:
        # Atleast some part of the image is out of bounds on the left
        if bigX < 0:
            isGrayBCompletelyOutOfBounds = True
        else:
            canvas_indices[2] = 0
            lenghtOfImageNotShowing = -smallX
            gray_indices[2] = lenghtOfImageNotShowing
    if bigX > imageSizeX - 1:
        # At least some part of the image is out of bounds on the right
        if smallX > imageSizeX - 1:
            isGrayBCompletelyOutOfBounds = True
        else:
            canvas_indices[3] = imageSizeX - 1
            lenghtOfImageShowing = imageSizeX - smallX
            gray_indices[3] = lenghtOfImageShowing - 1
    # Adjusting if gray_b is vertically out of bounds
    if smallY < 0:
        # Atleast some part of the image is out of bounds on the left
        if bigY < 0:
            isGrayBCompletelyOutOfBounds = True
        else:
            canvas_indices[0] = 0
            lenghtOfImageNotShowing = -smallY
            gray_indices[0] = lenghtOfImageNotShowing
    if bigY > imageSizeY - 1:
        # At least some part of the image is out of bounds on the right
        if smallY > imageSizeY - 1:
            isGrayBCompletelyOutOfBounds = True
        else:
            canvas_indices[1] = imageSizeY - 1
            lenghtOfImageShowing = imageSizeY - smallY
            gray_indices[1] = lenghtOfImageShowing - 1

    if not isGrayBCompletelyOutOfBounds:
        canvas[canvas_indices[0]: canvas_indices[1] + 1, canvas_indices[2]: canvas_indices[3] + 1] = \
            image[gray_indices[0]: gray_indices[1] + 1, gray_indices[2]: gray_indices[3] + 1]
    # canvas = canvas.astype(np.uint8)

    return canvas


# class Circle:
#     def __init__(self, centerX, centerY, radius):
#         self.radius = radius
#         self.centerX = centerX
#         self.centerY = centerY
#
#     # For Visualization Purposes
#     @property
#     def centerXInt(self):
#         return int(self.centerX)
#     def centerYInt(self):
#         return int(self.centerY)















