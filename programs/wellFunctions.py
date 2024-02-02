from programs.IntrinsicParameters import IntrinsicParameters
import numpy as np
from skimage import draw
import cv2 as cv
from scipy.io import loadmat
from programs.AuxilaryFunctions import x_seglen_to_3d_points
from programs.construct_model import f_x_to_model_bigger
from scipy import ndimage
import math
from skimage.transform import swirl
from programs.AuxilaryFunctions import roundHalfUp
from programs.AuxilaryFunctions import imGaussNoise
from programs.AuxilaryFunctions import uint8
from skimage.util import random_noise

# Inputs
imageSizeY, imageSizeX = 640, 640
theta_array = loadmat('inputs/generated_pose_all_2D_50k.mat')
theta_array = theta_array['generated_pose']

def points_in_bounds(pts, im_shape):
    h, w = im_shape[:2]
    # subtract by 1 for pythons format
    valid_x = (np.floor(pts[0,:]) >= 0) * (np.ceil(pts[0,:]) < w)
    valid_y = (np.floor(pts[1,:]) >= 0) * (np.ceil(pts[1,:]) < h)
    return valid_x * valid_y

def rotz(angle, vector):
    """
        rotation on the xy plane
    :param angle: float in radians
    :param vector: numpy vector of dim 2
    :return:
    """
    R = np.matrix([[math.cos(angle), -math.sin(angle)],
                   [math.sin(angle), math.cos(angle)]])
    # assuming it is a numpy array
    vector = np.matrix([[vector[0]],
                        [vector[1]]])
    result = R @ vector
    return np.squeeze( np.array( result ) )

def rotz_vector(angle, vector):
    """
        vectorized version of the above function which
        rotates a whole vector
    :param angle:
    :param vector:
    :return:
    """
    R = np.matrix([[math.cos(angle), -math.sin(angle)],
                   [math.sin(angle), math.cos(angle)]])
    result = np.matmul(R, vector)
    return np.array(result)

def distance_from_center_function(pts, circle):
    # copying to not modify the pts
    distance = np.copy(pts)
    distance[0, :] -= circle.centerX
    distance[1, :] -= circle.centerY

    distance = (distance[0, :] ** 2 + distance[1, :] ** 2) ** .5
    return distance

def are_pts_in_circle(pts, circle):
    # # copying to not modify the pts
    # distance = np.copy(pts)
    # distance[0,:] -= circle.centerX
    # distance[1,:] -= circle.centerY
    #
    # distance = (distance[0,:] ** 2 + distance[1,:] **2 ) ** .5

    distance = distance_from_center_function(pts, circle)
    if np.any( distance >= circle.radius ):
        return False
    else:
        return True

def rotate_coordinates_centered(rotation_angle, points, im_shape):
    cy, cx = im_shape[0] / 2, im_shape[1] / 2
    # centering
    points[0, :] -= cx
    points[1, :] -= cy
    # rotating
    points = rotz_vector(rotation_angle * (math.pi / 180), points)
    # sending them back
    points[0, :] += cx
    points[1, :] += cy

    return points

def generate_a_fish_in_circle(circle, distance_from_edge = None, return_angle = False):
    while True:

        # Generating a fish
        xVect = np.zeros((11))
        fishlen = (np.random.rand(1) - 0.5) * 30 + 70
        idxlen = np.floor((fishlen - 62) / 1.05) + 1
        # seglen = 5.6 + idxlen * 0.1
        #
        # seglen = seglen[0]
        # seglen = 2 + np.random.rand()
        seglen = IntrinsicParameters.seglen_distribution()
        # seglen = 7.1

        # Changing the distribution where the center can be to reduce looping
        # x, y = np.random.randint(0, imageSizeX), np.random.randint(0, imageSizeY)
        radius = circle.radius
        if distance_from_edge is None:
            # let it be in any part of the well plate

            #x = np.random.randint( circle.centerX - radius, circle.centerX + radius )
            #y = np.random.randint( circle.centerY - radius, circle.centerY + radius )

            chosen_radius = radius * np.random.rand()
            chosen_angle = np.pi * 2 * np.random.rand()
            x = np.cos(chosen_angle) * chosen_radius
            y = np.sin(chosen_angle) * chosen_radius
            x += circle.centerX
            y += circle.centerY

        else:
            # let it be constrained to being around the edge

            # how far way the first keypoint will be from the edge
            edge_offset = np.random.rand() * distance_from_edge
            chosen_radius = radius - edge_offset
            chosen_angle = np.random.rand() * 2 * np.pi
            x = chosen_radius * np.cos(chosen_angle)
            y = chosen_radius * np.sin(chosen_angle)
            x += circle.centerX
            y += circle.centerY


        theta_array_idx = np.random.randint(0, 500000)
        dtheta = theta_array[theta_array_idx, :]
        xVect[:2] = [x, y]
        xVect[2] = np.random.rand(1)[0] * 2 * np.pi
        xVect[3:] = dtheta
        fishVect = np.zeros((12))
        fishVect[0] = seglen
        fishVect[1:] = xVect

        # Checking if it is in bounds
        pts = x_seglen_to_3d_points(xVect, seglen)
        if are_pts_in_circle(pts, circle):

            if distance_from_edge is not None:
                # We got an extra criteria to check
                # We are checking that the fish is away from the center a certain
                # distance so that we can have reflections
                distance = distance_from_center_function(pts, circle)
                keypoints_distance_from_edge = circle.radius - distance
                if np.all( keypoints_distance_from_edge > distance_from_edge ):
                    continue
            if return_angle:
                return fishVect, chosen_angle
            else:
                return fishVect




