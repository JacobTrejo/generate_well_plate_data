import numpy as np
import imageio
from scipy import ndimage
from scipy.interpolate import griddata
# from AuxilaryFunctions import Circle
import warnings
import cv2 as cv
from skimage import draw
from programs.construct_model import f_x_to_model_bigger
import time
import scipy.signal as sig




def create_filled_out_mask_distance_1(in_arr):
    """
        function that will be used to create a continuous mask from a sparse mask
    """

    # normalizing
    og_mask = np.copy(in_arr)
    og_mask /= og_mask
    og_mask = np.nan_to_num(og_mask)

    # shifting to the right
    kernel = np.zeros((3, 3))
    kernel[1, 2] = 1
    right_shift = sig.convolve2d(og_mask, kernel, mode='same')

    # shifting to the left
    kernel_2 = np.zeros((3,3))
    kernel_2[1,0] = 1
    left_shift = sig.convolve2d(og_mask, kernel_2, mode='same')

    # shifting up
    kernel = np.zeros((3, 3))
    kernel[0, 1] = 1
    up_shift = sig.convolve2d(og_mask, kernel, mode='same')

    # shifting down
    kernel = np.zeros((3, 3))
    kernel[2, 1] = 1
    down_shift = sig.convolve2d(og_mask, kernel, mode='same')

    # filling out the points to the mask
    points_between_left_and_right = right_shift * left_shift
    points_between_up_and_down = up_shift * down_shift
    # NOTE: that we can then even multiply to get points that are even surrounded by pixels, this might be beneficial

    og_mask += points_between_left_and_right
    og_mask += points_between_up_and_down


    # normalizing
    og_mask /= og_mask
    og_mask = np.nan_to_num(og_mask)

    return og_mask

def create_filled_out_mask(in_arr, distance = 1):
    """
        function that will be used to create a continuous mask from a sparse mask
    """
    kernel_size = distance * 2 + 1
    kernel = np.zeros((kernel_size, kernel_size))

    # normalizing
    og_mask = np.copy(in_arr)
    # NOTE: dividing by zero when the count is 0
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        og_mask /= og_mask
    og_mask = np.nan_to_num(og_mask)

    # shifting to the right
    right_shift_kernel = np.zeros((kernel_size, kernel_size))
    right_shift_kernel[distance, distance + 1:] = 1
    right_shift = sig.convolve2d(og_mask, right_shift_kernel, mode='same')

    # shifting to the left
    left_shift_kernel = np.zeros((kernel_size, kernel_size))
    left_shift_kernel[distance, 0: distance] = 1
    left_shift = sig.convolve2d(og_mask, left_shift_kernel, mode='same')

    # shifting up
    up_shift_kernel = np.zeros((kernel_size, kernel_size))
    up_shift_kernel[0:distance, distance] = 1
    up_shift = sig.convolve2d(og_mask, up_shift_kernel, mode='same')

    # shifting down
    down_shift_kernel = np.zeros((kernel_size, kernel_size))
    down_shift_kernel[distance + 1:, distance] = 1
    down_shift = sig.convolve2d(og_mask, down_shift_kernel, mode='same')

    # filling out the points to the mask
    points_between_left_and_right = right_shift * left_shift
    points_between_up_and_down = up_shift * down_shift
    # NOTE: that we can then even multiply to get points that are even surrounded by pixels, this might be beneficial

    og_mask += points_between_left_and_right
    og_mask += points_between_up_and_down


    # NOTE: dividing by zero when the count is 0
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        og_mask /= og_mask
    og_mask = np.nan_to_num(og_mask)

    return og_mask

def fill_zeros_with_average(in_arr, size = 3):
    # copying to not modify original array
    arr = np.copy(in_arr)
    kernel = np.ones((size,size))

    # array that will be used to count how many in the vicinity are positive
    arr_4_count = np.copy(arr)
    arr_4_count[arr_4_count <= 0 ] = 0
    arr_4_count[arr_4_count > 0 ] = 1

    # masking negative values
    arr[arr < 0] = 0

    sum = sig.convolve2d(arr, kernel,mode='same')
    count = sig.convolve2d(arr_4_count, kernel, mode='same')

    # NOTE: dividing by zero when the count is 0
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        average = sum / count
    arr[arr == 0] = average[arr == 0]
    arr = np.nan_to_num(arr)
    return arr

def cylinder_distort(im, circle, top_circle, gamma_in_degrees, phi_in_degrees, in_pts = np.zeros((2,12)) ):
    pts = np.copy(in_pts)

    # Flags
    should_merge = True
    should_erase = True
    should_mask = True
    should_draw_circle = True

    output = np.zeros(im.shape)

    imageSizeY, imageSizeX = im.shape
    y, x = np.mgrid[0:imageSizeY, 0:imageSizeX]
    # circle = Circle(320,320,100)

    # TODO: create a new variable so that I do not have to create the
    # mesh grid all over again

    # NOTE: I added the next two lines
    x = x.astype(float)
    y = y.astype(float)
    x -= circle.centerX
    y -= circle.centerY

    # applying the transformation respectively to pts
    pts[0,:] -= circle.centerX
    pts[1,:] -= circle.centerY

    # flipping y for now to make visualizing easier
    y *= -1
    # applying the transformation respectively to pts
    pts[1,:] *= -1

    r = (x**2 + y**2)**.5
    # applying the transformation respectively to pts
    pts_r = (pts[0,:] ** 2 + pts[1,:] ** 2) ** .5

    length_of_cylinder_in_pixels = 150
    z = ((circle.radius - r) / circle.radius ) * length_of_cylinder_in_pixels
    # applying the transformation respectively to pts
    pts_z = ((circle.radius - pts_r) / circle.radius) * length_of_cylinder_in_pixels

    # NOTE: python throws a warning, but it is okay since numpy seems to take
    # care of it appropriately
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        angle = np.arctan(y / x)
        pts_angle = np.arctan(pts[1,:] / pts[0,:])

    # TODO: in the future make it only use radians to reduce the computations
    # converting it to degrees for now
    angle *= (180 / np.pi)
    # applying the transformation respectively to pts
    pts_angle *= (180/np.pi)

    # reducing the ambiguity of the angle
    angle[x < 0] += 180
    angle[(x > 0) * (y < 0)] = 360 + angle[(x > 0) * (y < 0)]
    # applying the transformation respectively to pts
    pts_angle[pts[0,:] < 0] += 180
    pts_angle[(pts[0,:] > 0) * (pts[1,:] < 0)] = 360 + pts_angle[ (pts[0,:] > 0) * (pts[1,:] < 0)]

    r = np.ones((angle.shape)) * circle.radius
    # applying the transformation respectively to pts
    pts_r = np.ones((pts_angle.shape)) * circle.radius

    # Transforming to radians
    angle *= (np.pi/ 180)
    # applying the transformation respectively to pts
    pts_angle *= (np.pi/180)

    mapped_y = r * np.sin(angle) + z * np.tan(gamma_in_degrees * (np.pi/180))
    # applying the transformation respectively to pts
    pts_mapped_y = pts_r * np.sin(pts_angle) + pts_z * np.tan(gamma_in_degrees * (np.pi/180))

    # Full motion
    mapped_x = r * np.cos(angle) + z * np.tan(  phi_in_degrees * (np.pi/180))
    # applying the transformation respectively to pts
    pts_mapped_x = pts_r * np.cos(pts_angle) + pts_z * np.tan(phi_in_degrees * (np.pi/180))

    # mapped_x = r * np.cos(angle)


    mapped_y *= -1
    mapped_y += circle.centerY
    mapped_x += circle.centerX
    # applying the transformation respectively to pts
    pts_mapped_y *= -1
    pts_mapped_x += circle.centerX
    pts_mapped_y += circle.centerY


    # TODO: the center has a nan value, the end program will probably not use the center
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        mapped_y = mapped_y.astype(int)
        mapped_x = mapped_x.astype(int)
        # applying the transformation respectively to pts
        pts_mapped_y = pts_mapped_y.astype(int)
        pts_mapped_x = pts_mapped_x.astype(int)

    # TODO: here is where x and y are computed again
    y, x = np.mgrid[0:imageSizeY, 0:imageSizeX]

    # mask for the points that are valid
    mask = (mapped_y >= 0) * (mapped_y < imageSizeY) * (mapped_x >= 0) * (mapped_x < imageSizeX)

    # # NOTE: for visualizing the points that are being feed to the interpolation
    output[mapped_y[mask] , mapped_x[mask]] = im[ y[mask], x[mask] ]

    output_mask = np.copy(output)
    output_mask[output_mask > 0 ] = 1
    x_indices = np.zeros(output.shape)
    x_indices[ mapped_y[mask] ,mapped_x[mask] ] = x[mask]
    x_indices = x_indices * output_mask
    # filling it because not all points are able to be captured
    if should_mask:
        x_mask = create_filled_out_mask(x_indices, distance=4)
    x_indices = fill_zeros_with_average(x_indices)
    if should_mask:
        x_indices *= x_mask


    y_indices = np.zeros(output.shape)
    y_indices[ mapped_y[mask], mapped_x[mask]] = y[mask]
    y_indices = y_indices * output_mask
    if should_mask:
        y_mask = create_filled_out_mask(y_indices, distance=4)
    y_indices = fill_zeros_with_average(y_indices)
    if should_mask:
        y_indices *= y_mask
    # x_indices[x_indices > 0] = 255
    # NOTE: quick merge with the original file
    # output += im
    mapped = ndimage.map_coordinates(im, [y_indices.ravel(), x_indices.ravel()], order=5)
    mapped.resize(im.shape)

    if should_erase and top_circle is not None:
        # erasing any objects that are outside the top cylinder
        top_circle_x = x - top_circle.centerX
        top_circle_y = y - top_circle.centerY
        r_with_respect_to_top_circle = \
            (top_circle_y ** 2 + top_circle_x ** 2) ** .5
        mapped[ r_with_respect_to_top_circle > top_circle.radius ] = 0
        # erasing any objects that are inside the bottom cylinder,
        # this area is only for the actual fish
        bottom_circle_x = x - circle.centerX
        bottom_circle_y = y - circle.centerY
        r_with_respect_to_bottom_circle = \
            (bottom_circle_x ** 2 + bottom_circle_y ** 2) ** .5
        mapped[ r_with_respect_to_bottom_circle < circle.radius ] = 0

        should_blur = True
        if should_blur:
            mapped = mapped.astype(float)
            mapped[ r_with_respect_to_bottom_circle > circle.radius] *= \
                (33 - (r_with_respect_to_bottom_circle[r_with_respect_to_bottom_circle > circle.radius] - circle.radius))/33

    mapped_pts = np.zeros((2, pts_mapped_x.shape[0]))
    mapped_pts[0,:] = pts_mapped_x
    mapped_pts[1,:] = pts_mapped_y
    return mapped, mapped_pts

# fishVect = np.load('fish.npy')
# fishVect[1] -= 290
# fishVect[2] -= 265
# fishVect[3:] = 0
# fishVect = [fishVect]
#
# seglen = fishVect[0][0]
# xVect = fishVect[0][1:]
# x = np.zeros((11,))
# xVect = np.array(xVect)
# x[:] = xVect[:]
# im, pts = f_x_to_model_bigger(x, seglen, 0, 300, 300)
#
# circle = Circle(150,150,100)
# gamma_in_degrees = 40
# phi_in_degrees = 0
# st = time.time()
# result = cylinder_distort(im, circle, None , gamma_in_degrees, phi_in_degrees)
# et = time.time()
# dur = et - st
# print('duration: ', dur)
# result += im
# length_of_cylinder_in_pixels = 150
# should_draw_circle = True
# if should_draw_circle:
#     rr, cc = draw.circle_perimeter(int(circle.centerY), int(circle.centerX), radius=circle.radius, shape=result.shape)
#     result[rr, cc] = 255
#
# top_circle = Circle(circle.centerX + length_of_cylinder_in_pixels * np.tan(  phi_in_degrees * (np.pi/180)),
#                     circle.centerY - length_of_cylinder_in_pixels * np.tan(gamma_in_degrees * (np.pi/180)),
#                     circle.radius )
# should_draw_top_circle = True
# if should_draw_top_circle:
#     rr, cc = draw.circle_perimeter(int(top_circle.centerY), int(top_circle.centerX), radius=top_circle.radius, shape=result.shape)
#     result[rr, cc] = 255
# cv.imwrite('test5.png', result)




