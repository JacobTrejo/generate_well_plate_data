from programs.Config import Config
import numpy as np
import cv2 as cv
import torch
from programs.AuxilaryFunctions import draw_circle
from programs.wellFunctions import generate_a_fish_in_circle
from programs.construct_model import f_x_to_model_bigger
from programs.fasterReflection2 import cylinder_distort
from programs.AuxilaryFunctions import get_valid_points_mask
from programs.AuxilaryFunctions import resize, randomly_crop, add_blur, crop_image, add_background_noise, add_background_noise_flipped, add_patchy_noise, applyGaussianKernel

class Aquarium:

    def generateRandomConfiguration(self):
        self.in_radius = Config.average_radius
        self.in_radius += np.random.randint(-Config.max_radius_offset, Config.max_radius_offset + 1)
        self.in_side_padding = Config.average_side_padding
        self.in_side_padding += np.random.randint(-Config.max_side_padding_offset, Config.max_side_padding_offset + 1)
        if self.in_side_padding < 0: self.in_side_padding = 0

        in_well_plate_side_lenght = self.in_radius * 2 + self.in_side_padding * 2
        self.smallCanvas = np.zeros((in_well_plate_side_lenght, in_well_plate_side_lenght))
        self.frameSizeY, self.frameSizeX = self.smallCanvas.shape
        self.circle = Circle(self.frameSizeX / 2, self.frameSizeY / 2, self.in_radius)

        # st = time.time()
        should_put_fish_near_edge = True if np.random.rand() < Config.chance_of_putting_fish_near_edge else False
        self.should_put_fish_near_edge = should_put_fish_near_edge
        if should_put_fish_near_edge:
            fishVect, chosen_angle = generate_a_fish_in_circle(self.circle, Config.maximum_distance_from_edge, return_angle=True)
        else:
            fishVect, chosen_angle = generate_a_fish_in_circle(self.circle, None, return_angle=True)

        should_angle_vary = 1 if np.random.rand() < Config.chance_that_the_angle_should_vary else 0
        chosen_angle += (np.random.randint(-Config.max_angle_offset, Config.max_angle_offset) * (np.pi / 180) * should_angle_vary)

        temp_radius = (np.random.rand() * Config.max_additional_distance_from_bottom_circle +
                       Config.minimum_distance_from_bottom_circle) + self.circle.radius

        top_circle_center_x = np.cos(chosen_angle) * temp_radius
        top_circle_center_y = np.sin(chosen_angle) * temp_radius
        top_circle_center_x += self.circle.centerX
        top_circle_center_y += self.circle.centerY
        self.top_circle = Circle(top_circle_center_x, top_circle_center_y, self.circle.radius)

        return fishVect

    def __init__(self, frame_idx):
        self.frame_idx = frame_idx
        self.fishList = []
        fishVect = self.generateRandomConfiguration()
        fish = Fish(fishVect, self.frameSizeY, self.frameSizeX, self.should_put_fish_near_edge,
                    self.circle, self.top_circle)
        self.fishList.append(fish)

        should_draw_empty_canvas = \
            True if np.random.rand() < Config.fraction_of_data_that_should_be_background else False
        if should_draw_empty_canvas: self.fishList = []


    def resize(self):
        should_resize = True
        if should_resize:
            ogFrameX = Config.original_length
            ogFrameX += np.random.randint(-Config.max_length_height_offset, Config.max_length_height_offset + 1)
            ogFrameY = Config.original_height
            ogFrameY += np.random.randint(-Config.max_length_height_offset, Config.max_length_height_offset + 1)
            xRatio = Config.shrunken_length / ogFrameX
            yRatio = Config.shrunken_height / ogFrameY

            new_width = int(np.ceil(self.smallCanvas.shape[1] * xRatio))
            new_height = int(np.ceil(self.smallCanvas.shape[0] * yRatio))
            # updating the ratio to get the right accuracy
            xRatio = new_width / self.smallCanvas.shape[1]
            yRatio = new_height / self.smallCanvas.shape[0]

            new_dimension = (new_width, new_height)
            self.smallCanvas = cv.resize(self.smallCanvas, new_dimension, interpolation=cv.INTER_LINEAR)

            # updating the points
            if len(self.fishList) != 0:
                tempFish = self.fishList[0]
                pts = tempFish.pts
                pts[0, :] *= xRatio
                pts[1, :] *= yRatio
                tempFish.pts = pts
                # updating the bounding box
                boundingBox = tempFish.boundingBox
                boundingBox.smallX = boundingBox.smallX * xRatio
                boundingBox.bigX = boundingBox.bigX * xRatio
                boundingBox.smallY = boundingBox.smallY * yRatio
                boundingBox.bigY = boundingBox.bigY * yRatio
                tempFish.boundingBox = boundingBox
                self.fishList[0] = tempFish

    def randomly_crop(self):
        amount_off_top, amount_off_bottom, amount_off_left, amount_off_right = 0, 0, 0, 0
        # cropping it randomly
        height, width = self.smallCanvas.shape[:2]
        if height > Config.max_height:
            amount_to_cut_off = height - Config.max_height
            amount_off_top = np.random.randint(0, amount_to_cut_off + 1)
            amount_off_top = np.clip(amount_off_top, 0, int(amount_to_cut_off / 2))
            amount_off_bottom = amount_to_cut_off - amount_off_top
            amount_off_bottom *= -1
            amount_off_top *= -1
        if width > Config.max_length:
            amount_to_cut_off = width - Config.max_length
            amount_off_left = np.random.randint(0, amount_to_cut_off + 1)
            amount_off_left = np.clip(amount_off_left, 0, int(amount_to_cut_off / 2))
            amount_off_right = amount_to_cut_off - amount_off_left
            amount_off_right *= -1
            amount_off_left *= -1

        self.smallCanvas = crop_image(amount_off_left, amount_off_right, amount_off_top, amount_off_bottom, self.smallCanvas)

        # updating the points
        if len(self.fishList) != 0:
            tempFish = self.fishList[0]
            pts = tempFish.pts
            pts[0, :] += amount_off_left
            pts[1, :] += amount_off_top
            tempFish.pts = pts
            # # Updating the bounding box
            boundingBox = tempFish.boundingBox
            boundingBox.smallX += amount_off_left
            boundingBox.bigX += amount_off_left
            boundingBox.smallY += amount_off_top
            boundingBox.bigY += amount_off_top
            tempFish.boundingBox = boundingBox

            self.fishList[0] = tempFish

    def addArtifactsOrDistortions(self):
        if Config.shouldResize: self.resize()
        self.randomly_crop()
        if Config.shouldBlurr: self.smallCanvas = add_blur(self.smallCanvas)

    def addNoise(self):
        self.smallCanvas = add_background_noise(self.smallCanvas)
        # self.smallCanvas = add_background_noise_flipped(self.smallCanvas)
        # NOTE: might want to add some patches to frames without a fish
        if len(self.fishList) != 0:
            self.smallCanvas = add_patchy_noise(self.smallCanvas, self.fishList[0])
   
    def addNoiseBackgroundBlurred(self):
        background = add_background_noise_flipped(np.zeros( self.smallCanvas.shape) )
        background = add_blur(background)
        #cv.imwrite('temp.png', background)
        
        #background[ self.smallCanvas > 0  ] = self.smallCanvas[ self.smallCanvas > 0  ]
        
        #self.smallCanvas = cv.add( self.smallCanvas, background )
        
        self.smallCanvas = np.clip( self.smallCanvas + background, 0, 255)
        self.smallCanvas = applyGaussianKernel(self.smallCanvas)

        # NOTE: might want to add some patches to frames without a fish
        
        if len(self.fishList) != 0 and Config.shouldAddPatchyNoise:
            self.smallCanvas = add_patchy_noise(self.smallCanvas, self.fishList[0])
    

    def saveImage(self):
        strIdxInFormat = format(self.frame_idx, '06d')
        path = Config.dataDirectory + 'images/im_' + strIdxInFormat + '.png'
        cv.imwrite(path, self.smallCanvas)

    def saveAnnotations(self):
        strIdxInFormat = format(self.frame_idx, '06d')
        #path = Config.dataDirectory + 'coor_2d/ann_' + strIdxInFormat + '.npy'
        path = Config.dataDirectory + 'coor_2d/ann_' + strIdxInFormat + '.pt'

        pts = np.zeros((2,12))
        if len(self.fishList) != 0:
            pts = self.fishList[0].pts

        #np.save(path, pts)
        
        pts = torch.tensor( pts, dtype=torch.float32)
        torch.save(pts, path) 


    def draw(self):
        for fish in self.fishList:
            fish.draw()

        for fish in self.fishList:
            if fish.shouldHaveReflection():
                fish.drawReflection()

        if Config.shouldDrawCircles:
            if len(self.fishList) != 0:
                fish.graymodel = draw_circle(self.circle, fish.graymodel)
                fish.graymodel = draw_circle(self.top_circle, fish.graymodel)

        for fish in self.fishList:
            self.smallCanvas = fish.graymodel

        self.addArtifactsOrDistortions()

        self.addNoise()
        # self.addNoiseBackgroundBlurred()
        
        # self.smallCanvas = np.stack([self.smallCanvas for _ in range(3)], axis=2)
        # pts = self.fishList[0].pts
        # pts = pts.astype(int)
        # red = [0,0,255]
        # green = [0,255,0]
        # self.smallCanvas[pts[1,:10], pts[0,:10], :] = green
        # self.smallCanvas[pts[1,10:], pts[0,10:]] = red
        # cv.imwrite('test.png', self.smallCanvas)



class Fish:
    class BoundingBox:
        BoundingBoxThreshold = Config.boundingBoxThreshold
        def __init__(self, smallY, bigY, smallX, bigX):
            self.smallY = smallY
            self.bigY = bigY
            self.smallX = smallX
            self.bigX = bigX

        def getHeight(self):
            return ( self.bigY - self.smallY)

        def getWidth(self):
            return ( self.bigX - self.smallX)

        def getCenterX(self):
            return (( self.bigX + self.smallX) / 2)

        def getCenterY(self):
            return (( self.bigY + self.smallY) / 2)

        def isValidBox(self):
            height = self.getHeight()
            width = self.getWidth()

            if (height <= Fish.BoundingBox.BoundingBoxThreshold) or (width <= Fish.BoundingBox.BoundingBoxThreshold):
                return False
            else:
                return True

    def __init__(self, fishVect, frameSizeY, frameSizeX, should_put_fish_near_edge, circle, top_circle):
        self.fishVect = fishVect
        self.frameSizeY = frameSizeY
        self.frameSizeX = frameSizeX
        self.should_put_fish_near_edge = should_put_fish_near_edge
        self.circle = circle
        self.top_circle = top_circle

    def draw(self):
        fishVect = [self.fishVect]

        seglen = fishVect[0][0]
        #seglen = 2 + np.random.rand()
        xVect = fishVect[0][1:]
        x = np.zeros((11,))
        xVect = np.array(xVect)
        x[:] = xVect[:]

        graymodel, pts = f_x_to_model_bigger(x, seglen, Config.randomizeFish, self.frameSizeX, self.frameSizeY)

        self.pts = pts
        self.graymodel = graymodel
        self.vis = np.ones((pts.shape[1]))
        # self.vis = np.zeros((pts.shape[1]))
        # self.vis[ self.valid_points_masks ] = 1

        # Creating the bounding box
        nonzero_coors = np.array( np.where(graymodel > 0) )
        try:
            smallY = np.min(nonzero_coors[0,:])
            bigY = np.max( nonzero_coors[0,:])
            smallX = np.min( nonzero_coors[1,:])
            bigX = np.max( nonzero_coors[1,:])
        except:
            smallY = 0
            bigY = 0
            smallX = 0
            bigX = 0
        self.boundingBox = Fish.BoundingBox(smallY, bigY, smallX, bigX)

    def shouldHaveReflection(self):
        yesOrNo = True if np.random.rand() < Config.fraction_to_draw_reflection else False

        # if we decided to put the fish near the edge then that was because we want a reflection
        if self.should_put_fish_near_edge: yesOrNo = True

        return yesOrNo

    def drawReflection(self):
        cylinder_height_in_pixels = Config.cylinder_height
        # # adding reflections
        # TODO: define a better way for defining and getting the angles
        y_distance = self.top_circle.centerY - self.circle.centerY
        y_distance *= -1
        # gamma_in_degrees = np.arccos(y_distance/cylinder_height_in_pixels) * (180/np.pi)
        # gamma_in_degrees = np.abs(90 - gamma_in_degrees) * 2
        gamma_in_degrees = np.arctan(y_distance / cylinder_height_in_pixels) * (180 / np.pi)

        x_distance = self.top_circle.centerX - self.circle.centerX
        # phi_in_degrees = np.arccos(x_distance/cylinder_height_in_pixels) * (180/np.pi)
        # phi_in_degrees = np.abs(90 - phi_in_degrees)
        phi_in_degrees = np.arctan(x_distance / cylinder_height_in_pixels) * (180 / np.pi)

        reflection, mapped_pts = cylinder_distort(self.graymodel, self.circle, self.top_circle,
                                                  gamma_in_degrees, phi_in_degrees, self.pts)

        # Threshold to keep out points that could be confused with noise from being considered
        visibility_threshold = 15
        # Threshold that decides how many points should be visible to count as the reflections being visible
        amount_of_points_threshold = 2
        is_reflection_visible = False
        reflected_fish = None
        amount_of_visible_points = len(np.where(reflection > visibility_threshold)[0])
        if amount_of_visible_points > amount_of_points_threshold:
            is_reflection_visible = True

        reflection = np.nan_to_num(reflection)

        mask = get_valid_points_mask(mapped_pts, reflection.shape)
        pts_int = mapped_pts.astype(int)
        pts_int_x = pts_int[0, :]
        pts_int_y = pts_int[1, :]
        values = reflection[pts_int_y[mask], pts_int_x[mask]]
        reflection_vis = np.zeros((self.pts.shape[1]))
        reflection_vis[mask] = values > visibility_threshold

        # erase_mask = np.copy(im.astype(int))
        # flag_value = -8
        # erase_mask[erase_mask <= 0] = flag_value
        # erase_mask[erase_mask > 0] = 0
        # erase_mask[erase_mask < 0] = 1
        # reflection *= erase_mask


        # if we forced the fish to be in a position for a reflection then we should draw its reflection
        # if should_put_fish_near_edge: should_add_reflection = True

        reflection *= Config.fraction_to_dim_reflection_by
        # if should_add_reflection:
        self.graymodel[(self.graymodel < visibility_threshold) * (reflection > 0)] = reflection[
            (self.graymodel < visibility_threshold) * (reflection > 0)]


    @property
    def xs(self):
        return self.pts[0,:]
    @property
    def ys(self):
        return self.pts[1,:]
    @property
    def intXs(self):
        return np.ceil( self.pts[0,:] ).astype(int)
    @property
    def intYs(self):
        return np.ceil( self.pts[1,:]).astype(int)
    # @property
    # def valid_points_masks(self):
    #     xs = self.intXs
    #     ys = self.intYs
    #     xs_in_bounds = (xs < imageSizeX) * (xs >= 0)
    #     ys_in_bounds = (ys < imageSizeY) * (ys >= 0)
    #     return xs_in_bounds * ys_in_bounds

    # def amount_of_vis_points(self):
    #     val_xs = self.pts[0,:][self.valid_points_masks]
    #     return val_xs.shape[0]

    @property
    def is_valid_fish(self):
        # if (self.amount_of_vis_points() >= 1) and self.boundingBox.isValidBox():
        #     return True
        # else:
        #     return False
        return True

class Circle:
    def __init__(self, centerX, centerY, radius):
        self.radius = radius
        self.centerX = centerX
        self.centerY = centerY

    # For Visualization Purposes
    @property
    def centerXInt(self):
        return int(self.centerX)
    def centerYInt(self):
        return int(self.centerY)






