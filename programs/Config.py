import yaml
import warnings

class Config:
    """
        Class that obtains all the variables from the configuration file
    """
    #   The Following are Default values

    # Data Variables
    dataDirectory = 'data'

    averageSizeOfFish = 70
    randomizeFish = 1

    average_radius = 80
    max_radius_offset = 10
    average_side_padding = 33
    max_side_padding_offset = 10
    cylinder_height = 90

    fraction_of_data_that_should_be_background = .1
    chance_of_putting_fish_near_edge = .5
    maximum_distance_from_edge = 15
    fraction_to_draw_reflection = .5
    fraction_to_dim_reflection_by = .8

    original_length = 1966
    original_height = 1480
    max_length_height_offset = 20
    shrunken_length = 960
    shrunken_height = 720

    max_height = 101
    max_length = 101

    # The top circle is set up to always allow for reflections, but they
    # only cause reflections a fraction of the time
    max_angle_offset = 15
    chance_that_the_angle_should_vary = .9
    minimum_distance_from_bottom_circle = 10
    max_additional_distance_from_bottom_circle = 20

    #   Thresholds
    # The following variable is similar to the one above except that it is for the bounding box of the fish passed
    # to Yolo.  This is necessary because sometimes the fish are barely visible at the edge causing the model to
    # learn to detect the edges as fish
    boundingBoxThreshold = 2

    averageAmountOfPatchyNoise = .2


    amountOfData = 50000
    fractionForTraining = .9

    # Parameters relating to the wells
    columns = 8
    rows = 6
    radius = 80
    wellPadding = 33
    fractionOfShiftedTopWells = .5
    minimumTopWellShift = 10
    maximumTopWellShift = 34

    # Dimensions which the original image will be resized to
    newHeight = 720
    newWidth = 960
    # Padding Variables
    max_padding_offset = 10
    average_left_of_frame_padding = 86
    average_right_of_frame_padding = 72
    average_top_of_frame_padding = 60
    average_bottom_of_frame_padding = 64


    # Noise Variables
    maxAmountOfPatchableFish = 9

    # Flags
    shouldAddPatchyNoise = True
    shouldAddStaticNoise = True
    shouldDrawBottomWells = False
    shouldDrawTopWells = False
    shouldSaveImages = True
    shouldSaveAnnotations = True
    shouldResize = False
    shouldBlurr = False
    shouldDrawCircles = False

    # None for now since it is going to get set after checking the yaml file
    biggestIdx4TrainingData = None

    # TODO: try setting this to a static method to make it more natural
    def __init__(self, pathToYamlFile):
        """
            Essentially just a function to update the variables accordingly
        """
        static_vars = list(vars(Config))[2:-3]

        file = open(pathToYamlFile, 'r')
        config = yaml.safe_load(file)
        keys = config.keys()
        list_of_vars_in_config = list(keys)

        # Updating the static variables
        for var in list_of_vars_in_config:
            if var in static_vars:
                value = config[var]
                line = 'Config.' + var + ' = '

                if not isinstance(value, str):
                    line += str(value)
                else:
                    line += "'" + value + "'"
                exec(line)
            else:
                warnings.warn(var + ' is not a valid variable, could be a spelling issue')

        Config.biggestIdx4TrainingData = Config.amountOfData * Config.fractionForTraining
        Config.dataDirectory += '/'

        # NOTE: the following was just left as an example for now
        # # Writing the variables to the corresponding classes static variables
        # Config.set_aquarium_vars()
        # Config.set_bounding_box_vars()
    # @staticmethod
    # def set_bounding_box_vars():
    #     print('setting the bounding box vars')

# Setting the variables of the Configuration Class
Config('inputs/config.yaml')
