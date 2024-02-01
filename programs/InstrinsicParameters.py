import yaml
import warnings
import functools

# NOTE: it is important to import the librarys that you use to model the intrinsic parameters here,
#       some general ones have already been imported below.

import numpy as np
from numpy.random import normal as normrnd
from scipy.stats import norm
from scipy.stats import skewnorm

class IntrinsicParameters:
    """
        Class that obtains all the intrinsic parameter values from the IntrinsicParameters.yaml file
    """

    # WARNING: the ordering below is important

    belly_br_function = np.random.normal
    belly_br_arguments = 0.405766627860074, 0.08510657640471547
    belly_br_function_call = None

    belly_l_function = np.random.normal
    belly_l_arguments = 1.0414934216489478, 0.0497660715368968
    belly_l_function_call = None

    belly_w_function = np.random.normal
    belly_w_arguments = 0.4509085008872612, 0.04318248156070455
    belly_w_function_call = None

    c_belly_function = np.random.normal
    c_belly_arguments = 0.8976990388969368, 0.046408648027029396
    c_belly_function_call = None

    c_eye_function = np.random.normal
    c_eye_arguments = 1.7175761389389386, 0.20589859105033523
    c_eye_function_call = None

    c_head_function = skewnorm.rvs
    c_head_arguments = 13.160772264490056, 0.9469252531374381, 0.3251749094426817
    c_head_function_call = None

    d_eye_function = np.random.normal
    d_eye_arguments = 1.144396567090266, 0.04442945643226941
    d_eye_function_call = None

    eye_br_function = skewnorm.rvs
    eye_br_arguments = -8.231518385739356, 0.6969923960604942, 0.1853606969178898
    eye_br_function_call = None

    eye_l_function = np.random.normal
    eye_l_arguments = 0.5122961600634638, 0.07434852108369809
    eye_l_function_call = None

    eye_w_function = np.random.normal
    eye_w_arguments = 0.4209238403150402, 0.040920763968052117
    eye_w_function_call = None

    head_br_function = np.random.normal
    head_br_arguments = 0.27670705295942777, 0.05571972183069393
    head_br_function_call = None

    head_l_function = np.random.normal
    head_l_arguments = 0.7889611717422198, 0.10205659701587598
    head_l_function_call = None

    head_w_function = np.random.normal
    head_w_arguments = 0.5077558460625518, 0.032475097562161506
    head_w_function_call = None



    firstFunction = print
    firstFunctionArguments = 1
    firstFunctionCall = None # The calls will be assigned at the end

    secondFunction = np.random.rand
    secondFunctionArguments = 3
    secondFunctionCall = None

    # The following is important for stitching together the function call.
    # The order is important


    # # This next part is important for parsing
    # functions = [firstFunction, secondFunction]
    # arguments = [firstFunctionArguments, secondFunctionArguments]
    # calls = [firstFunctionCall, secondFunctionCall]


    def __init__(self, pathToYamlFile):
        """
            Essentially just a function to update the variables accordingly
        """


        functions = [IntrinsicParameters.firstFunction, IntrinsicParameters.secondFunction]
        arguments = [IntrinsicParameters.firstFunctionArguments, IntrinsicParameters.secondFunctionArguments]
        calls = [IntrinsicParameters.firstFunctionCall, IntrinsicParameters.secondFunctionCall]

        static_vars = list(vars(IntrinsicParameters))[2:-3]

        file = open(pathToYamlFile, 'r')
        config = yaml.safe_load(file)
        keys = config.keys()
        list_of_vars_in_config = list(keys)

        # Updating the static variables
        for var in list_of_vars_in_config:
            if var in static_vars:
                value = config[var]
                line = 'IntrinsicParameters.' + var + ' = '

                line += str(value)

                try:
                    exec(line)
                except:
                    warnings.warn('\n' + line + ' could not be executed. \nYou might have forgotten to import the library in the Programs/IntrinsicParameters.py file \n' )
            else:
                warnings.warn(var + ' is not a valid variable, could be a spelling issue')

        # print(static_vars)



        # Parsing the Static Variables to get ready to stitch them together
        amountOfVar = int( round( len(static_vars) / 3))
        step = 3
        for varIdx in range(amountOfVar):
            realIdx = varIdx * step
            function = static_vars[realIdx]
            functionArguments = static_vars[realIdx + 1]
            functionCall = static_vars[realIdx + 2]

            # print(exec('type(IntrinsicParameters.' + functionArguments + ')'))
            # print(exec('type(IntrinsicParameters.' + functionArguments + ') is tuple'))


            # exec(
            # """print('One of the arguments was a tuple') if type(IntrinsicParameters.""" + functionArguments + """) is tuple \
            #  else print('One of the arguments was not a tuple')""")


            # print("""IntrinsicParameters.""" + functionCall + """\
            #  = functools.partial("""+ function +""", *"""+ functionArguments +""") \
            # if type(IntrinsicParameters.""" + functionArguments + """) is tuple \
            #  else print('One of the arguments was not a tuple')""")

            # exec(
            # """IntrinsicParameters.""" + functionCall + """\
            #  = functools.partial(IntrinsicParameters."""+ function +""", *IntrinsicParameters."""+ functionArguments +""") \
            # if type(IntrinsicParameters.""" + functionArguments + """) is tuple \
            #  else IntrinsicParameters."""+ functionCall +""" = functools.partial(IntrinsicParameters."""+ function +""", IntrinsicParameters."""+ functionArguments +""")  """)

            exec(
            """IntrinsicParameters.""" + functionCall + """\
             = functools.partial(IntrinsicParameters."""+ function +""", *IntrinsicParameters."""+ functionArguments +""") \
            if type(IntrinsicParameters.""" + functionArguments + """) is tuple \
             else functools.partial(IntrinsicParameters."""+ function +""", IntrinsicParameters."""+ functionArguments +""")  """)



        # Stitching together the calls
        # for functionCall, function, arguments in zip(calls, functions, arguments):
        #     print(functionCall)
        #
        #     if type(arguments) is tuple:
        #
        #         functionCall = functools.partial(function, *arguments)
        #     else:
        #         functionCall = functools.partial(function, arguments)



        # for call in calls:
        #     call = print
        #     IntrinsicParameters.firstFunctionCall = print
        #
        # for functionCall, function, arguments in zip(calls, functions, arguments):
        #
        #     if type(arguments) is tuple:
        #
        #         functionCall = functools.partial(function, *arguments)
        #     else:
        #         functionCall = functools.partial(function, arguments)

        #IntrinsicParameters.firstFunctionCall = functools.partial(IntrinsicParameters.firstFunction, *IntrinsicParameters.firstFunctionArguments)
        # IntrinsicParameters.third = functools.partialmethod(print, 'hello')

IntrinsicParameters('inputs/IntrinsicParameters.yaml')

# IntrinsicParameters = 5

# print( IntrinsicParameters.firstFunction( *(IntrinsicParameters.firstFunctionArguments) ) )



# print(IntrinsicParameters.eye_br_function_call())


# print(IntrinsicParameters.firstFunctionCall())
