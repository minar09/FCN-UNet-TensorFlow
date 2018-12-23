from __future__ import print_function
import function_definitions as fd
import cv2
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.color import gray2rgb
from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral, create_pairwise_gaussian, unary_from_softmax
from skimage.io import imread, imsave
import pydensecrf.densecrf as dcrf
from six.moves import xrange
import BatchDatsetReader as dataset
import datetime
import read_MITSceneParsingData as scene_parsing
import TensorflowUtils as utils
from PIL import Image
import numpy as np
import tensorflow as tf

# Hide the warning messages about CPU/GPU
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


"""
import for crf
"""


"""
    calculate IoU
"""


def _calcIOU(gtimage, predimage, num_classes):
    IoUs = []
    for i in range(num_classes):
        IoUs.append([0] * num_classes)

    height, width = gtimage.shape

    for label in range(num_classes):  # 0--> 17
        intersection = 0
        union = 0

        for y in range(height):  # 0->223
            # print(crossMat)

            for x in range(width):  # =-->223
                gtlabel = gtimage[y, x]
                predlabel = predimage[y, x]

                if predlabel >= num_classes or gtlabel >= num_classes:
                    print('gt:%d, pr:%d' % (gtlabel, predlabel))
                else:
                    if(gtlabel == label and predlabel == label):
                        intersection = intersection + 1
                    if(gtlabel == label or predlabel == label):
                        union = union + 1

        if(union == 0):
            IoUs[label] = 0.0
        else:
            # print("label:", label , "intersection:", intersection, " -
            # union:", union)
            IoUs[label] = (float)(intersection) / union

    return IoUs


"""
    calculate confusion matrix
"""


def _calcCrossMat(gtimage, predimage, num_classes):
    crossMat = []

    for i in range(num_classes):
        crossMat.append([0] * num_classes)
    # print(crossMat)
    height, width = gtimage.shape

    for y in range(height):
        # print(crossMat)

        for x in range(width):
            gtlabel = gtimage[y, x]
            predlabel = predimage[y, x]
            if predlabel >= num_classes or gtlabel >= num_classes:
                print('gt:%d, pr:%d' % (gtlabel, predlabel))
            else:
                crossMat[gtlabel][predlabel] = crossMat[gtlabel][predlabel] + 1

    return crossMat


"""
    calculate frequency weighted IoU
"""


def _calcFrequencyWeightedIOU(gtimage, predimage, num_classes):
    FrqWIoU = []
    for i in range(num_classes):
        FrqWIoU.append([0] * num_classes)

        gt_pixels = []
    height, width = gtimage.shape

    for label in range(num_classes):  # 0--> 17
        intersection = 0
        union = 0
        pred = 0
        gt = 0

        for y in range(height):  # 0->223
            # print(crossMat)

            for x in range(width):  # =-->223
                gtlabel = gtimage[y, x]
                predlabel = predimage[y, x]

                if predlabel >= num_classes or gtlabel >= num_classes:
                    print('gt:%d, pr:%d' % (gtlabel, predlabel))
                else:
                    if(gtlabel == label and predlabel == label):
                        intersection = intersection + 1
                        gt = gt + 1
                        pred = pred + 1
                    elif(gtlabel == label or predlabel == label):
                        union = union + 1
                        if(gtlabel == label):
                            gt = gt + 1
                        elif(predlabel == label):
                            pred = pred + 1

                gt_pixels.append(gt)
                
        # union = gt + pred - intersection
        # intersection = gt * pred
        # FrqWIoU[label] = (float)(intersection * gt) / union

        if(union == 0):
            FrqWIoU[label] = 0.0
        else:
            FrqWIoU[label] = (float)(intersection * gt) / union

    #pixel_sum = np.sum(gt_pixels)
    pixel_sum = predimage.shape[0] * predimage[1]
    
    if(pixel_sum == 0):
            meanFrqWIoU = 0.0
    else:
        meanFrqWIoU = (float)(np.sum(FrqWIoU)) / pixel_sum
    
    return FrqWIoU, meanFrqWIoU


"""
   Function which returns the labelled image after applying CRF
"""
# Original_image = Image which has to labelled
# Annotated image = Which has been labelled by some technique( FCN in this case)
# Output_image = The final output image after applying CRF
# Use_2d = boolean variable
# if use_2d = True specialised 2D fucntions will be applied
# else Generic functions will be applied


def crf(original_image, annotated_image, use_2d=True):

    # Converting annotated image to RGB if it is Gray scale
    print("crf function")

    # Converting the annotations RGB color to single 32 bit integer

    annotated_label = annotated_image[:,
                                      :,
                                      0] + (annotated_image[:,
                                                            :,
                                                            1] << 8) + (annotated_image[:,
                                                                                        :,
                                                                                        2] << 16)

    # Convert the 32bit integer color to 0,1, 2, ... labels.
    colors, labels = np.unique(annotated_label, return_inverse=True)

    # Creating a mapping back to 32 bit colors
    colorize = np.empty((len(colors), 3), np.uint8)
    colorize[:, 0] = (colors & 0x0000FF)
    colorize[:, 1] = (colors & 0x00FF00) >> 8
    colorize[:, 2] = (colors & 0xFF0000) >> 16

    # Gives no of class labels in the annotated image
    n_labels = NUM_OF_CLASSESS
    print("No of labels in the Image are ")

    # Setting up the CRF model
    if use_2d:
        d = dcrf.DenseCRF2D(
            original_image.shape[1],
            original_image.shape[0],
            n_labels)

        # get unary potentials (neg log probability)
        processed_probabilities = annotated_image
        softmax = processed_probabilities.transpose((2, 0, 1))

        U = unary_from_softmax(softmax)
        # U = unary_from_labels(labels, n_labels, gt_prob=0.7, zero_unsure=False)
        d.setUnaryEnergy(U)

        # This adds the color-independent term, features are the locations
        # only.
        d.addPairwiseGaussian(sxy=(3, 3), compat=3, kernel=dcrf.DIAG_KERNEL,
                              normalization=dcrf.NORMALIZE_SYMMETRIC)

        # This adds the color-dependent term, i.e. features are (x,y,r,g,b).
        d.addPairwiseBilateral(
            sxy=(
                10,
                10),
            srgb=(
                13,
                13,
                13),
            rgbim=original_image,
            compat=10,
            kernel=dcrf.DIAG_KERNEL,
            normalization=dcrf.NORMALIZE_SYMMETRIC)

    # Run Inference for 5 steps
    Q = d.inference(20)

    # Find out the most probable class for each pixel.
    MAP = np.argmax(Q, axis=0)

    # Convert the MAP (labels) back to the corresponding colors and save the image.
    # Note that there is no "unknown" here anymore, no matter what we had at
    # first.
    MAP = colorize[MAP, :]

    # Get output
    output = MAP.reshape(original_image.shape)
    output = rgb2gray(output)

    return MAP.reshape(original_image.shape), output


def vgg_net(weights, image):
    layers = (
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
        'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
        'relu4_3', 'conv4_4', 'relu4_4', 'pool4',


        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
        'relu5_3', 'conv5_4', 'relu5_4'
    )

    net = {}
    current = image
    for i, name in enumerate(layers):
        kind = name[:4]
        if kind == 'conv':
            kernels, bias = weights[i][0][0][0][0]
            # matconvnet: weights are [width, height, in_channels, out_channels]
            # tensorflow: weights are [height, width, in_channels,
            # out_channels]
            kernels = utils.get_variable(np.transpose(
                kernels, (1, 0, 2, 3)), name=name + "_w")
            bias = utils.get_variable(bias.reshape(-1), name=name + "_b")
            current = utils.conv2d_basic(current, kernels, bias)
        elif kind == 'relu':
            current = tf.nn.relu(current, name=name)
            # if FLAGS.debug:
            # util.add_activation_summary(current)
        elif kind == 'pool':
            current = utils.avg_pool_2x2(current)
        net[name] = current
        # added for resume better
    global_iter_counter = tf.Variable(0, name='global_step', trainable=False)
    net['global_step'] = global_iter_counter

    return net
