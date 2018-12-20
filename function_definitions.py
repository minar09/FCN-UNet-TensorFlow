from __future__ import print_function
import tensorflow as tf
import numpy as np
from PIL import Image
import TensorflowUtils as utils
import read_MITSceneParsingData as scene_parsing
import datetime
import BatchDatsetReader as dataset
from six.moves import xrange

"""
import for crf
"""
import pydensecrf.densecrf as dcrf
from skimage.io import imread, imsave
from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral, create_pairwise_gaussian, unary_from_softmax

from skimage.color import gray2rgb
from skimage.color import rgb2gray
import matplotlib.pyplot as plt

import cv2
import function_definitions as fd


"""
    calculate IoU
"""
def _calcIOU(gtimage, predimage, n):
    IoUs = []
    for i in range(n):
        IoUs.append([0] * n)

    height, width = gtimage.shape
    gtlabel = prelabel = 0
    for label in range(n): #0--> 17
        intersection = 0
        union = 0
        for y in range(height):#0->223
            # print(crossMat)
            for x in range(width):#=-->223
                gtlabel = gtimage[y, x]
                predlabel = predimage[y, x]

                if predlabel >= n or gtlabel >= n:
                    print('gt:%d, pr:%d' % (gtlabel, predlabel))
                else:
                    if(gtlabel==label and predlabel == label):
                        intersection = intersection + 1
                    if(gtlabel==label or predlabel == label):
                        union = union + 1
        if(union == 0):
            IoUs[label] = 0.0
        else:
            #print("label:", label , "intersection:", intersection, " - union:", union)
            IoUs[label] = (float)(intersection) / union
    return IoUs

	
"""
    calculate confusion matrix
"""
def _calcCrossMat(gtimage, predimage, n):
    crossMat = []
    for i in range(n):
        crossMat.append([0] * n)
    # print(crossMat)
    height, width = gtimage.shape
    gtlabel = prelabel = 0
    for y in range(height):
        # print(crossMat)
        for x in range(width):
            gtlabel = gtimage[y, x]
            predlabel = predimage[y, x]
            if predlabel >= n or gtlabel >= n:
                print('gt:%d, pr:%d' % (gtlabel, predlabel))
            else:
                crossMat[gtlabel][predlabel] = crossMat[gtlabel][predlabel] + 1;

    return crossMat

	
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

    #if (len(annotated_image.shape) < 3):
    #    annotated_image = gray2rgb(annotated_image)

    #imsave("testing2.png", annotated_image)

    # Converting the annotations RGB color to single 32 bit integer

    annotated_label = annotated_image[:, :, 0] + (annotated_image[:, :, 1] << 8) + (annotated_image[:, :, 2] << 16)

    # Convert the 32bit integer color to 0,1, 2, ... labels.
    colors, labels = np.unique(annotated_label, return_inverse=True)
    #print("colors: ",colors )
    #print("labels: ",labels )
    # Creating a mapping back to 32 bit colors
    colorize = np.empty((len(colors), 3), np.uint8)
    colorize[:, 0] = (colors & 0x0000FF)
    colorize[:, 1] = (colors & 0x00FF00) >> 8
    colorize[:, 2] = (colors & 0xFF0000) >> 16

    # Gives no of class labels in the annotated image
    #n_labels = len(set(labels.flat))
    n_labels = NUM_OF_CLASSESS
    print("No of labels in the Image are ")
    #print(n_labels)
    #print("annotated shape:", annotated_image.shape)

    # Setting up the CRF model
    if use_2d:
        d = dcrf.DenseCRF2D(original_image.shape[1], original_image.shape[0], n_labels)

        # get unary potentials (neg log probability)
        processed_probabilities = annotated_image
        softmax = processed_probabilities.transpose((2, 0, 1))

        U = unary_from_softmax(softmax)
        #U = unary_from_labels(labels, n_labels, gt_prob=0.7, zero_unsure=False)
        d.setUnaryEnergy(U)

        # This adds the color-independent term, features are the locations only.
        d.addPairwiseGaussian(sxy=(3, 3), compat=3, kernel=dcrf.DIAG_KERNEL,
                              normalization=dcrf.NORMALIZE_SYMMETRIC)

        # This adds the color-dependent term, i.e. features are (x,y,r,g,b).
        d.addPairwiseBilateral(sxy=(10, 10), srgb=(13, 13, 13), rgbim=original_image,
                               compat=10,
                               kernel=dcrf.DIAG_KERNEL,
                               normalization=dcrf.NORMALIZE_SYMMETRIC)

    # Run Inference for 5 steps
    Q = d.inference(20)
    #print(">>>>>>>>Qshape: ", Q.shape)
    # Find out the most probable class for each pixel.
    MAP = np.argmax(Q, axis=0)
    #print(">>>>>>>>MAP shape: ", MAP.shape)

    # Convert the MAP (labels) back to the corresponding colors and save the image.
    # Note that there is no "unknown" here anymore, no matter what we had at first.
    MAP = colorize[MAP, :]

    #print(">>>>>>>>>COlorized map shape: ", MAP.shape)
    #print(">>>>>>>>>map ount non zero0: ", np.count_nonzero(MAP,axis = 0))
    #print(">>>>>>>>>map ount non zero1: ", np.count_nonzero(MAP,axis = 1))

    output = MAP.reshape(original_image.shape)
    output = rgb2gray(output)
    #plt.imshow(output, cmap = "nipy_spectral")
    #print(">>>>>>>>>COlorized map shape: ", output.shape)
    #print(">>>>>>>>>map ount non zero0: ", np.count_nonzero(output, axis=0))
    #print(">>>>>>>>>map ount non zero1: ", np.count_nonzero(output, axis=1))
    #import cv2;
    #hist_crf = cv2.calcHist([output.astype(np.uint8)], [0], None, [256], [0, 256])
    #print("hist)crf:", hist_crf)
    #plt.show()
    #Get output

    """
    annotated_image  = np.reshape(annotated_image, [224*224,-1] )

    print(">>>>>>>>>>>>>>>>>>>shape annotated image0: ", annotated_image)
    MAP_bf = np.argmax(annotated_image, axis=0)
    print(">>>>>>>>>>>>>>>>>>>shape annotated image1: ", MAP_bf)
    MAP_bf = colorize[MAP_bf, :]
    print(">>>>>>>>>>>>>>>>>>>shape annotated image2: ", MAP_bf.shape)
    output_bf = MAP_bf.reshape(original_image.shape)
    output_bf = rgb2gray(output_bf)

    fig = plt.figure(figsize=(1,3))
    #error = output - annotated_image
    fig.add_subplot(1,1,original_image)
    fig.add_subplot(1,2, output_bf)
    fig.add_subplot(1,3, output)

    #fig.add_subplot(1,4,rgb2gray(error.reshape(original_image.shape)))
    plt.show()
    """
    #imsave(output_image, MAP.reshape(original_image.shape))
    #imsave(output_image + ".png", output)
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
            # tensorflow: weights are [height, width, in_channels, out_channels]
            kernels = utils.get_variable(np.transpose(kernels, (1, 0, 2, 3)), name=name + "_w")
            bias = utils.get_variable(bias.reshape(-1), name=name + "_b")
            current = utils.conv2d_basic(current, kernels, bias)
        elif kind == 'relu':
            current = tf.nn.relu(current, name=name)
            #if FLAGS.debug:
                #util.add_activation_summary(current)
        elif kind == 'pool':
            current = utils.avg_pool_2x2(current)
        net[name] = current
        # added for resume better
    global_iter_counter = tf.Variable(0, name='global_step', trainable=False)
    net['global_step'] = global_iter_counter

    return net
