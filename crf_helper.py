#
# CRF
#
import numpy as np
import pydensecrf.densecrf as dcrf

from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral, create_pairwise_gaussian, unary_from_softmax
# from pydensecrf import utils as utils

# import pydensecrf.utils as dcrfutils


# unary_from_labels()
# dcrfutils.unary_from_labels()

# Tuan - Function which returns the labelled image after applying CRF

#
#  building cross matrix
#
def calcConfussionMat(gtimage, predimage, n):
    confMat = []
    for i in range(n):
        confMat.append([0] * n)
    # print(confMat)
    height, width = gtimage.shape
    gtlabel = prelabel = 0

    for y in range(height):
        # print(confMat)
        for x in range(width):
            gtlabel = gtimage[y, x]
            predlabel = predimage[y, x]
            if predlabel >= n or gtlabel >= n:
                print('gt:%d, pr:%d' % (gtlabel, predlabel))
            else:
                confMat[gtlabel][predlabel] = confMat[gtlabel][predlabel] + 1

    return confMat

#
#  perf print
#


def calcuateAccuracy(mat, bprint=True):

    n = len(mat)

    acc = 0
    tot = 0
    for i in range(1, n):
        acc += mat[i][i]
        tot += sum(mat[i])
    acc1 = acc/tot

    if bprint:
        print("Acc-fg: %5.3f (%d/%d)" % (acc/tot, acc, tot))
    acc += mat[0][0]
    tot += sum(mat[0])
    if bprint:
        print("Acc-all: %5.3f (%d/%d)" % (acc/tot, acc, tot))

    return acc1, acc/tot  # fg accuracy, total accuracy


# Original_image = Image which has to labelled
# Annotated image = Which has been labelled by some technique( FCN in this case)
# Output_image = The final output image after applying CRF
# Use_2d = boolean variable
# if use_2d = True specialised 2D fucntions will be applied
# else Generic functions will be applied

def crf_with_probs(orig, probs, num_label, num_iter=5, use_2d=True):

    # Setting up the CRF model
    np.set_printoptions(threshold=10)
    probs = probs.transpose((2, 0, 1))
    #print("probs:", probs)
    #print("probs shape:", probs.shape)

    if use_2d:
        d = dcrf.DenseCRF2D(orig.shape[1], orig.shape[0], num_label)

        # get unary potentials (neg log probability)
        U = unary_from_softmax(probs)
        d.setUnaryEnergy(U)

        # This adds the color-independent term, features are the locations only.
        d.addPairwiseGaussian(sxy=(3, 3), compat=3, kernel=dcrf.DIAG_KERNEL,
                              normalization=dcrf.NORMALIZE_SYMMETRIC)

        # This adds the color-dependent term, i.e. features are (x,y,r,g,b).
        d.addPairwiseBilateral(sxy=(10, 10), srgb=(13, 13, 13), rgbim=orig,
                               compat=10,
                               kernel=dcrf.DIAG_KERNEL,
                               normalization=dcrf.NORMALIZE_SYMMETRIC)

    # Run Inference for 5 steps
    Q = d.inference(num_iter)
    # print(">>>>>>>>Qshape: ", Q.shape)
    # Find out the most probable class for each pixel.
    MAP = np.argmax(Q, axis=0)
    output = MAP.reshape((orig.shape[1], orig.shape[0]))  # orig.shape[)
    return output


def crf_with_labels(orig, annotated_image, num_label, num_iter=5, use_2d=True):

    # Setting up the CRF model
    if use_2d:
        d = dcrf.DenseCRF2D(orig.shape[1], orig.shape[0], num_label)

        # get unary potentials (neg log probability)
        U = unary_from_labels(annotated_image, num_label,
                              gt_prob=0.7, zero_unsure=False)
        d.setUnaryEnergy(U)

        # This adds the color-independent term, features are the locations only.
        d.addPairwiseGaussian(sxy=(3, 3), compat=3, kernel=dcrf.DIAG_KERNEL,
                              normalization=dcrf.NORMALIZE_SYMMETRIC)

        # This adds the color-dependent term, i.e. features are (x,y,r,g,b).
        d.addPairwiseBilateral(sxy=(10, 10), srgb=(13, 13, 13), rgbim=orig,
                               compat=10,
                               kernel=dcrf.DIAG_KERNEL,
                               normalization=dcrf.NORMALIZE_SYMMETRIC)

    # Run Inference for 5 steps
    Q = d.inference(num_iter)
    MAP = np.argmax(Q, axis=0)
    # original_image.shape[)
    output = MAP.reshape((orig.shape[1], orig.shape[0]))

    return output
