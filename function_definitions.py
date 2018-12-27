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
import time

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


def mode_visualize(sess, FLAGS, TEST_DIR, validation_dataset_reader, pred_annotation, image, annotation, keep_probability):
    if not os.path.exists(TEST_DIR):
            os.makedirs(TEST_DIR)
            
    valid_images, valid_annotations = validation_dataset_reader.get_random_batch(
        FLAGS.batch_size)
    pred = sess.run(
        pred_annotation,
        feed_dict={
            image: valid_images,
            annotation: valid_annotations,
            keep_probability: 1.0})
    valid_annotations = np.squeeze(valid_annotations, axis=3)
    pred = np.squeeze(pred, axis=3)

    for itr in range(FLAGS.batch_size):
            utils.save_image(valid_images[itr].astype(
                np.uint8), FLAGS.logs_dir + "Image/", name="inp_" + str(5 + itr))
            utils.save_image(valid_annotations[itr].astype(
                np.uint8) * 255 / 18, FLAGS.logs_dir + "Image/", name="gt_" + str(5 + itr))
            utils.save_image(pred[itr].astype(
                np.uint8) * 255 / 18, FLAGS.logs_dir + "Image/", name="pred_" + str(5 + itr))
            print("Saved image: %d" % itr)


def mode_test(sess, FLAGS, TEST_DIR, validation_dataset_reader, pred_annotation, image, annotation, keep_probability, logits):
    print(">>>>>>>>>>>>>>>>Test mode")
    start = time.time()

    if not os.path.exists(TEST_DIR):
        os.makedirs(TEST_DIR)

    crossMats = list()
    mIOU_all = list()
    mFWIOU_all = list()
    validation_dataset_reader.reset_batch_offset(0)
    pred_e = list()

    for itr1 in range(validation_dataset_reader.get_num_of_records() // FLAGS.batch_size):

        valid_images, valid_annotations = validation_dataset_reader.next_batch(
            FLAGS.batch_size)
        pred, logits1 = sess.run([pred_annotation, logits],
                                 feed_dict={image: valid_images, annotation: valid_annotations,
                                            keep_probability: 1.0})
        valid_annotations = np.squeeze(valid_annotations, axis=3)
        #logits1 = np.squeeze(logits1)
        pred = np.squeeze(pred)
        print("logits shape:", logits1.shape)
        np.set_printoptions(threshold=np.inf)
        # print("logits:", logits)

        for itr2 in range(FLAGS.batch_size):

            #print("Output file: ", FLAGS.logs_dir + "crf_" + str(itr1 * 2 + itr2))
            #crfoutput = fd.crf(valid_images[itr2].astype(np.uint8), logits1[itr2])

            fig = plt.figure()
            pos = 240 + 1
            plt.subplot(pos)
            plt.imshow(valid_images[itr2].astype(np.uint8))
            plt.axis('off')
            plt.title('Original')

            pos = 240 + 2
            plt.subplot(pos)
            plt.imshow(
                valid_annotations[itr2].astype(
                    np.uint8),
                cmap=plt.get_cmap('nipy_spectral'))
            plt.axis('off')
            plt.title('GT')

            pos = 240 + 3
            plt.subplot(pos)
            plt.imshow(
                pred[itr2].astype(
                    np.uint8),
                cmap=plt.get_cmap('nipy_spectral'))
            plt.axis('off')
            plt.title('Prediction')

            pos = 240 + 4
            plt.subplot(pos)
            #plt.imshow(crfoutput, cmap=plt.get_cmap('nipy_spectral'))
            plt.axis('off')
            plt.title('CRFPostProcessing')

            pos = 240 + 6
            plt.subplot(pos)
            ret, errorImage = cv2.threshold(
                cv2.absdiff(
                    pred[itr2].astype(
                        np.uint8), valid_annotations[itr2].astype(
                        np.uint8)), 0.5, 255, cv2.THRESH_BINARY)
            plt.imshow(errorImage, cmap=plt.get_cmap('gray'))
            plt.axis('off')
            plt.title('Pred Error:' + str(np.count_nonzero(errorImage)))
            pred_e.append(np.count_nonzero(errorImage))

            crossMat = fd._calcCrossMat(
                valid_annotations[itr2].astype(
                    np.uint8), pred[itr2].astype(
                    np.uint8), NUM_OF_CLASSESS)
            crossMats.append(crossMat)
            # print(crossMat)
            IoUs = fd._calcIOU(
                valid_annotations[itr2].astype(
                    np.uint8), pred[itr2].astype(
                    np.uint8), NUM_OF_CLASSESS)
            mIOU_all.append(IoUs)

            # Frequency weighted mIoUs
            FWIoUs, mFWIoU = fd._calcFrequencyWeightedIOU(
                valid_annotations[itr2].astype(
                    np.uint8), pred[itr2].astype(
                    np.uint8), NUM_OF_CLASSESS)
            mFWIOU_all.append(mFWIoU)

            #crfoutput = cv2.normalize(crfoutput, None, 0, 255, cv2.NORM_MINMAX)
            valid_annotations[itr2] = cv2.normalize(
                valid_annotations[itr2], None, 0, 255, cv2.NORM_MINMAX)

            pos = 240 + 8
            plt.subplot(pos)
            #ret, errorImage = cv2.threshold(cv2.absdiff(crfoutput.astype(np.uint8), valid_annotations[itr2].astype(np.uint8)), 10, 255, cv2.THRESH_BINARY)
            plt.imshow(errorImage, cmap=plt.get_cmap('gray'))
            plt.axis('off')
            plt.title('CRF Error:' + str(np.count_nonzero(errorImage)))

            # np.set_printoptions(threshold=np.inf)

            # plt.show()

            np.savetxt(FLAGS.logs_dir +
                       "Image/Crossmatrix" +
                       str(itr1 *
                           FLAGS.batch_size +
                           itr2) +
                       ".csv", crossMat, fmt='%4i', delimiter=',')
            np.savetxt(FLAGS.logs_dir +
                       "Image/IoUs" +
                       str(itr1 *
                           FLAGS.batch_size +
                           itr2) +
                       ".csv", IoUs, fmt='%4f', delimiter=',')
            np.savetxt(FLAGS.logs_dir +
                       "Image/FWIoUs" +
                       str(itr1 *
                           FLAGS.batch_size +
                           itr2) +
                       ".csv", FWIoUs, fmt='%4f', delimiter=',')
            plt.savefig(FLAGS.logs_dir + "Image/resultSum_" +
                        str(itr1 * FLAGS.batch_size + itr2))
            # ---------------------------------------------
            utils.save_image(valid_images[itr2].astype(np.uint8), FLAGS.logs_dir + "Image/",
                             name="inp_" + str(itr1 * FLAGS.batch_size + itr2))
            utils.save_image(valid_annotations[itr2].astype(np.uint8), FLAGS.logs_dir + "Image/",
                             name="gt_" + str(itr1 * FLAGS.batch_size + itr2))
            utils.save_image(pred[itr2].astype(np.uint8),
                             FLAGS.logs_dir + "Image/",
                             name="pred_" + str(itr1 * 2 + itr2))
            #utils.save_image(crfoutput, FLAGS.logs_dir + "Image/", name="crf_" + str(itr1 * 2 + itr2))

            plt.close('all')
            print("Saved image: %d" % (itr1 * FLAGS.batch_size + itr2))
            # save list of error to file

    with open(FLAGS.logs_dir + 'pred_e.txt', 'w') as file:
        for error in pred_e:
            file.write("%i\n" % error)

    np.savetxt(
        FLAGS.logs_dir +
        "Crossmatrix.csv",
        np.sum(
            crossMats,
            axis=0),
        fmt='%4i',
        delimiter=',')
    np.savetxt(
        FLAGS.logs_dir +
        "mIoUs" +
        ".csv",
        np.mean(
            mIOU_all,
            axis=0),
        fmt='%4f',
        delimiter=',')
    np.savetxt(
        FLAGS.logs_dir +
        "mFWIoUs" +
        ".csv",
        mFWIOU_all,
        fmt='%4f',
        delimiter=',')
        
    end = time.time()
    print("Learning time:", end - start, "seconds")

        
def mode_train(sess, FLAGS, net, train_dataset_reader, validation_dataset_reader, train_records, pred_annotation, image, annotation, keep_probability, logits, train_op, loss, summary_op, summary_writer, DISPLAY_STEP=300):
    
    start = time.time()

    valid = list()
    step = list()
    lo = list()

    global_step = sess.run(net['global_step'])
    global_step = 0
    MAX_ITERATION = round(
        (len(train_records) //
         FLAGS.batch_size) *
        FLAGS.training_epochs)
    print(
        "No. of maximum steps:",
        MAX_ITERATION,
        " Training epochs:",
        FLAGS.training_epochs)

    for itr in xrange(global_step, MAX_ITERATION):
        # 6.1 load train and GT images
        train_images, train_annotations = train_dataset_reader.next_batch(
            FLAGS.batch_size)
        #print("train_image:", train_images.shape)
        #print("annotation :", train_annotations.shape)

        feed_dict = {
            image: train_images,
            annotation: train_annotations,
            keep_probability: 0.85}

        # 6.2 training
        sess.run(train_op, feed_dict=feed_dict)

        if itr % 10 == 0:
            train_loss, summary_str = sess.run(
                [loss, summary_op], feed_dict=feed_dict)
            print("Step: %d, Train_loss:%g" % (itr, train_loss))
            summary_writer.add_summary(summary_str, itr)
            if itr % DISPLAY_STEP == 0 and itr != 0:
                lo.append(train_loss)

        if itr % DISPLAY_STEP == 0 and itr != 0:
            valid_images, valid_annotations = validation_dataset_reader.next_batch(
                FLAGS.batch_size)
            valid_loss = sess.run(
                loss,
                feed_dict={
                    image: valid_images,
                    annotation: valid_annotations,
                    keep_probability: 1.0})
            print(
                "%s ---> Validation_loss: %g" %
                (datetime.datetime.now(), valid_loss))
            global_step = sess.run(net['global_step'])
            saver.save(
                sess,
                FLAGS.logs_dir +
                "model.ckpt",
                global_step=global_step)

            valid.append(valid_loss)
            step.append(itr)
            # print("valid", valid, "step", step)

            plt.plot(step, valid)
            plt.ylabel("Loss")
            plt.xlabel("Step")
            plt.title('Validation Loss')
            plt.savefig(FLAGS.logs_dir + "validation_loss.jpg")

            plt.clf()
            plt.plot(step, lo)
            plt.title('Training Loss')
            plt.ylabel("Loss")
            plt.xlabel("Step")
            plt.savefig(FLAGS.logs_dir + "training_loss.jpg")

            plt.clf()

            plt.plot(step, lo)
            plt.plot(step, valid)
            plt.ylabel("Loss")
            plt.xlabel("Step")
            plt.title('Result')
            plt.legend(['Training Loss', 'Validation Loss'],
                       loc='upper right')
            plt.savefig(FLAGS.logs_dir + "merged_loss.jpg")

    end = time.time()
    print("Learning time:", end - start, "seconds")

