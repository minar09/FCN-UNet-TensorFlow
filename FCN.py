from __future__ import print_function
import time
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


FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("batch_size", "32", "batch size for training")
tf.flags.DEFINE_integer(
    "training_epochs",
    "30",
    "number of epochs for training")
tf.flags.DEFINE_string("logs_dir", "logs/FCN/", "path to logs directory")
tf.flags.DEFINE_string("data_dir", "E:/Dataset/Dataset10k/", "path to dataset")
#tf.flags.DEFINE_string("data_dir", "E:/Dataset/MIT_SceneParsing/ADEChallengeData2016/images/", "path to dataset")
tf.flags.DEFINE_float(
    "learning_rate",
    "1e-4",
    "Learning rate for Adam Optimizer")
tf.flags.DEFINE_string("model_dir", "Model_zoo/", "Path to vgg model mat")
tf.flags.DEFINE_bool('debug', "False", "Debug mode: True/ False")
tf.flags.DEFINE_string('mode', "train", "Mode train/ test/ visualize")
#tf.flags.DEFINE_string('mode', "test", "Mode train/ test/ visualize")
#tf.flags.DEFINE_string('mode', "visualize", "Mode train/ test/ visualize")

MODEL_URL = 'http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat'

MAX_ITERATION = int(1e5 + 1001)
NUM_OF_CLASSESS = 18  # human parsing  59 #cloth   151  # MIT Scene
IMAGE_SIZE = 224
DISPLAY_STEP = 300
TEST_DIR = FLAGS.logs_dir + "Image/"


"""
   Train, Test
"""


def inference(image, keep_prob):
    """
    Semantic segmentation network definition
    :param image: input image. Should have values in range 0-255
    :param keep_prob:
    :return:
    """
    # 1. donwload VGG pretrained model from network if not did before
    #    model_data is dictionary for variables from matlab mat file
    print("setting up vgg initialized conv layers ...")
    model_data = utils.get_model_data(FLAGS.model_dir, MODEL_URL)

    mean = model_data['normalization'][0][0][0]
    mean_pixel = np.mean(mean, axis=(0, 1))

    weights = np.squeeze(model_data['layers'])

    processed_image = utils.process_image(image, mean_pixel)

    # 2. construct model graph
    with tf.variable_scope("inference"):
        # 2.1 VGG
        image_net = fd.vgg_net(weights, processed_image)
        conv_final_layer = image_net["conv5_3"]
        #
        pool5 = utils.max_pool_2x2(conv_final_layer)

        W6 = utils.weight_variable([7, 7, 512, 4096], name="W6")
        b6 = utils.bias_variable([4096], name="b6")
        conv6 = utils.conv2d_basic(pool5, W6, b6)
        relu6 = tf.nn.relu(conv6, name="relu6")
        if FLAGS.debug:
            utils.add_activation_summary(relu6)
        relu_dropout6 = tf.nn.dropout(relu6, keep_prob=keep_prob)

        W7 = utils.weight_variable([1, 1, 4096, 4096], name="W7")
        b7 = utils.bias_variable([4096], name="b7")
        conv7 = utils.conv2d_basic(relu_dropout6, W7, b7)
        relu7 = tf.nn.relu(conv7, name="relu7")
        if FLAGS.debug:
            utils.add_activation_summary(relu7)
        relu_dropout7 = tf.nn.dropout(relu7, keep_prob=keep_prob)

        W8 = utils.weight_variable([1, 1, 4096, NUM_OF_CLASSESS], name="W8")
        b8 = utils.bias_variable([NUM_OF_CLASSESS], name="b8")
        conv8 = utils.conv2d_basic(relu_dropout7, W8, b8)
        # annotation_pred1 = tf.argmax(conv8, dimension=3, name="prediction1")

        # now to upscale to actual image size
        deconv_shape1 = image_net["pool4"].get_shape()
        W_t1 = utils.weight_variable(
            [4, 4, deconv_shape1[3].value, NUM_OF_CLASSESS], name="W_t1")
        b_t1 = utils.bias_variable([deconv_shape1[3].value], name="b_t1")
        conv_t1 = utils.conv2d_transpose_strided(
            conv8, W_t1, b_t1, output_shape=tf.shape(
                image_net["pool4"]))
        fuse_1 = tf.add(conv_t1, image_net["pool4"], name="fuse_1")

        deconv_shape2 = image_net["pool3"].get_shape()
        W_t2 = utils.weight_variable(
            [4, 4, deconv_shape2[3].value, deconv_shape1[3].value], name="W_t2")
        b_t2 = utils.bias_variable([deconv_shape2[3].value], name="b_t2")
        conv_t2 = utils.conv2d_transpose_strided(
            fuse_1, W_t2, b_t2, output_shape=tf.shape(
                image_net["pool3"]))
        fuse_2 = tf.add(conv_t2, image_net["pool3"], name="fuse_2")

        shape = tf.shape(image)
        deconv_shape3 = tf.stack(
            [shape[0], shape[1], shape[2], NUM_OF_CLASSESS])
        W_t3 = utils.weight_variable(
            [16, 16, NUM_OF_CLASSESS, deconv_shape2[3].value], name="W_t3")
        b_t3 = utils.bias_variable([NUM_OF_CLASSESS], name="b_t3")
        conv_t3 = utils.conv2d_transpose_strided(
            fuse_2, W_t3, b_t3, output_shape=deconv_shape3, stride=8)

        # prob = tf.nn.softmax(conv_t3, axis =3)
        annotation_pred = tf.argmax(conv_t3, dimension=3, name="prediction")

    return tf.expand_dims(annotation_pred, dim=3), conv_t3, image_net


"""inference
  optimize with trainable paramters (Check which ones)
  loss_val : loss operator (mean(
"""


def train(loss_val, var_list, global_step):
    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
    grads = optimizer.compute_gradients(loss_val, var_list=var_list)
    if FLAGS.debug:
        # print(len(var_list))
        for grad, var in grads:
            utils.add_gradient_summary(grad, var)
    return optimizer.apply_gradients(grads, global_step=global_step)


def main(argv=None):

    # 1. input placeholders
    keep_probability = tf.placeholder(tf.float32, name="keep_probabilty")
    image = tf.placeholder(
        tf.float32,
        shape=(
            None,
            IMAGE_SIZE,
            IMAGE_SIZE,
            3),
        name="input_image")
    annotation = tf.placeholder(
        tf.int32,
        shape=(
            None,
            IMAGE_SIZE,
            IMAGE_SIZE,
            1),
        name="annotation")
    #global_step = tf.Variable(0, trainable=False, name='global_step')

    # 2. construct inference network
    pred_annotation, logits, net = inference(image, keep_probability)
    tf.summary.image("input_image", image, max_outputs=3)
    tf.summary.image(
        "ground_truth",
        tf.cast(
            annotation,
            tf.uint8),
        max_outputs=3)

    tf.summary.image(
        "pred_annotation",
        tf.cast(
            pred_annotation,
            tf.uint8),
        max_outputs=3)

    # 3. loss measure
    loss = tf.reduce_mean(
        (tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits,
            labels=tf.squeeze(
                annotation,
                squeeze_dims=[3]),
            name="entropy")))
    tf.summary.scalar("entropy", loss)

    # 4. optimizing
    trainable_var = tf.trainable_variables()
    if FLAGS.debug:
        for var in trainable_var:
            utils.add_to_regularization_and_summary(var)
    train_op = train(loss, trainable_var, net['global_step'])

    print("Setting up summary op...")
    summary_op = tf.summary.merge_all()

    print("Setting up image reader from ", FLAGS.data_dir, "...")
    train_records, valid_records = scene_parsing.read_dataset(FLAGS.data_dir)
    print("data dir:", FLAGS.data_dir)
    print("train_records length :", len(train_records))
    print("valid_records length :", len(valid_records))

    print("Setting up dataset reader")
    image_options = {'resize': True, 'resize_size': IMAGE_SIZE}
    if FLAGS.mode == 'train':
        train_dataset_reader = dataset.BatchDatset(
            train_records, image_options)
    validation_dataset_reader = dataset.BatchDatset(
        valid_records, image_options)

    sess = tf.Session()

    print("Setting up Saver...")
    saver = tf.train.Saver()
    summary_writer = tf.summary.FileWriter(FLAGS.logs_dir, sess.graph)

    # 5. paramter setup
    # 5.1 init params
    sess.run(tf.global_variables_initializer())
    # 5.2 restore params if possible
    ckpt = tf.train.get_checkpoint_state(FLAGS.logs_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Model restored...")

    # 6. train-mode
    if FLAGS.mode == "train":
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
            # 6.2 traininging

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
                plt.savefig(FLAGS.logs_dir + "FCN_validation_loss.jpg")

                plt.clf()
                plt.plot(step, lo)
                plt.title('Training Loss')
                plt.ylabel("Loss")
                plt.xlabel("Step")
                plt.savefig(FLAGS.logs_dir + "FCN_training_loss.jpg")

                plt.clf()

                plt.plot(step, lo)
                plt.plot(step, valid)
                plt.ylabel("Loss")
                plt.xlabel("Step")
                plt.title('Result')
                plt.legend(['Training Loss', 'Validation Loss'], loc='upper right')
                plt.savefig(FLAGS.logs_dir + "FCN_loss_merged.jpg")

        end = time.time()
        print("Learning time:", end - start, "seconds")

    # test-mode
    elif FLAGS.mode == "visualize":

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

    elif FLAGS.mode == "test":  # heejune added
        print(">>>>>>>>>>>>>>>>Test mode")
        
        if not os.path.exists(TEST_DIR):
            os.makedirs(TEST_DIR)
        
        crossMats = list()
        mIOU_all = list()
        mFWIOU_all = list()
        validation_dataset_reader.reset_batch_offset(0)
        pred_e = list()

        for itr1 in range(
                validation_dataset_reader.get_num_of_records() //
                FLAGS.batch_size):

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


if __name__ == "__main__":
    tf.app.run()
