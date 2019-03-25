from __future__ import print_function
import tensorflow as tf
import numpy as np

import TensorflowUtils as utils
import read_MITSceneParsingData as scene_parsing
import datetime
import BatchDatsetReader as dataset
from six.moves import xrange

import crf_helper as mycrf  # CRF functions
import plot_helper as myplot
#import clothparsing as clothparsing
import cv2

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("batch_size", "2", "batch size for training")
tf.flags.DEFINE_string("logs_dir", "logs/", "path to logs directory")


#tf.flags.DEFINE_string("data_dir", "Data_zoo/MIT_SceneParsing/", "path to dataset")
#tf.flags.DEFINE_string("data_dir", "Data_zoo\\MIT_SceneParsing\\", "path to dataset")
tf.flags.DEFINE_string(
    "data_dir", "Data_zoo\\ClothParsing\\", "path to dataset")


tf.flags.DEFINE_float("learning_rate", "1e-4",
                      "Learning rate for Adam Optimizer")
tf.flags.DEFINE_string("model_dir", "Model_zoo/", "Path to vgg model mat")
tf.flags.DEFINE_bool('debug', "False", "Debug mode: True/ False")
tf.flags.DEFINE_string('mode', "train", "Mode train/ test/ visualize/ crftest")

MODEL_URL = 'http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat'

MAX_ITERATION = int(1e5 + 1)
NUM_OF_CLASSESS = 18  # human parsing  59 #cloth   151  # MIT Scene
IMAGE_SIZE = 224


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
            kernels = utils.get_variable(np.transpose(
                kernels, (1, 0, 2, 3)), name=name + "_w")
            bias = utils.get_variable(bias.reshape(-1), name=name + "_b")
            current = utils.conv2d_basic(current, kernels, bias)
        elif kind == 'relu':
            current = tf.nn.relu(current, name=name)
            if FLAGS.debug:
                utils.add_activation_summary(current)
        elif kind == 'pool':
            current = utils.avg_pool_2x2(current)
        net[name] = current

    # added for resume better
    global_iter_counter = tf.Variable(0, name='global_step', trainable=False)
    net['global_step'] = global_iter_counter
    return net


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
        image_net = vgg_net(weights, processed_image)
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
            conv8, W_t1, b_t1, output_shape=tf.shape(image_net["pool4"]))
        fuse_1 = tf.add(conv_t1, image_net["pool4"], name="fuse_1")

        deconv_shape2 = image_net["pool3"].get_shape()
        W_t2 = utils.weight_variable(
            [4, 4, deconv_shape2[3].value, deconv_shape1[3].value], name="W_t2")
        b_t2 = utils.bias_variable([deconv_shape2[3].value], name="b_t2")
        conv_t2 = utils.conv2d_transpose_strided(
            fuse_1, W_t2, b_t2, output_shape=tf.shape(image_net["pool3"]))
        fuse_2 = tf.add(conv_t2, image_net["pool3"], name="fuse_2")

        shape = tf.shape(image)
        deconv_shape3 = tf.stack(
            [shape[0], shape[1], shape[2], NUM_OF_CLASSESS])
        W_t3 = utils.weight_variable(
            [16, 16, NUM_OF_CLASSESS, deconv_shape2[3].value], name="W_t3")
        b_t3 = utils.bias_variable([NUM_OF_CLASSESS], name="b_t3")
        conv_t3 = utils.conv2d_transpose_strided(
            fuse_2, W_t3, b_t3, output_shape=deconv_shape3, stride=8)

        annotation_pred = tf.argmax(conv_t3, dimension=3, name="prediction")

    return tf.expand_dims(annotation_pred, dim=3), conv_t3, image_net


"""
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
    image = tf.placeholder(tf.float32, shape=(
        None, IMAGE_SIZE, IMAGE_SIZE, 3), name="input_image")
    annotation = tf.placeholder(tf.int32, shape=(
        None, IMAGE_SIZE, IMAGE_SIZE, 1), name="annotation")

    # 2. construct inference network
    pred_annotation, logits, net = inference(image, keep_probability)
    tf.summary.image("input_image", image, max_outputs=2)
    tf.summary.image("ground_truth", tf.cast(
        annotation, tf.uint8), max_outputs=2)
    tf.summary.image("pred_annotation", tf.cast(
        pred_annotation, tf.uint8), max_outputs=2)

    # 3. loss measure
    loss = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                          labels=tf.squeeze(
                                                                              annotation, squeeze_dims=[3]),
                                                                          name="entropy")))

    # for CRF
    if FLAGS.mode == 'crftest' or FLAGS.mode == 'predonly':
        probability = tf.nn.softmax(logits=logits, axis=3)  # the axis!

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
    train_records, valid_records = scene_parsing.read_dataset(
        FLAGS.data_dir)  # now it is ignored and hard coded to 10kdataset
    print("data dir:", FLAGS.data_dir)
    print("***** mode:", FLAGS.mode)
    print("train_records length :", len(train_records))
    print("valid_records length :", len(valid_records))

    print("Setting up dataset reader")
    image_options = {'resize': True, 'resize_size': IMAGE_SIZE}
    if FLAGS.mode == 'train':
        print("Loading Training Images.....")
        train_dataset_reader = dataset.BatchDatset(
            train_records, image_options)
    print("Loading Validation Images.....")
    validation_dataset_reader = dataset.BatchDatset(
        valid_records, image_options)

    # for simply check the memory size
    #x = input()
    # exit()

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

        global_step = sess.run(net['global_step'])

        for itr in xrange(global_step, MAX_ITERATION):
            # 6.1 load train and GT images
            train_images, train_annotations = train_dataset_reader.next_batch(
                FLAGS.batch_size)
            #print("train_image:", train_images.shape)
            #print("annotation :", train_annotations.shape)

            feed_dict = {image: train_images,
                         annotation: train_annotations, keep_probability: 0.85}
            # 6.2 traininging
            sess.run(train_op, feed_dict=feed_dict)

            if itr % 10 == 0:
                train_loss, summary_str = sess.run(
                    [loss, summary_op], feed_dict=feed_dict)
                print("Step: %d, Train_loss:%g" % (itr, train_loss))
                summary_writer.add_summary(summary_str, itr)

            if itr % 500 == 0:
                valid_images, valid_annotations = validation_dataset_reader.next_batch(
                    FLAGS.batch_size)
                valid_loss = sess.run(loss, feed_dict={image: valid_images, annotation: valid_annotations,
                                                       keep_probability: 1.0})
                print("%s ---> Validation_loss: %g" %
                      (datetime.datetime.now(), valid_loss))
                global_step = sess.run(net['global_step'])
                saver.save(sess, FLAGS.logs_dir + "model.ckpt",
                           global_step=global_step)

    # test-mode
    elif FLAGS.mode == "visualize":

        valid_images, valid_annotations = validation_dataset_reader.get_random_batch(
            FLAGS.batch_size)
        pred = sess.run(pred_annotation, feed_dict={image: valid_images, annotation: valid_annotations,
                                                    keep_probability: 1.0})
        valid_annotations = np.squeeze(valid_annotations, axis=3)
        pred = np.squeeze(pred, axis=3)

        for itr in range(FLAGS.batch_size):
            utils.save_image(valid_images[itr].astype(
                np.uint8), FLAGS.logs_dir, name="inp_" + str(5+itr))
            utils.save_image(valid_annotations[itr].astype(
                np.uint8), FLAGS.logs_dir, name="gt_" + str(5+itr))
            utils.save_image(pred[itr].astype(np.uint8),
                             FLAGS.logs_dir, name="pred_" + str(5+itr))
            print("Saved image: %d" % itr)

    elif FLAGS.mode == "test":  # heejune added

        validation_dataset_reader.reset_batch_offset(0)
        for itr1 in range(validation_dataset_reader.get_num_of_records()//2):
            valid_images, valid_annotations = validation_dataset_reader.next_batch(
                FLAGS.batch_size)
            pred = sess.run(pred_annotation, feed_dict={image: valid_images, annotation: valid_annotations,
                                                        keep_probability: 1.0})
            valid_annotations = np.squeeze(valid_annotations, axis=3)
            pred = np.squeeze(pred, axis=3)

            for itr2 in range(FLAGS.batch_size):
                utils.save_image(valid_images[itr2].astype(
                    np.uint8), FLAGS.logs_dir, name="inp_" + str(itr1*2+itr2))
                utils.save_image(valid_annotations[itr2].astype(
                    np.uint8), FLAGS.logs_dir, name="gt_" + str(itr1*2+itr2))
                utils.save_image(pred[itr2].astype(
                    np.uint8), FLAGS.logs_dir, name="pred_" + str(itr1*2 + itr2))
                print("Saved image: %d" % (itr1*2 + itr2))

    elif FLAGS.mode == "crftest":  # Tuan added for CRF postprocessing

        accuracies = np.zeros(
            (validation_dataset_reader.get_num_of_records(), 3, 2))
        nFailed = 0
        validation_dataset_reader.reset_batch_offset(0)
        for itr1 in range(validation_dataset_reader.get_num_of_records()//2):
            valid_images, valid_annotations = validation_dataset_reader.next_batch(
                FLAGS.batch_size)

            # pred = sess.run(pred_annotation, feed_dict={image: valid_images, annotation: valid_annotations,
            #                                        keep_probability: 1.0})

            predprob, pred = sess.run([probability, pred_annotation], feed_dict={image: valid_images, annotation: valid_annotations,
                                                                                 keep_probability: 1.0})
            np.set_printoptions(threshold=10)
            # print("Prediction Probability:", predprob)
            # print("Preprob shape:", predprob.shape)
            valid_annotations = np.squeeze(valid_annotations, axis=3)
            pred = np.squeeze(pred)
            predprob = np.squeeze(predprob)

            # @TODO: convert np once not repeatedly
            for itr2 in range(FLAGS.batch_size):
                # if(itr1 * 2 + itr2 <= 831):
                #    continue

                # 1. run CRF
                #print("Output file: ", FLAGS.logs_dir + "crf_" + str(itr1*2+itr2))
                #crfimage, crfoutput = crf(valid_images[itr2].astype(np.uint8),pred[itr2].astype(np.uint8))
                crfwithlabeloutput = mycrf.crf_with_labels(valid_images[itr2].astype(
                    np.uint8), pred[itr2].astype(np.uint8), NUM_OF_CLASSESS)
                crfwithprobsoutput = mycrf.crf_with_probs(
                    valid_images[itr2].astype(np.uint8), predprob[itr2], NUM_OF_CLASSESS)
                # 2. show result display
                if False:
                    print("orig:", valid_images[itr2].dtype)
                    print(
                        "annot:", valid_annotations[itr2].dtype, " shape:", valid_annotations[itr2].shape)
                    print("pred:", pred[itr2].dtype,
                          " shape:", pred[itr2].shape)
                    print("crfwithlabel:", crfwithlabeloutput.dtype)
                    print("crfwithprobs:", crfwithprobsoutput.dtype)

                orignal = valid_images[itr2].astype(np.uint8)
                groundtruth = valid_annotations[itr2].astype(np.uint8)
                fcnpred = pred[itr2].astype(np.uint8)
                crfwithlabelpred = crfwithlabeloutput.astype(np.uint8)
                crfwithprobspred = crfwithprobsoutput.astype(np.uint8)

                if False:
                    savefile = FLAGS.logs_dir + "result/error_result_" + \
                        str(itr1 * 2 + itr2) + ".png"
                    myplot.showResultImages(
                        orignal, groundtruth, fcnpred, crfwithlabelpred, crfwithprobspred)  # , savefile)

                # 3. Calculate confusion matrix between gtimage and prediction image and store to file
                pred_confusion_matrix = mycrf.calcConfussionMat(
                    groundtruth, fcnpred, NUM_OF_CLASSESS)
                crfwithlabelpred_confusion_matrix = mycrf.calcConfussionMat(
                    groundtruth, crfwithlabelpred, NUM_OF_CLASSESS)
                crfwithprobspred_confusion_matrix = mycrf.calcConfussionMat(
                    groundtruth, crfwithprobspred, NUM_OF_CLASSESS)
                accuracies[itr1*2 +
                           itr2][0] = mycrf.calcuateAccuracy(pred_confusion_matrix, False)
                accuracies[itr1*2 + itr2][1] = mycrf.calcuateAccuracy(
                    crfwithlabelpred_confusion_matrix, False)
                accuracies[itr1*2 + itr2][2] = mycrf.calcuateAccuracy(
                    crfwithprobspred_confusion_matrix, True)
                T_full = 0.9
                T_fgnd = 0.85
                if(accuracies[itr1*2 + itr2][2][1] < T_full or accuracies[itr1*2 + itr2][2][0] < T_fgnd):
                    nFailed += 1
                    print("Failed Image (%d-th): %d" %
                          (nFailed, itr1*2 + itr2))

                if False:
                    np.save(FLAGS.logs_dir + "pred_cross_matrix_" +
                            str(itr1*2 + itr2), pred_confusion_matrix)
                    np.save(FLAGS.logs_dir + "crfwithlabelpred_cross_matrix_" +
                            str(itr1 * 2 + itr2), crfwithlabelpred_confusion_matrix)
                    np.save(FLAGS.logs_dir + "crfwithprobspred_cross_matrix_" +
                            str(itr1 * 2 + itr2), crfwithprobspred_confusion_matrix)

                # 4. saving result
                if True:
                    # filenum = str(itr1 * 2 + itr2 +1) # to number the same as input
                    filenum = str(itr1 * 2 + itr2)  # now we have 0-index image

                    #print(FLAGS.logs_dir + "resultSum_" + filenum)
                    #plt.savefig(FLAGS.logs_dir + "resultSum_" + filenum)
                    utils.save_image(orignal, FLAGS.logs_dir,
                                     name="in_" + filenum)
                    utils.save_image(
                        groundtruth, FLAGS.logs_dir, name="gt_" + filenum)
                    #utils.save_image(fcnpred, FLAGS.logs_dir, name="pred_" + filenum)
                    #utils.save_image(crfwithlabelpred, FLAGS.logs_dir, name="crfwithlabel_" + filenum)
                    utils.save_image(crfwithprobspred,
                                     FLAGS.logs_dir, name="crf_" + filenum)

                    # ---End calculate cross matrix
                    print("Saved image: %s" % filenum)

        np.save(FLAGS.logs_dir + "accuracy", accuracies)

    elif FLAGS.mode == "predonly":  # only predict, no accuracy calcuation when no groundtruth

        nFailed = 0
        validation_dataset_reader.reset_batch_offset(0)
        for itr1 in range(validation_dataset_reader.get_num_of_records()//2):
            valid_images, _ = validation_dataset_reader.next_batch(
                FLAGS.batch_size)

            predprob, pred = sess.run([probability, pred_annotation], feed_dict={
                                      image: valid_images, keep_probability: 1.0})
            np.set_printoptions(threshold=10)
            #print("Prediction Probability:", predprob)
            #print("Preprob shape:", predprob.shape)
            pred = np.squeeze(pred)
            predprob = np.squeeze(predprob)

            # @TODO: convert np once not repeatedly
            for itr2 in range(FLAGS.batch_size):
                # if(itr1 * 2 + itr2 <= 831):
                #    continue

                # 1. run CRF
                #print("Output file: ", FLAGS.logs_dir + "crf_" + str(itr1*2+itr2))
                #crfimage, crfoutput = crf(valid_images[itr2].astype(np.uint8),pred[itr2].astype(np.uint8))
                crfwithlabeloutput = mycrf.crf_with_labels(valid_images[itr2].astype(
                    np.uint8), pred[itr2].astype(np.uint8), NUM_OF_CLASSESS)
                crfwithprobsoutput = mycrf.crf_with_probs(
                    valid_images[itr2].astype(np.uint8), predprob[itr2], NUM_OF_CLASSESS)
                # 2. show result display
                if False:
                    print("orig:", valid_images[itr2].dtype)
                    print(
                        "annot:", valid_annotations[itr2].dtype, " shape:", valid_annotations[itr2].shape)
                    print("pred:", pred[itr2].dtype,
                          " shape:", pred[itr2].shape)
                    print("crfwithlabel:", crfwithlabeloutput.dtype)
                    print("crfwithprobs:", crfwithprobsoutput.dtype)

                orignal = valid_images[itr2].astype(np.uint8)
                fcnpred = pred[itr2].astype(np.uint8)
                crfwithlabelpred = crfwithlabeloutput.astype(np.uint8)
                crfwithprobspred = crfwithprobsoutput.astype(np.uint8)

                if False:
                    savefile = FLAGS.logs_dir + "result/error_result_" + \
                        str(itr1 * 2 + itr2) + ".png"
                    myplot.showResultImages(
                        orignal, groundtruth, fcnpred, crfwithlabelpred, crfwithprobspred)  # , savefile)

                # 4. saving result
                if True:
                    # filenum = str(itr1 * 2 + itr2 +1) # to number the same as input
                    filenum = str(itr1 * 2 + itr2)  # now we have 0-index image

                    #print(FLAGS.logs_dir + "resultSum_" + filenum)
                    #plt.savefig(FLAGS.logs_dir + "resultSum_" + filenum)
                    utils.save_image(orignal, FLAGS.logs_dir,
                                     name="in_" + filenum)
                    #utils.save_image(fcnpred, FLAGS.logs_dir, name="pred_" + filenum)
                    #utils.save_image(crfwithlabelpred, FLAGS.logs_dir, name="crfwithlabel_" + filenum)
                    utils.save_image(crfwithprobspred,
                                     FLAGS.logs_dir, name="crf_" + filenum)

                    # ---End calculate cross matrix
                    print("Saved image: %s" % filenum)


if __name__ == "__main__":
    tf.app.run()
