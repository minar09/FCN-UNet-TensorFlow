
# Hide the warning messages about CPU/GPU
import sys
import scipy.io
import zipfile
import tarfile
from six.moves import urllib
import scipy.misc as misc
import numpy as np
import tensorflow as tf
import os
from functools import reduce
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Utils used with tensorflow implemetation


"""
   load data from Matlab mat file
"""


def get_model_data(dir_path, model_url='http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat'):
    # 1. dowload if needed
    maybe_download_and_extract(dir_path, model_url)
    filename = model_url.split("/")[-1]
    filepath = os.path.join(dir_path, filename)
    if not os.path.exists(filepath):
        raise IOError("VGG Model not found!")
    # 2. load data from mat file
    data = scipy.io.loadmat(filepath)
    return data


"""
   Download and etract file if needed
   dirpath: filename
   url_name:  network location to download
   is_xxx : type of file (could check by extenstion of file)
"""


def maybe_download_and_extract(
        dir_path,
        url_name,
        is_tarfile=False,
        is_zipfile=False):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    filename = url_name.split('/')[-1]
    filepath = os.path.join(dir_path, filename)

    # download and etxract it if not yet downloaded
    if not os.path.exists(filepath):
        # 1. download from network
        def _progress(count, block_size, total_size):
            sys.stdout.write(
                '\r>> Downloading %s %.1f%%' %
                (filename,
                 float(
                     count *
                     block_size) /
                    float(total_size) *
                    100.0))
            sys.stdout.flush()

        filepath, _ = urllib.request.urlretrieve(
            url_name, filepath, reporthook=_progress)
        print()
        statinfo = os.stat(filepath)
        print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')
        # 2. extract if needed
        if is_tarfile:
            tarfile.open(filepath, 'r:gz').extractall(dir_path)
        elif is_zipfile:
            with zipfile.ZipFile(filepath) as zf:
                zip_dir = zf.namelist()[0]
                zf.extractall(dir_path)


"""
    Save image
"""


def save_image(image, save_dir, name, mean=None):
    """
    Save image by unprocessing if mean given else just save
    :param mean:
    :param image:
    :param save_dir:
    :param name:
    :return:
    """
    if mean:
        image = unprocess_image(image, mean)
    misc.imsave(os.path.join(save_dir, name + ".png"), image)


def get_variable(weights, name):
    init = tf.constant_initializer(weights, dtype=tf.float32)
    var = tf.get_variable(name=name, initializer=init, shape=weights.shape)
    return var


def weight_variable(shape, stddev=0.02, name=None):
    # print(shape)
    initial = tf.truncated_normal(shape, stddev=stddev)
    if name is None:
        return tf.Variable(initial)
    else:
        return tf.get_variable(name, initializer=initial)


def bias_variable(shape, name=None):
    initial = tf.constant(0.0, shape=shape)
    if name is None:
        return tf.Variable(initial)
    else:
        return tf.get_variable(name, initializer=initial)


def get_tensor_size(tensor):
    from operator import mul
    return reduce(mul, (d.value for d in tensor.get_shape()), 1)


def conv2d_basic(x, W, bias):
    conv = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")
    return tf.nn.bias_add(conv, bias)


def conv2d_strided(x, W, b):
    conv = tf.nn.conv2d(x, W, strides=[1, 2, 2, 1], padding="SAME")
    return tf.nn.bias_add(conv, b)


def _upsample_filters(filters, rate):
    """Upsamples the filters by a factor of rate along the spatial dimensions.
    Args:
      filters: [h, w, in_depth, out_depth]. Original filters.
      rate: An int, specifying the upsampling rate.
    Returns:
      filters_up: [h_up, w_up, in_depth, out_depth]. Upsampled filters with
        h_up = h + (h - 1) * (rate - 1)
        w_up = w + (w - 1) * (rate - 1)
        containing (rate - 1) zeros between consecutive filter values along
        the filters' spatial dimensions.
    """
    if rate == 1:
        return filters
    # [h, w, in_depth, out_depth] -> [in_depth, out_depth, h, w]
    filters_up = np.transpose(filters, [2, 3, 0, 1])
    ker = np.zeros([rate, rate], dtype=np.float32)
    ker[0, 0] = 1
    filters_up = np.kron(filters_up, ker)[:, :, :-(rate - 1), :-(rate - 1)]
    # [in_depth, out_depth, h_up, w_up] -> [h_up, w_up, in_depth, out_depth]
    filters_up = np.transpose(filters_up, [2, 3, 0, 1])
    return filters_up


def conv2d_transpose_strided(x, W, b, output_shape=None, stride=2):
        # print x.get_shape()
        # print W.get_shape()
    if output_shape is None:
        output_shape = x.get_shape().as_list()
        output_shape[1] *= 2
        output_shape[2] *= 2
        output_shape[3] = W.get_shape().as_list()[2]
    # print output_shape
    conv = tf.nn.conv2d_transpose(x, W, output_shape, strides=[
                                  1, stride, stride, 1], padding="SAME")
    #conv = tf.nn.atrous_conv2d_transpose(x, _upsample_filters(W, stride), output_shape, rate=stride, padding="SAME")
    return tf.nn.bias_add(conv, b)


def leaky_relu(x, alpha=0.0, name=""):
    return tf.maximum(alpha * x, x, name)


def max_pool_2x2(x):
    return tf.nn.max_pool(
        x, ksize=[
            1, 2, 2, 1], strides=[
            1, 2, 2, 1], padding="SAME")


def avg_pool_2x2(x):
    return tf.nn.avg_pool(
        x, ksize=[
            1, 2, 2, 1], strides=[
            1, 2, 2, 1], padding="SAME")


def local_response_norm(x):
    return tf.nn.lrn(x, depth_radius=5, bias=2, alpha=1e-4, beta=0.75)


def batch_norm(x, n_out, phase_train, scope='bn', decay=0.9, eps=1e-5):
    """
    Code taken from http://stackoverflow.com/a/34634291/2267819
    """
    with tf.variable_scope(scope):
        beta = tf.get_variable(
            name='beta',
            shape=[n_out],
            initializer=tf.constant_initializer(0.0),
            trainable=True)
        gamma = tf.get_variable(
            name='gamma',
            shape=[n_out],
            initializer=tf.random_normal_initializer(
                1.0,
                0.02),
            trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=decay)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(
            phase_train, mean_var_with_update, lambda: (
                ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, eps)
    return normed


def process_image(image, mean_pixel):
    return image - mean_pixel


def unprocess_image(image, mean_pixel):
    return image + mean_pixel


def bottleneck_unit(
        x,
        out_chan1,
        out_chan2,
        down_stride=False,
        up_stride=False,
        name=None):
    """
    Modified implementation from github ry?!
    """

    def conv_transpose(tensor, out_channel, shape, strides, name=None):
        out_shape = tensor.get_shape().as_list()
        in_channel = out_shape[-1]
        kernel = weight_variable(
            [shape, shape, out_channel, in_channel], name=name)
        shape[-1] = out_channel
        return tf.nn.conv2d_transpose(
            x,
            kernel,
            output_shape=out_shape,
            strides=[
                1,
                strides,
                strides,
                1],
            padding='SAME',
            name='conv_transpose')

    def conv(tensor, out_chans, shape, strides, name=None):
        in_channel = tensor.get_shape().as_list()[-1]
        kernel = weight_variable(
            [shape, shape, in_channel, out_chans], name=name)
        return tf.nn.conv2d(
            x,
            kernel,
            strides=[
                1,
                strides,
                strides,
                1],
            padding='SAME',
            name='conv')

    def bn(tensor, name=None):
        """
        :param tensor: 4D tensor input
        :param name: name of the operation
        :return: local response normalized tensor - not using batch normalization :(
        """
        return tf.nn.lrn(
            tensor,
            depth_radius=5,
            bias=2,
            alpha=1e-4,
            beta=0.75,
            name=name)

    in_chans = x.get_shape().as_list()[3]

    if down_stride or up_stride:
        first_stride = 2
    else:
        first_stride = 1

    with tf.variable_scope('res%s' % name):
        if in_chans == out_chan2:
            b1 = x
        else:
            with tf.variable_scope('branch1'):
                if up_stride:
                    b1 = conv_transpose(
                        x,
                        out_chans=out_chan2,
                        shape=1,
                        strides=first_stride,
                        name='res%s_branch1' %
                        name)
                else:
                    b1 = conv(
                        x,
                        out_chans=out_chan2,
                        shape=1,
                        strides=first_stride,
                        name='res%s_branch1' %
                        name)
                b1 = bn(b1, 'bn%s_branch1' % name, 'scale%s_branch1' % name)

        with tf.variable_scope('branch2a'):
            if up_stride:
                b2 = conv_transpose(
                    x,
                    out_chans=out_chan1,
                    shape=1,
                    strides=first_stride,
                    name='res%s_branch2a' %
                    name)
            else:
                b2 = conv(
                    x,
                    out_chans=out_chan1,
                    shape=1,
                    strides=first_stride,
                    name='res%s_branch2a' %
                    name)
            b2 = bn(b2, 'bn%s_branch2a' % name, 'scale%s_branch2a' % name)
            b2 = tf.nn.relu(b2, name='relu')

        with tf.variable_scope('branch2b'):
            b2 = conv(
                b2,
                out_chans=out_chan1,
                shape=3,
                strides=1,
                name='res%s_branch2b' %
                name)
            b2 = bn(b2, 'bn%s_branch2b' % name, 'scale%s_branch2b' % name)
            b2 = tf.nn.relu(b2, name='relu')

        with tf.variable_scope('branch2c'):
            b2 = conv(
                b2,
                out_chans=out_chan2,
                shape=1,
                strides=1,
                name='res%s_branch2c' %
                name)
            b2 = bn(b2, 'bn%s_branch2c' % name, 'scale%s_branch2c' % name)

        x = b1 + b2
        return tf.nn.relu(x, name='relu')


def add_to_regularization_and_summary(var):
    if var is not None:
        tf.summary.histogram(var.op.name, var)
        tf.add_to_collection("reg_loss", tf.nn.l2_loss(var))


def add_activation_summary(var):
    if var is not None:
        tf.summary.histogram(var.op.name + "/activation", var)
        tf.summary.scalar(var.op.name + "/sparsity", tf.nn.zero_fraction(var))


def add_gradient_summary(grad, var):
    if grad is not None:
        tf.summary.histogram(var.op.name + "/gradient", grad)


def conv(
        inputs,
        filters,
        kernel_size=[
            3,
            3],
    activation=tf.nn.relu,
    l2_reg_scale=None,
        batchnorm_istraining=None):
    if l2_reg_scale is None:
        regularizer = None
    else:
        regularizer = tf.contrib.layers.l2_regularizer(scale=l2_reg_scale)
    conved = tf.layers.conv2d(
        inputs=inputs,
        filters=filters,
        kernel_size=kernel_size,
        padding="same",
        activation=activation,
        kernel_regularizer=regularizer
    )
    if batchnorm_istraining is not None:
        conved = bn(conved, batchnorm_istraining)

    return conved


def bn(inputs, is_training):
    normalized = tf.layers.batch_normalization(
        inputs=inputs,
        axis=-1,
        momentum=0.9,
        epsilon=0.001,
        center=True,
        scale=True,
        training=is_training,
    )
    return normalized


def pool(inputs):
    pooled = tf.layers.max_pooling2d(
        inputs=inputs, pool_size=[
            2, 2], strides=2)
    return pooled


def conv_transpose(inputs, filters, l2_reg_scale=None):
    if l2_reg_scale is None:
        regularizer = None
    else:
        regularizer = tf.contrib.layers.l2_regularizer(scale=l2_reg_scale)
    conved = tf.layers.conv2d_transpose(
        inputs=inputs,
        filters=filters,
        strides=[2, 2],
        kernel_size=[2, 2],
        padding='same',
        activation=tf.nn.relu,
        kernel_regularizer=regularizer
    )
    return conved
