"""Implementation of sample defense.

This defense loads inception v3 checkpoint and classifies all images
using loaded checkpoint.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
import time
import numpy as np
from scipy.misc import imread
import tensorflow as tf

sys.path.append('slim')

from slim.nets import nets_factory

slim = tf.contrib.slim

tf.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.flags.DEFINE_string(
    'checkpoint_dir', '', 'Path to checkpoint for inception network.')

tf.flags.DEFINE_string(
    'input_dir', '', 'Input directory with images.')

tf.flags.DEFINE_string(
    'output_file', '', 'Output file to save labels.')

tf.flags.DEFINE_integer(
    'image_width', 299, 'Width of each input images.')

tf.flags.DEFINE_integer(
    'image_height', 299, 'Height of each input images.')

tf.flags.DEFINE_integer(
    'batch_size', 4, 'How many images process at one time.')

FLAGS = tf.flags.FLAGS


def load_images(input_dir, batch_shape):
    """Read png images from input directory in batches.

    Args:
      input_dir: input directory
      batch_shape: shape of minibatch array, i.e. [batch_size, height, width, 3]

    Yields:
      filenames: list file names without path of each image
        Lenght of this list could be less than batch_size, in this case only
        first few images of the result are elements of the minibatch.
      images: array with all images from this batch
    """
    images = np.zeros(batch_shape)
    filenames = []
    idx = 0
    batch_size = batch_shape[0]
    for filepath in tf.gfile.Glob(os.path.join(input_dir, '*.png')):
        with tf.gfile.Open(filepath) as f:
            image = imread(f, mode='RGB').astype(np.float) / 255.0
        # Images for inception classifier are normalized to be in [-1, 1] interval.
        images[idx, :, :, :] = image * 2.0 - 1.0
        filenames.append(os.path.basename(filepath))
        idx += 1
        if idx == batch_size:
            yield filenames, images
            filenames = []
            images = np.zeros(batch_shape)
            idx = 0
    if idx > 0:
        yield filenames, images


def mean_filter_gray(images, k=3):
    patches = tf.extract_image_patches(images, [1, k, k, 1], [1, 1, 1, 1], [1, 1, 1, 1], padding='SAME')
    mean = tf.reduce_mean(patches, axis=3, keep_dims=True)
    return mean


def mean_filter(images, k=3):
    stacked = tf.concat(
        [mean_filter_gray(images[:, :, :, i:i + 1], k) for i in range(images.get_shape().as_list()[-1])], axis=3)
    return stacked


def median_filter_gray(images, k=3):
    patches = tf.extract_image_patches(images, [1, k, k, 1], [1, 1, 1, 1], [1, 1, 1, 1], padding='SAME')
    m_idx = int(k * k // 2 + 1)
    top = tf.nn.top_k(patches, m_idx, sorted=True)[0]
    median = tf.slice(top, [0, 0, 0, m_idx - 1], [-1, -1, -1, 1])
    return median


def median_filter(images, k=3):
    stacked = tf.concat(
        [median_filter_gray(images[:, :, :, i:i + 1], k) for i in range(images.get_shape().as_list()[-1])], axis=3)
    return stacked


def inception_preprocess(images, crop_height, crop_width):
    return (images + 1.0) / 2.0


def vgg_preprocess(images, crop_height, crop_width):
    image_height, image_width = FLAGS.image_height, FLAGS.image_width
    offset_height = int((image_height - crop_height) / 2)
    offset_width = int((image_width - crop_width) / 2)

    means = [123.68, 116.779, 103.939]

    images = tf.image.crop_to_bounding_box((images + 1.0) * 255.0 / 2.0, offset_height, offset_width, crop_height,
                                           crop_width)

    return images - means


def model(model_name, x_input, num_classes, preprocess=False, is_training=False, label_offset=0, scope=None,
          reuse=None):
    network_fn = nets_factory.get_network_fn(
        model_name,
        num_classes=num_classes - label_offset,
        is_training=is_training,
        scope=scope,
        reuse=reuse)

    eval_image_size = network_fn.default_image_size
    print('model[' + model_name + ']', 'scope:', scope, 'eval_image_size:', eval_image_size)

    images = x_input
    if preprocess:
        #         images = preprocess_batch(model_name, x_input, eval_image_size, eval_image_size)
        if model_name.startswith('resnet_v1') or model_name.startswith('vgg'):
            images = vgg_preprocess(x_input, eval_image_size, eval_image_size)
        if model_name.startswith('inception_v'):
            images = inception_preprocess(x_input, eval_image_size, eval_image_size)

    logits, _ = network_fn(images)

    return logits


def initialize_uninitialized(sess):
    global_vars = tf.global_variables()
    is_not_initialized = sess.run([tf.is_variable_initialized(var) for var in global_vars])
    not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]

    print([str(i.name) for i in not_initialized_vars])  # only for testing
    if len(not_initialized_vars):
        sess.run(tf.variables_initializer(not_initialized_vars))


def main(_):
    main_start_time = time.time()

    batch_size = FLAGS.batch_size
    image_height = FLAGS.image_height
    image_width = FLAGS.image_width

    ckpt_dir = FLAGS.checkpoint_dir

    batch_shape = [batch_size, image_height, image_width, 3]
    num_classes = 1001

    model_checkpoints = {

        'resnet_v1_50': ckpt_dir + 'resnet_v1_50/resnet_v1_50.ckpt',
        'resnet_v1_101': ckpt_dir + 'resnet_v1_101/resnet_v1_101.ckpt',
        'resnet_v1_152': ckpt_dir + 'resnet_v1_152/resnet_v1_152.ckpt',
        'resnet_v2_50': ckpt_dir + 'resnet_v2_50/resnet_v2_50.ckpt',
        'resnet_v2_101': ckpt_dir + 'resnet_v2_101/resnet_v2_101.ckpt',
        'resnet_v2_152': ckpt_dir + 'resnet_v2_152/resnet_v2_152.ckpt',

        'vgg_16': ckpt_dir + 'vgg_16/vgg_16.ckpt',
        'vgg_19': ckpt_dir + 'vgg_19/vgg_19.ckpt',

        'inception_v3': ckpt_dir + 'inception_v3/inception_v3.ckpt',
        'inception_v4': ckpt_dir + 'inception_v4/inception_v4.ckpt',
        'inception_resnet_v2': ckpt_dir + 'inception_resnet_v2/inception_resnet_v2_2016_08_30.ckpt',

        # adversarially trained model:
        'adv_inception_v3': ckpt_dir + 'adv_inception_v3/adv_inception_v3.ckpt',

        # ensemble adversarially trained model:
        'ens3_adv_inception_v3': ckpt_dir + 'ens3_adv_inception_v3_2017_08_18/ens3_adv_inception_v3.ckpt',
        'ens4_adv_inception_v3': ckpt_dir + 'ens4_adv_inception_v3_2017_08_18/ens4_adv_inception_v3.ckpt',
        'ens_adv_inception_resnet_v2': ckpt_dir + 'ens_adv_inception_resnet_v2/ens_adv_inception_resnet_v2.ckpt',
    }

    tf.logging.set_verbosity(tf.logging.INFO)

    graph_start_time = time.time()
    with tf.Graph().as_default():

        # Prepare graph
        x_input = tf.placeholder(tf.float32, batch_shape, 'x_input')

        models_scopes = {
            'ens_adv_inception_resnet_v2': 'ens_adv_inception_resnet_v2',
            'adv_inception_v3': 'adv_inception_v3',
            # 'inception_v3': 'InceptionV3',
            # 'inception_v4': 'InceptionV4',
            'ens3_adv_inception_v3': 'ens3_adv_inception_v3',
            'ens4_adv_inception_v3': 'ens4_adv_inception_v3',
            # 'resnet_v1_50' : 'resnet_v1_50',
            # 'resnet_v2_50' : 'resnet_v2_50',
            # 'resnet_v2_152' : 'resnet_v2_152',
            # 'resnet_v1_152' : 'resnet_v1_152',
            # 'vgg_19' : 'vgg_19',
            # 'resnet_v1_101' : 'resnet_v1_101',
            # 'inception_resnet_v2' : 'InceptionResnetV2',
            # 'resnet_v2_101' : 'resnet_v2_101',
            # 'vgg_16' : 'vgg_16'
        }

        network_mapping = {
            'ens_adv_inception_resnet_v2': 'inception_resnet_v2',
            'adv_inception_v3': 'inception_v3',
            'ens3_adv_inception_v3': 'inception_v3',
            'ens4_adv_inception_v3': 'inception_v3',
        }

        model_init_start = time.time()

        ############################################
        # Ensemble models inference of fixed image #
        ############################################

        # x_filtered = mean_filter(x_input, k=2)
        x_filtered = median_filter(x_input, k=2)

        probs = []
        for model_name in models_scopes.keys():
            scope = models_scopes[model_name]
            label_offset = 1 if model_name.startswith('resnet_v1') or model_name.startswith('vgg') else 0

            model_name = model_name if model_name not in network_mapping else network_mapping[model_name]

            logits = model(model_name, x_filtered, num_classes, True, label_offset=label_offset, scope=scope)

            if label_offset > 0:
                prob = tf.pad(tf.nn.softmax(logits), tf.constant([[0, 0], [1, 0]]))
                logits = tf.pad(logits, tf.constant([[0, 0], [1, 0]]))
            else:
                prob = tf.nn.softmax(logits)

            probs.append(prob)

        probabilities = tf.add_n(probs) / len(probs)
        predictions = tf.argmax(probabilities, axis=1)

        print('Model initialization time:', (time.time() - model_init_start))

        print('Total graph init time:', (time.time() - graph_start_time))

        with tf.Session() as sess:
            restore_start = time.time()

            checkpoint_mapping = {
                'ens_adv_inception_resnet_v2': 'InceptionResnetV2',
                'adv_inception_v3': 'InceptionV3',
                'ens3_adv_inception_v3': 'InceptionV3',
                'ens4_adv_inception_v3': 'InceptionV3',
                'fs_inception_v3': 'InceptionV3',
                'fs_inception_v3_adv': 'InceptionV3',
            }

            var_lists = []
            for i, model_name in enumerate(models_scopes.keys()):
                scope = models_scopes[model_name]
                mapping_scope = scope if model_name not in checkpoint_mapping else checkpoint_mapping[model_name]

                var_list = {(mapping_scope + v.name[len(scope):][:-2]): v
                            for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)}

                var_lists.append(var_list)

            savers = [tf.train.Saver(var_lists[i]) for i, scope in enumerate(models_scopes.values())]

            for i, model_name in enumerate(models_scopes.keys()):
                savers[i].restore(sess, model_checkpoints[model_name])

            initialize_uninitialized(sess)
            print('Restored in:', (time.time() - restore_start))

            process_start_time = time.time()

            with tf.gfile.Open(FLAGS.output_file, 'w') as out_file:
                for filenames, images in load_images(FLAGS.input_dir, batch_shape):

                    labels = sess.run(predictions, feed_dict={x_input: images})

                    for filename, label in zip(filenames, labels):
                        out_file.write('{0},{1}\n'.format(filename, label))

                    sys.stdout.write('.')
                    sys.stdout.flush()
            print()
            print('processed in:', (time.time() - process_start_time))
    print('Main processed in:', (time.time() - main_start_time))


if __name__ == '__main__':
    tf.app.run()
