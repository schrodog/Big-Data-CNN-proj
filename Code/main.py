import numpy as np
import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
FLAGS = tf.app.flags.FLAGS

'''
50,000 (train+validation), 10,000 test image
32x32 (1024) images, 3 channels


batch_label: training batch 1 of 5
data: [59,43,50,...,140,84,72],
      ...
labels: [6,9,9,4,1, ...]
filenames: ['xx.png', 'xxx.png', ...]
'''

tf.app.flags.DEFINE_integer('batch_size', 128, "batch size")
DATA_DIR = os.path.join(os.getcwd(), '..', 'cifar-10-batches-bin')

NUM_CLASS = 10

# def unpickle(file):
#     import pickle
#     with open(file, 'rb') as fo:
#         dict = pickle.load(fo, encoding='bytes')
#     return dict

def generate_input(file_list):
    reader = tf.FixedLengthRecordReader(record_bytes=3073)
    key, value = reader.read(file_list)

    record_bytes = tf.decode_raw(value, tf.uint8)
    label = tf.cast(tf.strided_slice(record_bytes, [0], [1]), tf.int32)

    depth = tf.reshape(
        tf.strided_slice(record_bytes, [0], [3072]), [3, 32, 32]
    )
    image_data = tf.transpose(depth, [1,2,0])

    return (image_data, label)



# res = unpickle( os.path.join(DATA_DIR, 'data_batch_1'))
# generate_input(res[b'data'], res[b'labels'], FLAGS.batch_size)

filenames = [os.path.join(DATA_DIR, 'data_batch_'+str(i)+'.bin') for i in range(1,6) ]
file_list = tf.train.string_input_producer(filenames)

image_data, label  = generate_input(file_list)


# CNN

def _weighted_variable(shape, name='weights'):
    init = tf.random_normal_initializer(mean=0, stddev=0.5 )
    return tf.get_variable(name=name, shape=shape,  initializer=init, dtype=tf.float32)

def _bias_variable(shape, name='bias'):
    init = tf.constant_initializer([0.])
    return tf.get_variable(name=name, shape=shape, initializer=init, dtype=tf.float32)
    
def _conv2d(inputs, kernel, strides, padding='SAME', name='conv'):
    return tf.nn.conv2d(inputs, kernel, strides, padding=padding, name=name)
    
def _pooling(inputs, ksize, strides, padding='SAME', name='pool'):
    return tf.nn.max_pool(inputs, ksize=ksize, strides=strides, padding=padding, name=name)
    
def _activation(conv, bias, name='activation'):
    return tf.nn.relu(tf.bias_add(conv, bias), name=name)
    

def cnn_structure(input_x, input_y):
    with tf.variable_scope("layer1"):
        filters1 = _weighted_variable([5,5,3,64])
        conv1 = _conv2d(input_x, filters1, [1,1,1,1])
        bias1 = _bias_variable([64])
        activ1 = _activation(conv1, bias1)
        pool1 = _pool(activ1, ksize=[1,3,3,1], strides=[1,2,2,1])
        
    with tf.variable_scope("layer2"):
        filters2 = _weighted_variable([5,5,64,64])
        conv2 = _conv2d(input_x, filters, [1,1,1,1])
        bias2 = _bias_variable([64])
        activ2 = _activation(conv2, bias2)
        pool2 = _pool(activ2, ksize=[1,3,3,1], strides=[1,2,2,1])
    
    with tf.variable_scope("layer3"):
        filters3 = _weighted_variable([3,3,64,64])
        conv3 = _conv2d(input_x, filters, [1,1,1,1])
        bias3 = _bias_variable([64])
        activ3 = _activation(conv2, bias2)
        pool3 = _pool(activ3, ksize=[1,2,2,1], strides=[1,1,1,1])
    
    with tf.variable_scope("layer4"):
        reshape = tf.reshape(pool3, [input_x.shape.as_list()[0], -1])
        depth = reshape.shape[1].value
        # col = tf.shape(pool3)
        
        weight4 = _weighted_variable([depth, 384])
        bias4 = _bias_variable([384])
        activ4 = tf.nn.relu(tf.matmul(pool3, weight4), bias4)

    with tf.variable_scope("output_layer"):
        weight5 = _weighted_variable([384,192])
        bias5 = _bias_variable([NUM_CLASS])
        softmax = tf.add(tf.matmul(activ4, weight5), bias5)
        
    return softmax



# === MAIN ===

init_op = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init_op)

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

try:
    while not coord.should_stop():
        print(sess.run(tf.shape(image_data)))
except tf.errors.OutOfRangeError:
    print('Done')
finally:
    coord.request_stop()

coord.join(threads)
sess.close()

# sess = tf.InteractiveSession()
# tf.train.start_queue_runners(sess=sess)
#
# print(sess.run(tf.shape(image_data)))













