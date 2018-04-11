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

NUM_CLASSES = 10

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


def cnn_structure(input_x, input_y):
    with tf.variable_scope("conv1") as scope:


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













