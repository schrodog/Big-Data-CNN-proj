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
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
keep_prob = tf.placeholder(tf.float32)

# def unpickle(file):
#     import pickle
#     with open(file, 'rb') as fo:
#         dict = pickle.load(fo, encoding='bytes')
#     return dict

def read_input(file_list):
    reader = tf.FixedLengthRecordReader(record_bytes=3073)
    key, value = reader.read(file_list)

    record_bytes = tf.decode_raw(value, tf.uint8)
    label = tf.cast(tf.strided_slice(record_bytes, [0], [1]), tf.int32)

    depth = tf.reshape(
        tf.strided_slice(record_bytes, [0], [3072]), [3, 32, 32]
    )
    image_data = tf.transpose(depth, [1,2,0])

    return (image_data, label)
	 
def generate_input(image, label, min_list, batch_size, shuffle):
    num_preprocess_threads = 16
	 if shuffle:
        images, label_batch = tf.train.shuffle_batch([image, label], batch_size=batch_size, num_threads=num_preprocess_threads, capacity=min_list + 3 * batch_size, min_after_dequeue=min_list)
	 else:
        images, label_batch = tf.train.shuffle_batch([image, label], batch_size=batch_size, num_threads=num_preprocess_threads, capacity=min_list + 3 * batch_size)
	 return images, tf.reshape(label_batch, [batch_size])

'''for training'''
def distorted_input(data_dir, batch_size):
    # res = unpickle( os.path.join(DATA_DIR, 'data_batch_1'))
    # read_input(res[b'data'], res[b'labels'], FLAGS.batch_size)

    filenames = [os.path.join(DATA_DIR, 'data_batch_'+str(i)+'.bin') for i in range(1,6) ]

    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)

    file_list = tf.train.string_input_producer(filenames)

    image_data, label  = read_input(file_list)


    with tf.name_scope('preprocess'):
        read_input = read_input(file_list)
	     reshaped_image = tf.cast(read_input.image_data, tf.float32)
	     # crop a section of the image
	     distorted_image = tf.random_crop(reshaped_image, [32, 32, 3])
	     # flip the image horizontally
	     distorted_image = tf.image.random_flip_left_right(distorted_image)
	     # randomize the order of the operation
	     distorted_image = tf.image.random_brightness(distorted_image, max_delta=63)
	     distorted_image = tf.image.random_contrast(distorted_image, lower=0.2, upper=1.8)
	     # Subtract off the mean and divide by the variance of the pixels.
        float_image = tf.image.per_image_standardization(distorted_image)
	     # Set the shapes of tensors.
	     float_image.set_shape([32, 32, 3])
	     read_input.label.set_shape([1])
	     # random shuffling
	     min_fraction_of_examples_in_queue = 0.4
        min_list_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN * min_fraction_of_examples_in_queue)
	     print ('Filling queue with %d CIFAR images before starting to train. '
           'This will take a few minutes.' % min_list_examples)
	     return generate_input(float_image, read_input.label, min_list_examples, batch_size, shuffle=True)

	 
# CNN

def _weighted_variable(shape, name='weights'):
    init = tf.random_normal_initializer(mean=0, stddev=0.5 )
    return tf.get_variable(name=name, shape=shape,  initializer=init, dtype=tf.float32)

def _bias_variable(shape, name='bias'):
    init = tf.constant_initializer([0.])
    return tf.get_variable(name=name, shape=shape, initializer=init, dtype=tf.float32)
    
def _conv2d(inputs, kernel, strides, padding='SAME', name='conv'):
    return tf.nn.conv2d(inputs, kernel, strides, padding=padding, name=name)
    
def _pool(inputs, ksize, strides, padding='SAME', name='pool'):
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
        conv2 = _conv2d(input_x, filters2, [1,1,1,1])
        bias2 = _bias_variable([64])
        activ2 = _activation(conv2, bias2)
        pool2 = _pool(activ2, ksize=[1,3,3,1], strides=[1,2,2,1])
    
    with tf.variable_scope("layer3"):
        filters3 = _weighted_variable([3,3,64,64])
        conv3 = _conv2d(input_x, filters3, [1,1,1,1])
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
		  #activ4 = tf.nn.relu(tf.matmul(reshape, weight4) + bias4) # choose which activ4?
		  drop4 = tf.nn.dropout(activ4, keep_prob) #keep_prob usually 0.5 or 0.3

    with tf.variable_scope("output_layer"):
        weight5 = _weighted_variable([384,192])
        bias5 = _bias_variable([NUM_CLASS])
        softmax = tf.add(tf.matmul(activ4, weight5), bias5)
        #softmax = tf.nn.softmax(tf.add(tf.matmul(drop4, weight5), bias5)) # choose which one
    return softmax


# loss

def loss(input_x, input_y):
    with tf.name_scope('loss'):
        input_y = tf.cast(input_y, tf.int32)
        cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=input_y, logits=input_x)
        mean = tf.reduce_mean(cross_entropy, name='cross_entropy')

# train
def train(loss, learning_rate):
    global_step = tf.train.get_or_create_global_step()
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)
    return train_op
	
# accuracy
def accuracy(input_x, input_y):
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(input_x, 1), tf.argmax(input_y, 1)), dtype=tf.float32))
    return accuracy


	
# === MAIN ===

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

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










