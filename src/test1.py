import numpy as np
import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
FLAGS = tf.app.flags.FLAGS

'''
50,000 (train+validation), 10,000 test image
32x32 (1024) images, 3 channels
binary format: [1,1024,1024,1024]
'''

# tf.app.flags.DEFINE_integer('batch_size', 128, "batch size")
DATA_DIR = os.path.join(os.getcwd(), '..', 'cifar-10-batches-bin')

NUM_EPOCHS = 300
NUM_CLASS = 10
NUM_EXAMPLE_TRAIN = 5000
BATCH_SIZE = 250
learning_rate = 0.0001
keep_prob = tf.placeholder(tf.float32)


def read_input(file_list):
    reader = tf.FixedLengthRecordReader(record_bytes=3073)
    key, value = reader.read(file_list)

    record_bytes = tf.decode_raw(value, tf.uint8)
    label = tf.cast(tf.strided_slice(record_bytes, [0], [1]), tf.int32)

    reshaped_bytes = tf.reshape(
        tf.strided_slice(record_bytes, [0], [3072]), [3, 32, 32]
    )
    # [3,32,32] -> [32,32,3]
    image_data = tf.transpose(reshaped_bytes, [1,2,0])

    return image_data, label

def generate_input(image, label, min_list, batch_size, shuffle):
    num_preprocess_threads = 16
    if shuffle:
        images, label_batch = tf.train.shuffle_batch([image, label], batch_size=batch_size,
        num_threads=num_preprocess_threads, capacity=min_list + 3 * batch_size, min_after_dequeue=min_list)
        return images, tf.reshape(label_batch, [batch_size] )
    else:
        images, label_batch = tf.train.batch([image, label], batch_size=batch_size, num_threads=num_preprocess_threads, capacity=min_list + 3 * batch_size )
        return images, tf.reshape(label_batch, [batch_size] )

'''for training'''
def distorted_input(data_dir, batch_size, mode):
    # res = unpickle( os.path.join(DATA_DIR, 'data_batch_1'))
    # read_input(res[b'data'], res[b'labels'], FLAGS.batch_size)

    filename = ''
    if mode == 'train':
        filenames = [os.path.join(data_dir, 'data_batch_'+str(i)+'.bin') for i in range(1,2) ]
    else:
        filenames = [os.path.join(data_dir, 'data_batch_5.bin') ]

    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)

    file_list = tf.train.string_input_producer(filenames, num_epochs=NUM_EPOCHS)

    image_data, label  = read_input(file_list)
    
    if mode == 'test':
        reshaped_image = tf.cast(image_data, tf.float32)
        reshaped_image.set_shape([32, 32, 3])
        label.set_shape([1])
        return generate_input(reshaped_image, label, 10000, batch_size, shuffle=False)

    with tf.name_scope('preprocess'):
        # read_input_data = read_input(file_list)
        reshaped_image = tf.cast(image_data, tf.float32)
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
        label.set_shape([1])
        # random shuffling
        min_fraction_example = 0.4
        min_list_examples = int(NUM_EXAMPLE_TRAIN * min_fraction_example)
        print ('Filling queue with %d CIFAR images' % min_list_examples)
        
        return generate_input(float_image, label, min_list_examples, batch_size, shuffle=True)


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
    return tf.nn.relu(tf.nn.bias_add(conv, bias), name=name)


def cnn_network(input_x, mode):

    with tf.variable_scope("layer1", reuse=tf.AUTO_REUSE):
        # [b,32,32,3]
        filters1 = _weighted_variable([3,3,3,64])
        conv1 = _conv2d(input_x, filters1, [1,1,1,1])
        conv1 = _conv2d(input_x, filters1, [1,1,1,1])
        bias1 = _bias_variable([64])
        activ1 = _activation(conv1, bias1)
        # [b,16,16,64]
        pool1 = _pool(activ1, ksize=[1,2,2,1], strides=[1,2,2,1])

    with tf.variable_scope("layer2", reuse=tf.AUTO_REUSE):
        # [b,16,16,64]
        filters2 = _weighted_variable([3,3,64,64])
        conv2 = _conv2d(pool1, filters2, [1,1,1,1])
        conv2 = _conv2d(pool1, filters2, [1,1,1,1])
        bias2 = _bias_variable([64])
        activ2 = _activation(conv2, bias2)
        # [b,16,16,64]
        pool2 = _pool(activ2, ksize=[1,2,2,1], strides=[1,1,1,1])

    with tf.variable_scope("layer3", reuse=tf.AUTO_REUSE):
        # [b,16,16,96]
        filters3 = _weighted_variable([3,3,64,96])
        conv3 = _conv2d(pool2, filters3, [1,1,1,1])
        conv3 = _conv2d(pool2, filters3, [1,1,1,1])
        bias3 = _bias_variable([96])
        activ3 = _activation(conv3, bias3)
        # [b,16,16,96]
        pool3 = _pool(activ3, ksize=[1,2,2,1], strides=[1,2,2,1])

    with tf.variable_scope("fc1", reuse=tf.AUTO_REUSE):
        flat = tf.contrib.layers.flatten(pool3)
        
        # [b,384]
        fc1 = tf.layers.dense(flat, 384)
        drop5 = tf.nn.dropout(fc1, keep_prob=0.5) #keep_prob usually 0.5 or 0.3

    with tf.variable_scope("fc2", reuse=tf.AUTO_REUSE):
        flat = tf.contrib.layers.flatten(fc1)
        
        # [b,768]
        fc2 = tf.layers.dense(flat, 768)
        drop6 = tf.nn.dropout(fc2, keep_prob=0.5) #keep_prob usually 0.5 or 0.3

    with tf.variable_scope("output_layer", reuse=tf.AUTO_REUSE):
        #softmax = tf.nn.softmax(tf.add(tf.matmul(drop4, weight5), bias5)) # choose which one
        if (mode == "train"):
            out = tf.layers.dense(drop6, NUM_CLASS)
        else:
            out = tf.layers.dense(fc2, NUM_CLASS)
        # [b,10]
        softmax = tf.nn.softmax(out)

    return softmax


# loss

def loss(input_x, input_y):
    with tf.name_scope('loss'):
        input_y = tf.cast(input_y, tf.int32)
        cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=input_y, logits=input_x)
        # mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    return cross_entropy

# train
global_step = tf.train.get_or_create_global_step()
def train(losses, learning_rate):
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(losses, global_step=global_step)
    
    return train_op

# accuracy
def accuracy(input_x, input_y):
    input_x = tf.cast(input_x, dtype=tf.int32)
    accuracy = tf.reduce_mean(
        tf.cast(tf.equal( input_x , input_y ),
        dtype=tf.float32)
        )

    return accuracy


def model_fn(features,labels, mode):

    logits_train = cnn_network(features, mode='train')
    logits_test = cnn_network(features, mode='test')

    predict_class = tf.argmax(logits_test, axis=1)
    # predict_prob = tf.nn.softmax(logits_test)

    # if mode == tf.estimator.ModeKeys.PREDICT:
    #     return tf.estimator.EstimatorSpec(mode, predictions=predict_class)

    if mode == 'train':
        loss_op = loss(logits_train, labels)
        train_op = train(loss_op, learning_rate=learning_rate)
        return loss_op
        
    if mode == 'test':
        accuracy_op = accuracy(predict_class, labels)
        return accuracy_op

    # estimate_specs = tf.estimator.EstimatorSpec(
    #     mode=mode,
    #     predictions=predict_class,
    #     loss=loss,
    #     train_op=train_op,
    #     eval_metric_ops={'accuracy': accuracy_op}
    # )

    # return loss_op, train_op, accuracy_op, logits_test, predict_class


# === MAIN ===


# losses, train, acc, aa,bb = model_fn(image_data, label, 'train')
# softmax = cnn_network(image_data, 'train')

# input_fn = {'features': image_data, 'labels': label}
# model = tf.estimator.Estimator(model_fn)
#
# training = model.train( input_fn, max_steps=100)

count = 0

image_data_train, label_train = distorted_input(DATA_DIR, BATCH_SIZE, 'train')
losses = model_fn(image_data_train, label_train, 'train')

image_data_test, label_test = distorted_input(DATA_DIR, BATCH_SIZE, 'test')
accuracy = model_fn(image_data_test, label_test, 'test')

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    # writer = tf.summary.FileWriter("logs/", sess.graph)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
        while not coord.should_stop():

            count +=1
            # print(sess.run(tf.train.get_global_step()))
            
            print('batch:', count, 'loss:',sess.run(losses))
            
            if count%5 == 0:
                print('accuracy: ',sess.run(accuracy))
            
            # with tf.variable_scope('layer1', reuse=True):
            #     print(sess.run(tf.get_variable('weights')))
            #     print(sess.run(tf.shape(tf.get_variable('weights'))))
            
            
            # if count>=1:
            #     break
            

    except tf.errors.OutOfRangeError:
        print('Done')
    finally:
        coord.request_stop()

    coord.join(threads)









