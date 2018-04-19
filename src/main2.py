import numpy as np
import tensorflow as tf
import os

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# FLAGS = tf.app.flags.FLAGS

'''
50,000 (train+validation), 10,000 test image
32x32 (1024) images, 3 channels
binary format: [1,1024,1024,1024]
'''

# tf.app.flags.DEFINE_integer('batch_size', 128, "batch size")
DATA_DIR = os.path.join(os.getcwd(), '..', 'cifar-10-batches-bin')

NUM_EPOCHS = 100
NUM_CLASS = 10
NUM_EXAMPLE_TRAIN = 50000
BATCH_SIZE = 128
learning_rate = 0.0004
DECAY_EPOCH = 300
DECAY_FACTOR = 0.96
# keep_prob = tf.placeholder(tf.float32)


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
        filenames = [os.path.join(data_dir, 'data_batch_'+str(i)+'.bin') for i in range(1,5) ]
    elif mode == 'validation':
        filenames = [os.path.join(data_dir, 'data_batch_5.bin') ]
    elif mode == 'test':
        filenames = [os.path.join(data_dir, 'test_batch.bin') ]

    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)

    if mode == 'train' or mode == 'validation':
        file_list = tf.train.string_input_producer(filenames, num_epochs=NUM_EPOCHS)
    else:
        file_list = tf.train.string_input_producer(filenames, num_epochs=1)

    image_data, label  = read_input(file_list)

    if mode == 'test' or mode == 'validation':
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
    init = tf.truncated_normal(shape, mean=0, stddev=0.05 )
    # init = tf.random_normal_initializer(mean=0, stddev=0.5 )
    return tf.get_variable(name=name, initializer=init, dtype=tf.float32)

def _bias_variable(shape, name='bias'):
    init = tf.constant_initializer([0.0])
    return tf.get_variable(name=name, shape=shape, initializer=init, dtype=tf.float32)

def _conv2d(inputs, kernel, strides, padding='SAME', name='conv'):
    return tf.nn.conv2d(inputs, kernel, strides, padding=padding, name=name)

def _pool(inputs, ksize, strides, padding='SAME', name='pool'):
    return tf.nn.max_pool(inputs, ksize=ksize, strides=strides, padding=padding, name=name)

def _activation(conv, bias, name='activation'):
    return tf.nn.relu(tf.add(conv, bias), name=name)


def cnn_network(input_x, mode):

    with tf.variable_scope("layer1", reuse=tf.AUTO_REUSE):
        # [b,32,32,3]
        filters1 = _weighted_variable([5,5,3,64])
        conv1 = _conv2d(input_x, filters1, [1,1,1,1])
        bias1 = _bias_variable([64])
        activ1 = _activation(conv1, bias1)
        # [b,16,16,64]
        pool1 = _pool(activ1, ksize=[1,3,3,1], strides=[1,2,2,1])
        tf.summary.histogram('layer1', pool1)

    with tf.variable_scope("layer2", reuse=tf.AUTO_REUSE):
        # [b,16,16,64]
        filters2 = _weighted_variable([5,5,64,128])
        conv2 = _conv2d(pool1, filters2, [1,1,1,1])
        bias2 = _bias_variable([128])
        activ2 = _activation(conv2, bias2)
        # [b,16,16,64]
        pool2 = _pool(activ2, ksize=[1,3,3,1], strides=[1,2,2,1])
        tf.summary.histogram('layer2', pool2)

    with tf.variable_scope("layer3", reuse=tf.AUTO_REUSE):
        # [b,16,16,128]
        filters3 = _weighted_variable([3,3,128,128])
        conv3 = _conv2d(pool2, filters3, [1,1,1,1])
        bias3 = _bias_variable([128])
        activ3 = _activation(conv3, bias3)
        # [b,16,16,128]
        pool3 = _pool(activ3, ksize=[1,3,3,1], strides=[1,1,1,1])
        tf.summary.histogram('layer3', pool3)

    # with tf.variable_scope("layer4", reuse=tf.AUTO_REUSE):
    #     # [b,8,8,196]
    #     filters4 = _weighted_variable([3,3,128,256])
    #     conv4 = _conv2d(pool3, filters4, [1,1,1,1])
    #     bias4 = _bias_variable([256])
    #     activ4 = _activation(conv4, bias4)
    #     # [b,8,8,196]
    #     pool4 = _pool(activ4, ksize=[1,2,2,1], strides=[1,1,1,1])
    #     tf.summary.histogram('layer4', pool4)

    with tf.variable_scope("fc1", reuse=tf.AUTO_REUSE):
        n = 8; m = 128
        reshape = tf.reshape(pool2, [-1, n*n*m])
        w_fc1 = _weighted_variable([n*n*m, 1024])
        b_fc1 = _bias_variable([1024])
        # flat = tf.contrib.layers.flatten(pool2)
	    #activ4 = tf.nn.relu(tf.matmul(reshape, weight4) + bias4) # choose which activ4?

        # [b,384]
        activ_fc1 = tf.nn.relu(tf.matmul(reshape, w_fc1) + b_fc1)
        tf.summary.histogram('fc1', activ_fc1)

        # fc1 = tf.layers.dense(flat, 384)
        # drop5 = tf.nn.dropout(fc1, keep_prob=0.5) #keep_prob usually 0.5 or 0.3

    with tf.variable_scope("fc2", reuse=tf.AUTO_REUSE):
        # reshape = tf.reshape(pool4, [input_x.shape.as_list()[0], -1])
        # flat = tf.contrib.layers.flatten(pool2)
        #activ4 = tf.nn.relu(tf.matmul(reshape, weight4) + bias4) # choose which activ4?
        weights4 = _weighted_variable([1024,384])
        bias4 = _bias_variable([384])
        activ4 = tf.nn.relu(tf.matmul(activ_fc1,weights4) + bias4 )
        tf.summary.histogram('fc2', activ4)

        # [b,384]
        # fc1 = tf.layers.dense(flat, 384)
        drop5 = tf.nn.dropout(activ4, keep_prob=0.5) #keep_prob usually 0.5 or 0.3

    with tf.variable_scope("output_layer", reuse=tf.AUTO_REUSE):

        #softmax = tf.nn.softmax(tf.add(tf.matmul(drop4, weight5), bias5)) # choose which one
        # if (mode == "train"):
        # else:
        #     out = tf.layers.dense(fc1, NUM_CLASS)
        # [b,10]
        # out = tf.layers.dense(drop5, NUM_CLASS)
        weight5 = _weighted_variable([384, NUM_CLASS])
        bias5 = _bias_variable([NUM_CLASS])
        softmax = tf.nn.softmax(tf.matmul(drop5, weight5) + bias5)

        tf.summary.histogram('softmax', softmax)


    return softmax


# loss
def loss_fn(input_x, input_y):
    with tf.name_scope('loss'):
        label = tf.cast(input_y, tf.int64)
        cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=label, logits=input_x)
        mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
        tf.summary.scalar('cross_entropy', mean)
    return mean, input_x, input_y

# train
global_step = tf.train.get_or_create_global_step()
def train(losses, learning_rate):
    lr = tf.train.exponential_decay(learning_rate, global_step, DECAY_EPOCH, DECAY_FACTOR, staircase=True)
    tf.summary.scalar('learning rate', lr)
    train_op = tf.train.AdamOptimizer(lr).minimize(losses, global_step=global_step)

    return train_op

# accuracy
def accuracy_fn(input_x, input_y):
    input_x = tf.cast(input_x, dtype=tf.int32)
    accuracy = tf.reduce_mean(
        tf.cast(tf.equal( input_x , input_y ), dtype=tf.float32))
    tf.summary.scalar('accuracy', accuracy)
    return accuracy


def model_fn(features,labels, mode):

    logits_train = cnn_network(features, mode='train')
    logits_test = cnn_network(features, mode='validation')

    predict_class = tf.argmax(logits_test, axis=1)
    # predict_prob = tf.nn.softmax(logits_test)

    if mode == 'train':
        return logits_train, labels
        # loss_op = loss(logits_train, labels)
        # train_op = train(loss_op, learning_rate=learning_rate)

    if mode == 'test':
        accuracy_op = accuracy_fn(predict_class, labels)
        return accuracy_op


# === MAIN ===

# training
def main(argv=None):
    count = 0

    # train
    image_data_train, label_train = distorted_input(DATA_DIR, BATCH_SIZE, 'train')
    logits_train, label_trains = model_fn(image_data_train, label_train, 'train')
    loss_op, lossx, lossy = loss_fn(logits_train, label_trains)
    train_op = train(loss_op, learning_rate=learning_rate)

    # validate
    image_data_valid, label_valid = distorted_input(DATA_DIR, BATCH_SIZE, 'validation')
    accuracy_op = model_fn(image_data_valid, label_valid, 'test')
    # accuracy_op = accuracy(predict_class, label_test2)

    summaries = tf.summary.merge_all()
    saver = tf.train.Saver()
    # loss_op = loss_fn(logits_train, label_train)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        writer = tf.summary.FileWriter("logs/", sess.graph)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)


        try:
            while not coord.should_stop():
                count +=1

                sess.run(train_op)
                summ = sess.run(summaries)
                writer.add_summary(summ, global_step=sess.run(global_step))

                # print(sess.run(lossx), sess.run(lossy))
                # with tf.variable_scope('layer2', reuse=True):
                #     print(sess.run(tf.get_variable('weights')[0,0,0]))
                    # print(sess.run(tf.get_variable('bias')[0]))

                if count % 5 == 0:
                    print('step:', count, 'loss:',sess.run(loss_op))
                        # print(sess.run(tf.shape(tf.get_variable('weights'))))

                if count%50 == 0:
                    print('accuracy: ',sess.run(accuracy_op))

        except tf.errors.OutOfRangeError:
            print('Done')
        finally:
            coord.request_stop()

        coord.join(threads)
        saver_path = saver.save(sess, os.path.join(DATA_DIR,'..', 'model' , 'model2.ckpt'))
        print("Model saved in path: %s" % saver_path)


# testing performance
def eval_fn():
    image_data_test, label_test = distorted_input(DATA_DIR, BATCH_SIZE, 'test')
    predict = cnn_network(image_data_test,'test')

    # top_k metrics
    top_k = tf.nn.in_top_k(predict, label_test, 2)
    top_k = tf.reduce_sum(tf.cast(top_k, tf.float32)) / BATCH_SIZE
    # RMSE
    predict_class = tf.cast(tf.argmax(predict, axis=1), dtype=tf.int32)
    rmse = tf.metrics.mean_squared_error(label_test, predict_class)
    # confusion matrix
    confu_mat = tf.confusion_matrix(label_test, predict_class)

    # tf.reset_default_graph()
    # saver = tf.train.import_meta_graph(os.path.join(os.getcwd(), '..', 'model', 'model2.ckpt.meta'))
    saver = tf.train.Saver()

    with tf.Session() as sess:
        # sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        saver.restore(sess, os.path.join(os.getcwd(), '..', 'model', 'model2.ckpt'))


        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        result = 0; count = 0; total1 = 0; total2 = 0;

        try:
            while not coord.should_stop():

                top_k_value = sess.run(top_k)
                rmse_value = sess.run(rmse)[0]
                total1 += top_k_value
                total2 += rmse_value
                print(top_k_value, rmse_value)
                # print(sess.run(confu_mat))

                count += 1

        except tf.errors.OutOfRangeError:
                print('Done')

        finally:
            coord.request_stop()
            print('final accuracy:',total1/count)
            print('final rmse:',total2/count)

            coord.join(threads)



if __name__ == '__main__':
    # main()
    eval_fn()



