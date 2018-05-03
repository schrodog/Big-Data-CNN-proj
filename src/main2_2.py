import numpy as np
import tensorflow as tf
import os
import numpy as np
import benchmark

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

'''
50,000 (train+validation), 10,000 test image
32x32 (1024) images, 3 channels
binary format: [1,1024,1024,1024]
'''

DATA_DIR = os.path.join(os.getcwd(), '..', 'cifar-10-batches-bin')
# file_run: log file number; model file number
FILE_RUN = 66
MODEL_NUM = 66

NUM_EPOCHS = 10000
NUM_CLASS = 10
NUM_EXAMPLE_TRAIN = 50000
BATCH_SIZE = 256
learning_rate = 0.003
DECAY_EPOCH = 8000
DECAY_FACTOR = 0.96


def read_input(file_list):
    reader = tf.FixedLengthRecordReader(record_bytes=3073)
    key, value = reader.read(file_list)

    record_bytes = tf.decode_raw(value, tf.uint8)
    label = tf.cast(tf.strided_slice(record_bytes, [0], [1]), tf.int32)

    reshaped_bytes = tf.reshape(
        tf.strided_slice(record_bytes, [0], [3072]), [3, 32, 32]
    )
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
    filename = ''
    if mode == 'train':
        filenames = [os.path.join(data_dir, 'data_batch_'+str(i)+'.bin') for i in range(1,6) ]
    elif mode == 'validation':
        filenames = [os.path.join(data_dir, 'data_batch_'+str(i)+'.bin') for i in range(1,6) ]
    elif mode == 'test':
        filenames = [os.path.join(data_dir, 'test_batch.bin') ]

    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)

    if mode == 'train' or mode == 'validation':
        file_list = tf.train.string_input_producer(filenames, num_epochs=NUM_EPOCHS)
    else:
        file_list = tf.train.string_input_producer(filenames, num_epochs=NUM_EPOCHS)

    image_data, label  = read_input(file_list)

    if mode == 'test' or mode == 'validation':
        reshaped_image = tf.cast(image_data, tf.float32)
        reshaped_image.set_shape([32, 32, 3])
        label.set_shape([1])
        if mode == 'test':
            return generate_input(reshaped_image, label, 10000, 10000, shuffle=False)
        else:
            return generate_input(reshaped_image, label, 10000, batch_size, shuffle=True)

    with tf.name_scope('preprocess'):
        reshaped_image = tf.cast(image_data, tf.float32)
        
        # flip the image horizontally
        distorted_image = tf.image.random_flip_left_right(reshaped_image)
        # randomize the order of the operation
        distorted_image = tf.image.random_brightness(distorted_image, max_delta=63)
        distorted_image = tf.image.random_contrast(distorted_image, lower=0.2, upper=1.8)

        # Subtract off the mean and divide by the variance of the pixels.
        float_image = tf.image.per_image_standardization(distorted_image)
        # Set the shapes of tensors.
        float_image.set_shape([32, 32, 3])
        label.set_shape([1])
        # random shuffling
        min_fraction_example = 1.0
        min_list_examples = int(NUM_EXAMPLE_TRAIN * min_fraction_example)

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

def _activation(mult_add, name='activation', norm=False, mode='train', fc=False):
    return tf.nn.relu(mult_add, name=name)


def cnn_network(input_x, mode):

    if mode == 'train': #or mode=='validation':
        bn_training = True
    else:
        bn_training = True

    with tf.variable_scope("layer1", reuse=tf.AUTO_REUSE):
        filters1 = _weighted_variable([3,3,3,64])
        conv1 = _conv2d(input_x, filters1, [1,1,1,1])
        bias1 = _bias_variable([64])
        # bn1 =
        bn1 = tf.layers.batch_normalization(conv1+bias1, momentum=0.9, training=bn_training)
        activ1 = _activation(bn1, mode=mode)
        # activ1 = _activation(conv1+bias1, mode=mode)

        pool1 = _pool(activ1, ksize=[1,3,3,1], strides=[1,2,2,1])

    with tf.variable_scope("layer2", reuse=tf.AUTO_REUSE):
        filters2 = _weighted_variable([3,3,64,64])
        conv2 = _conv2d(pool1, filters2, [1,1,1,1])
        bias2 = _bias_variable([64])

        bn2 = tf.layers.batch_normalization(conv2+bias2, momentum=0.9, training=bn_training)
        activ2 = _activation(bn2, norm=False, mode=mode)
        # activ2 = _activation(conv2+bias2, norm=False, mode=mode)

        pool2 = _pool(activ2, ksize=[1,3,3,1], strides=[1,1,1,1])

    with tf.variable_scope("layer3", reuse=tf.AUTO_REUSE):
        filters3 = _weighted_variable([3,3,64,64])
        conv3 = _conv2d(pool2, filters3, [1,1,1,1])
        bias3 = _bias_variable([64])

        bn3 = tf.layers.batch_normalization(conv3+bias3, momentum=0.9, training=bn_training)
        activ3 = _activation(bn3, mode=mode)
        # activ3 = _activation(conv3+bias3, mode=mode)
        pool3 = _pool(activ3, ksize=[1,3,3,1], strides=[1,2,2,1])


    with tf.variable_scope("fc1", reuse=tf.AUTO_REUSE):
        n = 8; m = 64
        reshape = tf.reshape(pool3, [-1, n*n*m])
        w_fc1 = _weighted_variable([n*n*m, 1024])
        b_fc1 = _bias_variable([1024])

        bn_fc1 = tf.layers.batch_normalization(tf.matmul(reshape, w_fc1) + b_fc1, momentum=0.9, training=bn_training)
        activ_fc1 = tf.nn.relu(bn_fc1)

    with tf.variable_scope("output_layer", reuse=tf.AUTO_REUSE):

        w_out = _weighted_variable([1024, NUM_CLASS])
        b_out = _bias_variable([NUM_CLASS])
        softmax = tf.nn.softmax(tf.matmul(activ_fc1, w_out) + b_out)

        # tf.summary.histogram('softmax', softmax)
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

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = tf.train.AdamOptimizer(lr).minimize(losses, global_step=global_step)
        # train_op = tf.keras.optimizers.Nadam(lr).minimize(losses)
        return train_op, update_ops


# accuracy
def accuracy_fn(input_x, input_y):
    input_x = tf.cast(input_x, dtype=tf.int32)
    accuracy = tf.reduce_mean(
        tf.cast(tf.equal( input_x , input_y ), dtype=tf.float32))
    tf.summary.scalar('accuracy', accuracy)
    return accuracy


def model_fn(features,labels, mode):

    logits_train = cnn_network(features, mode='train')
    logits_test = cnn_network(features, mode='test')

    predict_class = tf.argmax(logits_test, axis=1)

    if mode == 'train':
        return logits_train, labels

    if mode == 'test' or mode=='validation':
        accuracy_op = accuracy_fn(predict_class, labels)
        return accuracy_op


# === MAIN ===

# training
def main(argv=None):
    count = 0; high_count = 0

    # train
    image_data_train, label_train = distorted_input(DATA_DIR, BATCH_SIZE, 'train')
    logits_train, label_trains = model_fn(image_data_train, label_train, 'train')
    loss_op, lossx, lossy = loss_fn(logits_train, label_trains)
    train_op, update_op = train(loss_op, learning_rate=learning_rate)

    # test
    image_data_test, label_test = distorted_input(DATA_DIR, BATCH_SIZE, 'test')
    predict = cnn_network(image_data_test,'test')

    top_k = tf.nn.in_top_k(predict, label_test, 1)
    top_k = tf.reduce_sum(tf.cast(top_k, tf.float32))/10000

    # validate
    image_data_valid, label_valid = distorted_input(DATA_DIR, BATCH_SIZE, 'validation')
    accuracy_op = model_fn(image_data_valid, label_valid, 'test')

    summaries = tf.summary.merge_all()

    # saver = tf.train.Saver(var_list=var_list, max_to_keep=5)
    saver = tf.train.Saver()
    # loss_op = loss_fn(logits_train, label_train)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        # saver.restore(sess, os.path.join(os.getcwd(), '..', 'model', 'model64.ckpt'))


        writer = tf.summary.FileWriter("logs/run"+str(FILE_RUN), sess.graph)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)


        try:
            while not coord.should_stop():
                count +=1

                sess.run(train_op)
                sess.run(update_op)
                summ = sess.run(summaries)
                writer.add_summary(summ, global_step=sess.run(global_step))

                if count % 5 == 0:
                    print('step:', count, 'loss:',sess.run(loss_op))

                acc_value = sess.run(accuracy_op)

                if count%50 == 0:
                    top_k_value = sess.run(top_k)
                    print('accuracy: ',acc_value, top_k_value )

        except tf.errors.OutOfRangeError:
            print('Done')

        finally:
            coord.request_stop()
            coord.join(threads)

            saver_path = saver.save(sess, os.path.join(DATA_DIR,'..', 'model' , 'model'+str(MODEL_NUM)+'.ckpt'))
            print("Model saved in path:", saver_path)


# testing performance
def eval_fn(num):
    image_data_test, label_test = distorted_input(DATA_DIR, BATCH_SIZE, 'test')
    predict = cnn_network(image_data_test,'test')

    # top_k metrics
    top_k1 = tf.nn.in_top_k(predict, label_test, 1)
    top_k = tf.reduce_mean(tf.cast(top_k1, tf.float32))

    # RMSE
    predict_class = tf.cast(tf.argmax(predict, axis=1), dtype=tf.int32)
    rmse = tf.metrics.mean_squared_error(label_test, predict_class)

    res_matrix = tf.stack([ tf.cast(label_test, tf.int32), tf.cast(predict_class, tf.int32)], axis=0)

    # confusion matrix
    confu_mat = tf.confusion_matrix(label_test, predict_class)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.local_variables_initializer())
        saver.restore(sess, os.path.join(os.getcwd(), '..', 'model', 'model'+str(num)+'.ckpt'))

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # eval stat
        top_k_value = sess.run(top_k)
        rmse_value = sess.run(rmse)[0]
        total1 += top_k_value
        total2 += rmse_value
        print(top_k_value, rmse_value)
        print(sess.run(confu_mat))

        benchmark.f1_score(sess.run(confu_mat))
        benchmark.showROC(sess.run(res_matrix))


        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    # main()
    eval_fn(66)



