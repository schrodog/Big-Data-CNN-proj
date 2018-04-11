import tensorflow as tf
import os

DATA_DIR = os.path.join(os.getcwd(), '..', 'cifar-10-batches-bin', 'data_batch_1.bin')

filenames = [DATA_DIR]
filename_queue = tf.train.string_input_producer(filenames)

reader = tf.FixedLengthRecordReader(record_bytes=4)

key, value = reader.read(filename_queue)
b = value
sess = tf.InteractiveSession()
tf.train.start_queue_runners(sess=sess)

print(sess.run(b))
print('\n')
print(sess.run(b))