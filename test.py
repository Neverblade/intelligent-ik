import tensorflow as tf

a = tf.constant([0, 0, 0, 0])
b = tf.constant([1, 2, 3, 4])
c = tf.losses.mean_squared_error(a, b)

with tf.Session() as sess:
    cc = sess.run(c)
    print(cc)