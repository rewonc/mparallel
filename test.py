import warnings
warnings.filterwarnings('ignore', '.*Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated*',)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import numpy as np
import time
import custom_nccl


def fwd_bwd_test(n_batch, hidden_dim, n_layers, distributed, n_devices=None):
    # test.
    n_burn_in = 10
    iters = 25

    x_in = tf.get_variable("data", dtype=tf.float32,
                           shape=[hidden_dim, n_batch], trainable=False)

    if not distributed:
        x = x_in
        for i in range(n_layers):
            W = tf.get_variable(str(i), initializer=tf.random_uniform_initializer(-0.01, 0.01),
                                dtype=tf.float32, shape=[hidden_dim, hidden_dim])
            Wx = tf.matmul(W, x)
            h = tf.nn.relu(Wx)
            x = x + h
        print(x.get_shape())
        y = tf.reduce_mean(x)
        grads = tf.gradients(y, tf.trainable_variables())
        meangrad = tf.reduce_mean(grads)

    if distributed:
        with tf.variable_scope('x', reuse=True):
            x_split = list(tf.split(x_in, n_devices, axis=0))
            for i in range(n_layers):
                Wxs = []
                for n in range(n_devices):
                    with tf.device('/gpu:{}'.format(n)):
                        W = tf.get_variable("{}-{}".format(i, n),
                                            shape=[hidden_dim, hidden_dim / n_devices],
                                            dtype=tf.float32,
                                            initializer=tf.random_uniform_initializer(-0.01, 0.01))
                        x = x_split[n]
                        Wx = tf.matmul(W, x)
                        Wxs.append(Wx)
                # Allreduce
                with tf.device('/gpu:0'):
                    Wxs = custom_nccl.all_sum(Wxs)
                # Postprocess
                for n in range(n_devices):
                    with tf.device('/gpu:{}'.format(n)):
                        # Shared normalizations, etc -- none here
                        # Now we focus each device on their split again
                        x = Wxs[n]
                        h = tf.split(x, n_devices, axis=0)[n]
                        h = tf.nn.relu(h)
                        x_split[n] = x_split[n] + h

            y = tf.concat(x_split, axis=0)
            print(y.get_shape())
            y = tf.reduce_mean(y)
            grads = tf.gradients(y, tf.trainable_variables(), colocate_gradients_with_ops=True)
            meangrad = tf.reduce_mean(grads)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        t0 = None
        for i in range(n_burn_in + iters):
            if i == n_burn_in:
                t0 = time.time()
            val, mg = sess.run([y, meangrad])
        t1 = time.time()
    print(val, mg, "runtime: {}".format((t1-t0)/iters), "parameters: {}GB".format(hidden_dim * hidden_dim * n_layers * 1e-9))


def correctness_test():
    matmul_size = 2048
    batch = 1024
    n_devices = 1

    W = np.arange(matmul_size**2).reshape([matmul_size, matmul_size]).astype(np.float32)
    W -= W.mean()
    W /= W.std()
    x = np.arange(matmul_size*batch)[::-1].reshape([matmul_size, batch]).astype(np.float32)
    x -= x.mean()
    x /= x.std()

    with tf.Graph().as_default():
        W_ = tf.constant(W)
        x_ = tf.constant(x)
        Wx = tf.matmul(W_, x_)
        h = tf.nn.relu(Wx - tf.reduce_mean(Wx)) + x_
        y_ = tf.reduce_mean(h)
        print(h.get_shape())
        with tf.Session() as sess:
            y = sess.run(y_)

    print("dense matmul result:", y)

    with tf.Graph().as_default():
        s = matmul_size // n_devices
        Ws = [tf.constant(W[:, s*i:s*(i+1)]) for i in range(n_devices)]
        xs = [tf.constant(x[s*i:s*(i+1), :]) for i in range(n_devices)]
        Wxs = []
        for n in range(n_devices):
            with tf.device('/gpu:{}'.format(n)):
                W = Ws[n]
                x = xs[n]
                Wx = tf.matmul(W, x)
                Wxs.append(Wx)
        with tf.device('/gpu:0'):
            Wxs = custom_nccl.all_sum(Wxs)
        for n in range(n_devices):
            with tf.device('/gpu:{}'.format(n)):
                # We can do normalizations, etc on the whole input
                Wx = Wxs[n]
                Wx -= tf.reduce_mean(Wx)
                # The output vector should be the appropriate input slice
                h = tf.split(Wx, n_devices, axis=0)[n]
                h = tf.nn.relu(h)
                xs[n] = xs[n] + h

        y = tf.concat(xs, axis=0)
        # Wx = tf.add_n(Wxs) / len(Wxs)
        print(y.get_shape())
        y_ = tf.reduce_mean(y)
        with tf.Session() as sess:
            y = sess.run(y_)

    print("dist matmul result:", y)




if __name__ == "__main__":
    np.random.seed(42)
    # correctness_test()
    fwd_bwd_test(n_batch=1024, hidden_dim=4096, n_layers=53, distributed=False)
    fwd_bwd_test(n_batch=1024, hidden_dim=4096, n_layers=53, distributed=True, n_devices=2)

    # Non model parallel version:
    # 425054600.0 runtime: 0.5763507175445557 parameters: 0.8891924480000001GB

    # model parallel version:
    # 7741546000.0 runtime: 0.521600046157837 parameters: 0.8891924480000001GB

    # biggest model w/ 2 gpus, 4 gpus, 8 gpus
    # speed keeping it the same
    # speed making it bigger
