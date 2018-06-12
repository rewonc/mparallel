import warnings
warnings.filterwarnings('ignore', '.*Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated*',)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import numpy as np
import time
import custom_nccl

'''
Just write the ops manually to not deal w/ Tensorflow's stuff.

keep a dict:
    {var_name: tensor}

x = input
y = x + relu(Wx)



'''

def custom_back_test():
    input_dict = {}
    grad_dict = {}
    n_layers = 5
    hidden_dim = 128
    n_batch = 4
    x_in = tf.get_variable("data", dtype=tf.float32,
                           shape=[hidden_dim, n_batch], trainable=False)
    x = x_in
    with tf.variable_scope("test"):
        for i in range(n_layers):
            x = fwd(str(i), x, hidden_dim, input_dict, grad_dict)
    y = fwd_sum('sum', x, input_dict, grad_dict)


    dy = bwd_sum('sum', y, input_dict, grad_dict)
    with tf.variable_scope("test", reuse=True):
        for i in range(n_layers)[::-1]:
            dy = bwd(str(i), dy, hidden_dim, input_dict, grad_dict)

    weights = list(grad_dict.keys())
    tf_grads = tf.gradients(y, weights)
    my_grads = [grad_dict[k] for k in weights]
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        g1, g2 = sess.run([tf_grads, my_grads])
    print([x.sum() for x in g1])
    print([x.sum() for x in g2])
    import pdb; pdb.set_trace()


def fwd(name, x, hidden_dim, input_dict, grad_dict):
    W = tf.get_variable(name + "W", initializer=tf.random_uniform_initializer(-0.01, 0.01),
                        dtype=tf.float32, shape=[hidden_dim, hidden_dim])
    input_dict[name] = x
    Wx = tf.matmul(W, x)
    input_dict[name + "Wx"] = Wx
    h = tf.nn.relu(Wx)
    return h


def bwd(name, deltas, hidden_dim, input_dict, grad_dict):
    W = tf.get_variable(name + "W", initializer=tf.random_uniform_initializer(-0.01, 0.01),
                        dtype=tf.float32, shape=[hidden_dim, hidden_dim])
    x = input_dict[name]
    Wx = input_dict[name + "Wx"]
    deltas *= tf.to_float(tf.transpose(Wx) > 0)
    # x will be like 128x6
    # import pdb; pdb.set_trace()
    # deltas should be the same size, 128x6
    dydW = tf.matmul(x, deltas)
    if W not in grad_dict:
        grad_dict[W] = dydW
    else:
        grad_dict[W] += dydW

    dydx = tf.matmul(deltas, W)
    return dydx



def fwd_sum(name, xs, input_dict, grad_dict):
    # input size -- 128x6
    input_dict[name] = xs
    return tf.reduce_sum(xs)

def bwd_sum(name, ys, input_dict, grad_dict):
    xs = input_dict[name]
    return tf.transpose(tf.ones_like(xs))


def fwd_bwd_test(n_batch, hidden_dim, n_layers, distributed, n_devices=None, backprop=True):
    # test.
    n_burn_in = 10
    iters = 25

    with tf.Graph().as_default():
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
            y = tf.reduce_mean(x)
            weights = tf.trainable_variables()
            grads = tf.gradients(y, weights)
            updates = []
            for w, g in zip(weights, grads):
                updates.append(tf.assign(w, w - g * 0.01))
            train = tf.group(updates)

        if distributed:
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
            y = tf.reduce_mean(y)
            weights = tf.trainable_variables()
            grads = tf.gradients(y, weights, colocate_gradients_with_ops=True)
            updates = []
            for w, g in zip(weights, grads):
                assert w.device == g.device
                with tf.device(w.device):
                    updates.append(tf.assign(w, w - g * 0.01))
            train = tf.group(updates)
            # meangrad = tf.reduce_mean(grads)

        run_options = tf.RunOptions(report_tensor_allocations_upon_oom=True)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            t0 = None
            for i in range(n_burn_in + iters):
                if i == n_burn_in:
                    t0 = time.time()
                if backprop:
                    val = "n/a"
                    _ = sess.run(train, options=run_options)
                else:
                    val = sess.run(y, options=run_options)
            t1 = time.time()
        print("n_layers", n_layers, "devices:", n_devices, val, "runtime: {}".format((t1-t0)/iters), "parameters: {:.4f}B".format(hidden_dim * hidden_dim * n_layers * 1e-9))


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
    # custom_back_test()
    # correctness_test()

    # New tests, after fix:
    fwd_bwd_test(n_batch=1024, hidden_dim=4096*2, n_layers=36, distributed=False)
    # n_layers 36 devices: None n/a runtime: 1.732391471862793 parameters: 2.4159B
    fwd_bwd_test(n_batch=1024, hidden_dim=4096*2, n_layers=61, distributed=True, n_devices=2)
    # n_layers 61 devices: 2 n/a runtime: 1.8960966396331786 parameters: 4.0936B
    # fwd_bwd_test(n_batch=1024, hidden_dim=4096*2, n_layers=90, distributed=True, n_devices=4)
    fwd_bwd_test(n_batch=1024, hidden_dim=4096*2, n_layers=89, distributed=True, n_devices=4)
    # n_layers 89 devices: 4 n/a runtime: 2.1880528831481936 parameters: 5.9727B
    fwd_bwd_test(n_batch=1024, hidden_dim=4096*2, n_layers=106, distributed=True, n_devices=8)
# n_layers 106 devices: 8 n/a runtime: 2.2055407524108888 parameters: 7.1135B

    # for i in [36, 36, 37, 37]:
    # # for i in range(30, 40):
    #     try:
    #         fwd_bwd_test(n_batch=1024, hidden_dim=4096*2, n_layers=i, distributed=False)
    #     except tf.errors.ResourceExhaustedError:
    #         print(i, "errored")
    #         break
    # for i in [60, 65, 70, 75, 80, 90, 100]:
    # for i in range(60, 65):
    #     try:
    #         # fwd_bwd_test(n_batch=4096*2, hidden_dim=4096, n_layers=i, distributed=True, n_devices=2)
    #         fwd_bwd_test(n_batch=1024, hidden_dim=4096*2, n_layers=i, distributed=True, n_devices=2)
    #     except tf.errors.ResourceExhaustedError:
    #         print(i, "errored")
    #         break

    # for i in [80, 90, 100, 110, 120, 130, 140, 150, 160]:
    # fwd_bwd_test(n_batch=1024, hidden_dim=4096*2, n_layers=92, distributed=True, n_devices=4)
    # for i in range(90, 140):
    #     try:
    #         # fwd_bwd_test(n_batch=4096*2, hidden_dim=4096, n_layers=i, distributed=True, n_devices=2)
    #         fwd_bwd_test(n_batch=1024, hidden_dim=4096*2, n_layers=i, distributed=True, n_devices=4)
    #     except tf.errors.ResourceExhaustedError:
    #         print(i, "errored")
    #         break

    # for i in [90, 100, 110, 120, 130]:
    # for i in range(100, 120):
    #     try:
    #         # fwd_bwd_test(n_batch=4096*2, hidden_dim=4096, n_layers=i, distributed=True, n_devices=2)
    #         fwd_bwd_test(n_batch=1024, hidden_dim=4096*2, n_layers=i, distributed=True, n_devices=8)
    #     except tf.errors.ResourceExhaustedError:
    #         print(i, "errored")
    #         break

    # for i in [120, 130, 140, 150, 160, 170, 180, 190]:
    # # for i in range(50, 60):
    #     try:
    #         # fwd_bwd_test(n_batch=4096*2, hidden_dim=4096, n_layers=i, distributed=True, n_devices=2)
    #         fwd_bwd_test(n_batch=1024, hidden_dim=4096*2, n_layers=i, distributed=True, n_devices=8)
    #     except tf.errors.ResourceExhaustedError:
    #         print(i, "errored")
    #         break


    # for i in [50, 60, 70, 80, 90, 100, 120, 140, 160]:
    #     try:
    #         fwd_bwd_test(n_batch=4096*2, hidden_dim=4096, n_layers=i, distributed=True, n_devices=8)
    #         fwd_bwd_test(n_batch=4096*2, hidden_dim=4096, n_layers=i, distributed=True, n_devices=8)
    #     except tf.errors.ResourceExhaustedError:
    #         errors = 1
    #         for j in range(5):
    #             try:
    #                 fwd_bwd_test(n_batch=4096*2, hidden_dim=4096, n_layers=i, distributed=True, n_devices=8)
    #             except tf.errors.ResourceExhaustedError:
    #                 errors += 1
    #                 if errors >= 3:
    #                     break
    #         print(i, "errored", errors, "times")
    #         if errors >= 3:
    #             break


    # fwd_bwd_test(n_batch=1024, hidden_dim=4096, n_layers=1, distributed=False, backprop=False)
    # fwd_bwd_test(n_batch=1024, hidden_dim=4096, n_layers=1, distributed=True, n_devices=2, backprop=False)

    # fwd_bwd_test(n_batch=1024, hidden_dim=4096, n_layers=80, distributed=False)
    # fwd_bwd_test(n_batch=1024, hidden_dim=4096, n_layers=53, distributed=True, n_devices=2)
    # fwd_bwd_test(n_batch=1024, hidden_dim=4096, n_layers=53, distributed=True, n_devices=4)
    # fwd_bwd_test(n_batch=1024, hidden_dim=4096, n_layers=53, distributed=True, n_devices=8)


    # for i in [65, 68, 70, 75, 80, 85, 90]:
    #     try:
    #         fwd_bwd_test(n_batch=1024, hidden_dim=4096, n_layers=i, distributed=True, n_devices=2)
    #     except tf.errors.ResourceExhaustedError:
    #         print(i, "errored.")

    # for i in [68, 70, 75, 80, 85, 90]:
    #     try:
    #         fwd_bwd_test(n_batch=1024, hidden_dim=4096, n_layers=i, distributed=True, n_devices=4)
    #     except tf.errors.ResourceExhaustedError:
    #         print(i, "errored.")
    
    # fwd_bwd_test(n_batch=1024, hidden_dim=4096, n_layers=62, distributed=True, n_devices=2)
    # fwd_bwd_test(n_batch=1024, hidden_dim=4096, n_layers=71, distributed=True, n_devices=4)
    # fwd_bwd_test(n_batch=1024, hidden_dim=4096, n_layers=72, distributed=True, n_devices=8)
    # fwd_bwd_test(n_batch=1024, hidden_dim=4096, n_layers=65, distributed=True, n_devices=2)

    # Does this change for wider networks
    # fwd_bwd_test(n_batch=1024, hidden_dim=4096*2, n_layers=13, distributed=False)
    # fwd_bwd_test(n_batch=1024, hidden_dim=4096*2, n_layers=15, distributed=True, n_devices=2)
    # fwd_bwd_test(n_batch=1024, hidden_dim=4096*2, n_layers=17, distributed=True, n_devices=4)
    # fwd_bwd_test(n_batch=1024, hidden_dim=4096*2, n_layers=18, distributed=True, n_devices=8)

    # How does the scaling change w/ batch size?
    # fwd_bwd_test(n_batch=256, hidden_dim=4096*2, n_layers=13, distributed=False)
    # fwd_bwd_test(n_batch=256, hidden_dim=4096*2, n_layers=15, distributed=True, n_devices=2)
    # fwd_bwd_test(n_batch=256, hidden_dim=4096*2, n_layers=17, distributed=True, n_devices=4)
    # fwd_bwd_test(n_batch=256, hidden_dim=4096*2, n_layers=19, distributed=True, n_devices=8)


    # biggest model w/ 2 gpus, 4 gpus, 8 gpus
    # speed keeping it the same
    # speed making it bigger
