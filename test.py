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


def nomp_vs_mp(n_layers, hidden_dim, n_batch, n_devices=None):
    input_dict = {}
    grad_dict = {}
    n_iters = 25
    x_in = tf.get_variable("data", dtype=tf.float32,
                           shape=[hidden_dim, n_batch], trainable=False)
    x = x_in if not n_devices else tf.split(x_in, n_devices, axis=0)

    with tf.variable_scope("test"):
        for i in range(n_layers):
            x = fwd(str(i), x, hidden_dim, input_dict, grad_dict, n_devices)
    y = fwd_sum('sum', x, input_dict, grad_dict, n_devices)


    dy = bwd_sum('sum', y, input_dict, grad_dict, n_devices)
    with tf.variable_scope("test", reuse=True):
        for i in range(n_layers)[::-1]:
            dy = bwd(str(i), dy, hidden_dim, input_dict, grad_dict, n_devices)

    weights = list(grad_dict.keys())
    grads = [grad_dict[k] for k in weights]

    grad_steps = []
    for w, g in zip(weights, grads):
        grad_steps.append(tf.assign(w, w - 0.01 * g))
    train = tf.group(grad_steps)

    run_metadata = tf.RunMetadata()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # g1, g2 = sess.run([tf_grads, my_grads])
        for i in range(n_iters):
            _ = sess.run(train, options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
                          run_metadata=run_metadata)

    tf.profiler.profile(
        tf.get_default_graph(),
        run_meta=run_metadata, cmd='op',
        options=tf.profiler.ProfileOptionBuilder.time_and_memory())


def back_vs_allreduce():
    n_layers = 8
    hidden_dim = 4096
    n_iters = 25
    n_batch = 256
    n_devices = 4

    x_in = tf.get_variable("data", dtype=tf.float32,
                           shape=[hidden_dim, n_batch], trainable=False)
    x = x_in
    input_dict = {}
    grad_dict = {}
    with tf.variable_scope("test"):
        for i in range(n_layers):
            x = fwd(str(i), x, hidden_dim, input_dict, grad_dict)
    y = fwd_sum('sum', x, input_dict, grad_dict)

    dy = bwd_sum('sum', y, input_dict, grad_dict)
    with tf.variable_scope("test", reuse=True):
        for i in range(n_layers)[::-1]:
            dy = bwd(str(i), dy, hidden_dim, input_dict, grad_dict)

    weights = list(grad_dict.keys())
    grads1 = [grad_dict[k] for k in weights]

    x = tf.split(x_in, n_devices, axis=0)
    input_dict = {}
    grad_dict = {}

    with tf.variable_scope("test-mp"):
        for i in range(n_layers):
            x = fwd(str(i), x, hidden_dim, input_dict, grad_dict, n_devices=n_devices)
    y = fwd_sum('sum', x, input_dict, grad_dict, n_devices=n_devices)

    dy = bwd_sum('sum', y, input_dict, grad_dict, n_devices=n_devices)
    with tf.variable_scope("test-mp", reuse=True):
        for i in range(n_layers)[::-1]:
            dy = bwd(str(i), dy, hidden_dim, input_dict, grad_dict, n_devices=n_devices)

    weights = list(grad_dict.keys())
    grads2 = [grad_dict[k] for k in weights]

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(n_iters):
            g1 = sess.run(grads1)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(n_iters):
            g2 = sess.run(grads2)
    print([x.sum() for x in g1])
    print([x.sum() for x in g2])


def custom_back_test():
    input_dict = {}
    grad_dict = {}
    n_layers = 8
    hidden_dim = 4096
    n_iters = 25
    n_batch = 256
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

    run_metadata = tf.RunMetadata()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # g1, g2 = sess.run([tf_grads, my_grads])
        for i in range(n_iters):
            g1 = sess.run(tf_grads, options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
                          run_metadata=run_metadata)

    print([x.sum() for x in g1])
    tf.profiler.profile(
        tf.get_default_graph(),
        run_meta=run_metadata, cmd='op',
        options=tf.profiler.ProfileOptionBuilder.time_and_memory())

    run_metadata = tf.RunMetadata()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # g1, g2 = sess.run([tf_grads, my_grads])
        for i in range(n_iters):
            g2 = sess.run(my_grads, options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
                          run_metadata=run_metadata)

    tf.profiler.profile(
        tf.get_default_graph(),
        run_meta=run_metadata, cmd='op',
        options=tf.profiler.ProfileOptionBuilder.time_and_memory())

    print([x.sum() for x in g1])
    print([x.sum() for x in g2])


def fwd(name, x, hidden_dim, input_dict, grad_dict, n_devices=None):
    if not n_devices:
        W = tf.get_variable(name + "W", initializer=tf.random_uniform_initializer(-0.01, 0.01),
                            dtype=tf.float32, shape=[hidden_dim, hidden_dim])
        input_dict[name] = x
        Wx = tf.matmul(W, x)
        input_dict[name + "Wx"] = Wx
        h = tf.nn.relu(Wx)
        return h
    input_dict[name] = x
    Wxs = []
    input_dict[name + "Wxs"] = Wxs
    for n in range(n_devices):
        with tf.device('/gpu:{}'.format(n)):
            x_ = x[n]
            W = tf.get_variable("{}-{}".format(name, n),
                                shape=[hidden_dim, hidden_dim / n_devices],
                                dtype=tf.float32,
                                initializer=tf.random_uniform_initializer(-0.01, 0.01))
            Wx = tf.matmul(W, x_)
            Wxs.append(Wx)
    # Allreduce
    with tf.device('/gpu:0'):
        Wxs_sum = custom_nccl.all_sum(Wxs)
    input_dict[name + "Wxs_sum"] = Wxs_sum
    # Postprocess
    output = []
    for n in range(n_devices):
        with tf.device('/gpu:{}'.format(n)):
            # Shared normalizations, etc -- none here
            # Now we focus each device on their split again
            x = Wxs_sum[n]
            h = tf.split(x, n_devices, axis=0)[n]
            h = tf.nn.relu(h)
            output.append(h)
    return output


def bwd(name, deltas, hidden_dim, input_dict, grad_dict, n_devices=None):
    if not n_devices:
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
            grad_dict[W] = tf.transpose(dydW)
        else:
            grad_dict[W] += tf.transpose(dydW)

        dydx = tf.matmul(deltas, W)
        return dydx

    # deltas should be the slice of y
    x = input_dict[name]
    Wxs = input_dict[name + "Wxs"]
    Wxs_sum = input_dict[name + "Wxs_sum"]
    deltas = deltas

    dys = []
    for n in range(n_devices):
        with tf.device('/gpu:{}'.format(n)):
            Wxs_sum_n = tf.split(Wxs_sum[n], n_devices, axis=0)[n]
            deltas_n = deltas[n]
            deltas_n *= tf.to_float(tf.transpose(Wxs_sum_n) > 0)
            total = [tf.zeros_like(deltas_n) for _ in range(n_devices)]
            total[n] = deltas_n
            total = tf.concat(total, axis=1)
            dys.append(total)
    # Allreduce
    with tf.device('/gpu:0'):
        deltas_sum = custom_nccl.all_sum(dys)

    outputs = []
    for n in range(n_devices):
        with tf.device('/gpu:{}'.format(n)):
            W = tf.get_variable("{}-{}".format(name, n),
                                shape=[hidden_dim, hidden_dim / n_devices],
                                dtype=tf.float32,
                                initializer=tf.random_uniform_initializer(-0.01, 0.01))
            deltas_n = deltas_sum[n]
            x_n = x[n]
            dydW = tf.matmul(x_n, deltas_n)
            if W not in grad_dict:
                grad_dict[W] = tf.transpose(dydW)
            else:
                grad_dict[W] += tf.transpose(dydW)
            dydx = tf.matmul(deltas_n, W)
            outputs.append(dydx)
    return outputs


def fwd_sum(name, xs, input_dict, grad_dict, n_devices=None):
    # input size -- 128x6
    input_dict[name] = xs
    return tf.reduce_sum(xs)


def bwd_sum(name, ys, input_dict, grad_dict, n_devices=None):
    xs = input_dict[name]
    if not n_devices:
        return tf.transpose(tf.ones_like(xs))
    return [tf.transpose(tf.ones_like(x)) for x in xs]


def fwd_bwd_test(n_batch, hidden_dim, n_layers, distributed, n_devices=None, backprop=True):
    # test.
    n_burn_in = 1
    iters = 1

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
            print(x.get_shape())
            y = tf.reduce_mean(x)
            weights = tf.trainable_variables()
            grads = tf.gradients(y, weights)
            grad_steps = []
            for w, g in zip(weights, grads):
                grad_steps.append(tf.assign(w, w - 0.01 * g))
            train = tf.group(grad_steps)


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
            print(y.get_shape())
            y = tf.reduce_mean(y)
            weights = tf.trainable_variables()
            grads = tf.gradients(y, weights)
            grad_steps = []
            for w, g in zip(weights, grads):
                import pdb; pdb.set_trace()
                grad_steps.append(tf.assign(w, w - 0.01 * g))
            train = tf.group(grad_steps)

        run_options = tf.RunOptions(report_tensor_allocations_upon_oom=True)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            t0 = None
            for i in range(n_burn_in + iters):
                if i == n_burn_in:
                    t0 = time.time()
                if backprop:
                    val, mg = sess.run([train], options=run_options)
                else:
                    val = sess.run(y, options=run_options)
                    mg = "n/a"
            t1 = time.time()
        print(val, mg, "runtime: {}".format((t1-t0)/iters), "parameters: {:.4f}B".format(hidden_dim * hidden_dim * n_layers * 1e-9))


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
    # back_vs_allreduce()
    # nomp_vs_mp(n_layers=53, hidden_dim=4096, n_batch=1024)
    # nomp_vs_mp(n_layers=63, hidden_dim=4096, n_batch=1024, n_devices=2)

    # correctness_test()
    # fwd_bwd_test(n_batch=1024, hidden_dim=4096, n_layers=53, distributed=False)
    fwd_bwd_test(n_batch=1024, hidden_dim=4096, n_layers=53, distributed=True, n_devices=2)
    # fwd_bwd_test(n_batch=1024, hidden_dim=4096, n_layers=53, distributed=True, n_devices=4)
    # fwd_bwd_test(n_batch=1024, hidden_dim=4096, n_layers=53, distributed=True, n_devices=8)

    # How deep can you go, with storing all acts?
    # fwd_bwd_test(n_batch=1024, hidden_dim=4096, n_layers=62, distributed=True, n_devices=2)
    # fwd_bwd_test(n_batch=1024, hidden_dim=4096, n_layers=71, distributed=True, n_devices=4)
    # fwd_bwd_test(n_batch=1024, hidden_dim=4096, n_layers=72, distributed=True, n_devices=8)

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
