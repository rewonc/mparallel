import tensorflow as tf
from tensorflow.contrib.nccl.python.ops import nccl_ops


@tf.RegisterGradient('NcclAllReduceFixed')
def _all_sum_grad(op, grad):
    """The gradients for `all_sum`.
    Args:
      op: The `all_sum` `Operation` that we are differentiating.
      grad: Gradient with respect to the output of the `all_sum` op.
    Returns:
      The gradient with respect to the output of `all_sum`.
    Raises:
      LookupError: If `reduction` is not `sum`.
    """
    if op.get_attr('reduction').decode('utf-8') != 'sum':
        raise LookupError('No gradient defined for NcclAllReduce except sum.')

    nccl_ops._check_device(grad, expected=op.device)
    num_devices = op.get_attr('num_devices')
    shared_name = op.get_attr('shared_name').decode('utf-8') + '_grad'

    with tf.device(op.device):
        return nccl_ops.gen_nccl_ops.nccl_all_reduce(
            input=grad,
            reduction='sum',
            num_devices=num_devices,
            shared_name=shared_name)


def all_sum(x):
    g = tf.get_default_graph()
    with g.gradient_override_map({"NcclAllReduce": "NcclAllReduceFixed"}):
        return nccl_ops.all_sum(x)
