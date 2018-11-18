
### Adapted from https://github.com/act65/mri-reconstruction/blob/2dcf30e10c37a482f1aab2524c5966d03eb72085/src/flows.py
import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp
tfb = tfp.bijectors
tfd = tfp.distributions


from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util

from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import tensor_shape

from tensorflow.python.ops import nn
from tensorflow_probability.python import bijectors
from tensorflow_probability.python.distributions import TransformedDistribution,Normal

from pymisca.tensorflow_extra_.tfp_transformed_distribution import TransformedDistribution
from pymisca.tensorflow_extra_.transjector import Transjector

def non_square_det(x, reltol=1e-6):
    """
    Idea taken from https://www.quora.com/How-do-we-calculate-the-determinant-of-a-non-square-matrix
    # for n != m
    A = tf.random_normal([n, m])
    det(A) := sqrt(det(A.A^T))
    Args:
        x (tf.tensor): shape in [..., a, b]
    Returns:
        [..., ]
    """
    # squared_mat = tf.matmul(x, x, transpose_b=True)
    # return tf.sqrt(tf.linalg.det(squared_mat))

    s = tf.svd(x, compute_uv=False)

    # atol = tf.reduce_max(s) * reltol
    # s = tf.diag(tf.where(tf.greater(atol, tf.abs(s)), tf.ones_like(s), s))

    return tf.reduce_prod(s)

def pinv(A, reltol=1e-6):
    """
    Args:
        A (tf.tensor): the matrix to be inverted shape=[n, m]
    Returns:
        inverse (tf.tensor): the invserse of A, s.t. A_T.A = I. shape=[m,n]
    """
    s, u, v = tf.svd(A)

    atol = tf.reduce_max(s) * reltol
    s_inv = tf.diag(tf.where(tf.greater(tf.abs(s), atol), 1.0/s, tf.zeros_like(s)))
    # s_inv = tf.diag(1./s)

    return tf.matmul(v, tf.matmul(s_inv, u, transpose_b=True))

# class NonSquareLinearTransform(Transjector):
class NonSquareLinearTransform(bijectors.Bijector):

    """
    Want a hierarchical flow.
    Map some low dim distribution to a manifold in a higher dimensional space.
    For more info on bijectors see tfb.Bijector, I simply cloned the general
    structure.
    """
    def __init__(self, weights=None, n_inputs= None, n_outputs = None,validate_args=False, name=''):
#     def __init__(self, n_inputs, n_outputs, validate_args=False, name=''):
        """
        Args:
            n_inputs (int): the number of features (last dim)
            n_outputs (int): the target num of feautres
        """
        super(self.__class__, self).__init__(
            validate_args=validate_args,
            is_constant_jacobian=True,
#             forward_min_event_ndims = 0,
            forward_min_event_ndims=1,
            inverse_min_event_ndims=1,
            name=name)
        with tf.variable_scope('dense'+name):
            if weights is not None:
                weights = tf.convert_to_tensor(weights)
                self.weights = weights
                shape = weights.shape
                self.n_inputs = n_inputs = shape[0].value
                self.n_outputs = n_outputs = shape[1].value

            else:
                assert n_inputs is not None
                assert n_outputs is not None
                self.weights = tf.get_variable(name='weights',
                                               shape=[n_inputs, n_outputs],
                                               dtype=tf.float32,
                                initializer=tf.initializers.orthogonal()
                                               )
#             self.bias = tf.get_variable(name='bias',
#                                         shape=[n_outputs],
#                                         dtype=tf.float32,
#                                         initializer=tf.initializers.zeros()
#                                         )

    @property
    def _is_injective(self):
        return True

    def  _forward_event_shape(self,input_shape):    
        e = input_shape[:-1].concatenate(
            tensor_shape.as_dimension(self.n_outputs))
        return e

#     def _invserse_event_shape_tensor(self, shape):
#         return tf.shape([shape[0], self.n_outputs])

    def _forward(self, x):
        y = tf.matmul(x, self.weights)
#         y = y + self.bias
        return  y

    def _inverse(self, y):
        
        weights_inv = pinv(self.weights)
#         y = y-self.bias
        return tf.matmul(y , weights_inv)

    def _forward_log_det_jacobian(self, x):
        return tf.log(non_square_det(self.weights))

    def _inverse_log_det_jacobian(self, y):
        return - tf.log(non_square_det(self.weights))