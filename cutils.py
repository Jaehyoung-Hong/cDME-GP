from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import tensor_util
from tensorflow_probability.python.math.psd_kernels.internal import util
from tensorflow_probability.python.math.psd_kernels.positive_semidefinite_kernel import PositiveSemidefiniteKernel

from tensorflow import keras
import tensorflow_probability as tfp
tfb = tfp.bijectors
tfd = tfp.distributions
psd_kernels = tfp.math.psd_kernels

import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

def get_mean_params(gp):
    params = []
    for param in gp.trainable_variables:
        if 'warping' in param.name:
            params.append(param)
    return tuple(params)

def get_kernel_params(gp):
    """
    Exclude embedding parameters. NOTE tentative
    """
    params = []
    for param in gp.trainable_variables:
        if 'warping' in param.name:
            continue
        params.append(param)
    return tuple(params)
  
def get_no_embed_params(gp):
    """
    Exclude embedding parameters. NOTE tentative
    """
    params = []
    for param in gp.trainable_variables:
        if 'embed' in param.name:
            continue
        params.append(param)
    return tuple(params)

def get_mean_no_embed_params(gp):
    """
    Exclude embedding parameters. NOTE tentative
    """
    params = []
    for param in gp.trainable_variables:
        if 'embed' in param.name:
            continue
        if 'warping' in param.name:
            params.append(param)
    return tuple(params)
  
# Where to save the figures
PROJECT_ROOT_DIR = "."
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "Result_article")
os.makedirs(IMAGES_PATH, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

class RBF_embed(PositiveSemidefiniteKernel):
  """The ExponentiatedQuadratic kernel.
  Sometimes called the "squared exponential", "Gaussian" or "radial basis
  function", this kernel function has the form
    ```none
    k(x, y) = amplitude**2 * exp(-||x - y||**2 / (2 * length_scale**2))
    ```
  where the double-bars represent vector length (ie, Euclidean, or L2 norm).
  """

  def __init__(self,
               amplitude=None,
               length_scale=None,
               feature_ndims=1,
               validate_args=False,
               embed_fn=None,
               name='RBF_embed'):
    """Construct an ExponentiatedQuadratic kernel instance.
    Args:
      amplitude: floating point `Tensor` that controls the maximum value
        of the kernel. Must be broadcastable with `length_scale` and inputs to
        `apply` and `matrix` methods. Must be greater than zero. A value of
        `None` is treated like 1.
        Default value: None
      length_scale: floating point `Tensor` that controls how sharp or wide the
        kernel shape is. This provides a characteristic "unit" of length against
        which `||x - y||` can be compared for scale. Must be broadcastable with
        `amplitude` and inputs to `apply` and `matrix` methods. A value of
        `None` is treated like 1.
        Default value: None
      feature_ndims: Python `int` number of rightmost dims to include in the
        squared difference norm in the exponential.
      validate_args: If `True`, parameters are checked for validity despite
        possibly degrading runtime performance
      name: Python `str` name prefixed to Ops created by this class.
    """
    parameters = dict(locals())
    with tf.name_scope(name):
      dtype = util.maybe_get_common_dtype(
          [amplitude, length_scale])
      self._amplitude = tensor_util.convert_nonref_to_tensor(
          amplitude, name='amplitude', dtype=dtype)
      self._length_scale = tensor_util.convert_nonref_to_tensor(
          length_scale, name='length_scale', dtype=dtype)
      super(RBF_embed, self).__init__(
          feature_ndims,
          dtype=dtype,
          name=name,
          validate_args=validate_args,
          parameters=parameters)
        
    self.embed_fn = embed_fn

  @property
  def amplitude(self):
    """Amplitude parameter."""
    return self._amplitude

  @property
  def length_scale(self):
    """Length scale parameter."""
    return self._length_scale

  def _batch_shape(self):
    scalar_shape = tf.TensorShape([])
    return tf.broadcast_static_shape(
        scalar_shape if self.amplitude is None else self.amplitude.shape,
        scalar_shape if self.length_scale is None else self.length_scale.shape)

  def _batch_shape_tensor(self):
    return tf.broadcast_dynamic_shape(
        [] if self.amplitude is None else tf.shape(self.amplitude),
        [] if self.length_scale is None else tf.shape(self.length_scale))

  def _apply_with_distance(
      self, x1, x2, pairwise_square_distance, example_ndims=0):
    exponent = -0.5 * pairwise_square_distance
    if self.length_scale is not None:
      length_scale = tf.convert_to_tensor(self.length_scale)
      length_scale = util.pad_shape_with_ones(
          length_scale, example_ndims)
      exponent = exponent / length_scale**2

    if self.amplitude is not None:
      amplitude = tf.convert_to_tensor(self.amplitude)
      amplitude = util.pad_shape_with_ones(amplitude, example_ndims)
      exponent = exponent + 2. * tf.math.log(amplitude)

    return tf.exp(exponent)

  def _apply(self, x1, x2, example_ndims=0):
    embed_x1 = self.embed_fn(x1)
    embed_x2 = self.embed_fn(x2)
    
    pairwise_square_distance = util.sum_rightmost_ndims_preserving_shape(
        tf.math.squared_difference(embed_x1, embed_x2), self.feature_ndims)
    return self._apply_with_distance(
        embed_x1, embed_x1, pairwise_square_distance, example_ndims=example_ndims)

  def _matrix(self, x1, x2):
    embed_x1 = self.embed_fn(x1)
    embed_x2 = self.embed_fn(x2)
    pairwise_square_distance = util.pairwise_square_distance_matrix(
        embed_x1, embed_x2, self.feature_ndims)
    return self._apply_with_distance(
        embed_x1, embed_x2, pairwise_square_distance, example_ndims=2)

  def _tensor(self, x1, x2, x1_example_ndims, x2_example_ndims):
    embed_x1 = self.embed_fn(x1)
    embed_x2 = self.embed_fn(x2)
    pairwise_square_distance = util.pairwise_square_distance_tensor(
        embed_x1, embed_x2, self.feature_ndims, x1_example_ndims, x2_example_ndims)
    return self._apply_with_distance(
        embed_x1, embed_x2, pairwise_square_distance,
        example_ndims=(x1_example_ndims + x2_example_ndims))

  def _parameter_control_dependencies(self, is_init):
    if not self.validate_args:
      return []
    assertions = []
    for arg_name, arg in dict(amplitude=self.amplitude,
                              length_scale=self.length_scale).items():
      if arg is not None and is_init != tensor_util.is_ref(arg):
        assertions.append(assert_util.assert_positive(
            arg,
            message='{} must be positive.'.format(arg_name)))
    return assertions