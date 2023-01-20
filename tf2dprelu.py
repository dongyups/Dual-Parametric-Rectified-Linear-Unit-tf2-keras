### Reference: https://github.com/keras-team/keras/blob/v2.11.0/keras/layers/activation/prelu.py#L30-L124
"""Dual-Parametric Rectified Linear Unit"""

import tensorflow as tf
from tensorflow.keras import constraints
from tensorflow.keras import initializers
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Layer, InputSpec


class DPReLU(Layer):
    """
    It follows:
    ```
      f(x) = alpha * x for x < 0
      f(x) =  beta * x for x >= 0
    ```
    where `alpha` and `beta` are each a learned array with the same shape as x.

    Input shape:
      Arbitrary. Use the keyword argument `input_shape`
      (tuple of integers, does not include the samples axis)
      when using this layer as the first layer in a model.

    Output shape:
      Same shape as the input.

    Args:
      alpha/beta_initializer: Initializer function for the weights.
      alpha/beta_regularizer: Regularizer for the weights.
      alpha/beta_constraint: Constraint for the weights.
      shared_axes: The axes along which to share learnable
        parameters for the activation function.
        For example, if the incoming feature maps
        are from a 2D convolution
        with output shape `(batch, height, width, channels)`,
        and you wish to share parameters across space
        so that each filter only has one set of parameters,
        set `shared_axes=[1, 2]`.
    """
    def __init__(
        self,
        alpha_initializer=initializers.Constant(0.25),
        alpha_regularizer=None,
        alpha_constraint=None,
        beta_initializer=initializers.Constant(0.25),
        beta_regularizer=None,
        beta_constraint=None,
        shared_axes=None,
        name="dp_relu",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.supports_masking = True
        self.alpha_initializer = initializers.get(alpha_initializer)
        self.alpha_regularizer = regularizers.get(alpha_regularizer)
        self.alpha_constraint = constraints.get(alpha_constraint)
        self.beta_initializer = initializers.get(beta_initializer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)

        if shared_axes is None:
            self.shared_axes = None
        elif not isinstance(shared_axes, (list, tuple)):
            self.shared_axes = [shared_axes]
        else:
            self.shared_axes = list(shared_axes)

    def build(self, input_shape):
        param_shape = list(input_shape[1:])
        if self.shared_axes is not None:
            for i in self.shared_axes:
                param_shape[i - 1] = 1
        self.alpha = self.add_weight(
            shape=param_shape,
            name="alpha",
            initializer=self.alpha_initializer,
            regularizer=self.alpha_regularizer,
            constraint=self.alpha_constraint,
        )
        self.beta = self.add_weight(
            shape=param_shape,
            name="beta",
            initializer=self.beta_initializer,
            regularizer=self.beta_regularizer,
            constraint=self.beta_constraint,
        )
        # Set input spec
        axes = {}
        if self.shared_axes:
            for i in range(1, len(input_shape)):
                if i not in self.shared_axes:
                    axes[i] = input_shape[i]
        self.input_spec = InputSpec(ndim=len(input_shape), axes=axes)
        self.built = True

    def call(self, inputs):
        neg = -self.alpha * tf.nn.relu(-inputs)
        pos = self.beta * tf.nn.relu(inputs)
        return neg + pos

    def get_config(self):
        config = {
            "alpha_initializer": initializers.serialize(self.alpha_initializer),
            "alpha_regularizer": regularizers.serialize(self.alpha_regularizer),
            "alpha_constraint": constraints.serialize(self.alpha_constraint),
            "beta_initializer": initializers.serialize(self.beta_initializer),
            "beta_regularizer": regularizers.serialize(self.beta_regularizer),
            "beta_constraint": constraints.serialize(self.beta_constraint),
            "shared_axes": self.shared_axes,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape
