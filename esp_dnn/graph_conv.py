# Copyright 2019 Astex Therapeutics Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import

from keras import activations, initializers, regularizers, constraints
from keras import backend as K
from keras.constraints import min_max_norm
from keras.engine.topology import Layer, InputSpec
import tensorflow as tf


class PrintLayerInput(Layer):
    def call(self, inputs):
        inputs = tf.Print(inputs, data=[inputs],
                          message="layer inputs: ", summarize=100)
        return inputs


class GraphConv(Layer):

    def __init__(self,
                 width,
                 activation=None,
                 use_bias=True,
                 kernel_initializer="glorot_uniform",
                 bias_initializer="zeros",
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 # "single": only one weight applied to all neighbor sums
                 # "all": a different weight for each property
                 conv_wts="single",
                 **kwargs):

        if "input_shape" not in kwargs and "input_dim" in kwargs:
            kwargs["input_shape"] = (kwargs.pop("input_dim"),)

        allowed_conv_wts = ("all", "single")
        if conv_wts not in allowed_conv_wts:
            raise ValueError("conv_wt should be one of %r" % allowed_conv_wts)

        super(GraphConv, self).__init__(**kwargs)

        self.width = width
        self.conv_wts = conv_wts

        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = [InputSpec(ndim=3), InputSpec(ndim=3)]

    def build(self, input_shapes):
        X_shape = input_shapes[0]
        # number of atom props * output width
        kernel_shape = (X_shape[-1], self.width)

        # atom (self) weights
        self.kernel_dense = self.add_weight(shape=kernel_shape,
                                            initializer=self.kernel_initializer,
                                            name="dense_kernel",
                                            regularizer=self.kernel_regularizer,
                                            constraint=self.kernel_constraint)
        if self.use_bias is not None:
            self.bias = self.add_weight(shape=(self.width,),
                                        initializer=self.bias_initializer,
                                        name="bias",
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None

        constraint = min_max_norm(
            min_value=0.0, max_value=1.0, rate=1.0, axis=0)
        if self.conv_wts == "single":
            self.kernel_neigh = self.add_weight(shape=[1],
                                                initializer=self.kernel_initializer,
                                                name="kernel_neigh",
                                                regularizer=self.kernel_regularizer,
                                                constraint=constraint)
            self.kernel_self = self.add_weight(shape=[1],
                                               initializer=self.kernel_initializer,
                                               name="kernel_self",
                                               regularizer=self.kernel_regularizer,
                                               constraint=constraint)
        elif self.conv_wts == "all":
            self.kernel_neigh = self.add_weight(shape=(self.width,),
                                                initializer=self.kernel_initializer,
                                                name="kernel_neigh",
                                                regularizer=self.kernel_regularizer,
                                                constraint=constraint)
            self.kernel_self = self.add_weight(shape=(self.width,),
                                               initializer=self.kernel_initializer,
                                               name="kernel_neigh",
                                               regularizer=self.kernel_regularizer,
                                               constraint=constraint)

        self.built = True

    def call(self, inputs):
        x = inputs[0]  # n_atom * n_props
        d = inputs[1]  # [n_atoms, n_atoms]

        self_output = K.dot(x, self.kernel_dense)
        # sum values from the neighbors
        neigh_output = K.batch_dot(d, self_output, axes=[2, 1])

        if self.conv_wts == "single":
            neigh_output = neigh_output * self.kernel_neigh[0]
            self_output = self_output * self.kernel_self[0]
        elif self.conv_wts == "all":
            neigh_output = neigh_output * self.kernel_neigh
            self_output = self_output * self.kernel_self

        output = self_output + neigh_output
        if self.use_bias is not None:
            output += K.reshape(self.bias, (1, self.width))
        output = self.activation(output)
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], self.width)

    def get_config(self):
        config = {
            "width": self.width,
            "conv_wts": self.conv_wts,
            "activation": activations.serialize(self.activation),
            "use_bias": self.use_bias,
            "kernel_initializer": initializers.serialize(self.kernel_initializer),
            "bias_initializer": initializers.serialize(self.bias_initializer),
            "kernel_regularizer": regularizers.serialize(self.kernel_regularizer),
            "bias_regularizer": regularizers.serialize(self.bias_regularizer),
            "activity_regularizer": regularizers.serialize(self.activity_regularizer),
            "kernel_constraint": constraints.serialize(self.kernel_constraint),
            "bias_constraint": constraints.serialize(self.bias_constraint)
        }
        base_config = super(GraphConv, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
