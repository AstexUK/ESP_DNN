from keras.layers import Dense, Input, Activation, SpatialDropout1D, BatchNormalization
from keras import Model
from keras.optimizers import Adam
from graph_conv import GraphConv
import tensorflow as tf
import keras.backend as K
import cStringIO
from tempfile import NamedTemporaryFile, mkstemp
import os
from keras.utils.np_utils import normalize
from keras.utils.training_utils import multi_gpu_model
from keras.models import load_model


def sum_squared_error_nan(y_true, y_pred):
    # y_true = tf.Print(y_true, message="y_true", data=[y_true], summarize=100)
    y_pred = tf.Print(y_pred, message="y_pred", data=[y_pred], summarize=100)
    return K.sum(tf.where(tf.is_nan(y_true), K.zeros_like(y_pred), K.square(y_pred - y_true)), axis=-1)
    # loss = tf.Print(loss, message="loss", data=[loss], summarize=100000)
    # return loss


def sum_absolute_error_nan(y_true, y_pred):
    # this sums esp values for all features in an atom and
    # finally the sum of error is divided by all atoms (including the dummy
    # atoms added for padding (with all nan values). However, this should be
    # alright for minimization purpose.
    return K.sum(tf.where(tf.is_nan(y_true), K.zeros_like(y_pred), K.abs(y_pred - y_true)), axis=-1)


def mean_absolute_error_nans(y_true, y_pred):
    y_total = tf.cast(tf.reduce_sum(tf.cast(~tf.is_nan(y_true), tf.int32)), tf.float32)
    # y_total = tf.Print(y_total, message="y_total: ", data=[y_total], summarize=100)

    y_sum_err = tf.reduce_sum(tf.where(tf.is_nan(y_true), tf.zeros_like(y_pred), K.abs(y_pred - y_true)))
    # y_sum_err = tf.Print(y_sum_err, message="y_sum_error: ", data=[y_sum_err], summarize=100)

    error = tf.cond(y_total < 0.1, lambda: tf.constant(0.0), lambda: tf.divide(y_sum_err, y_total))
    # error = tf.Print(error, message="error: ", data=[error], summarize=100)

    return (error)


def build_model(n_features_per_atom, n_ys_per_atom, space, n_gpus=None):
    A = X = Input(name="X", shape=(None, n_features_per_atom))  # variable number of atoms
    D = Input(name="D", shape=(None, None))     # variable atom connectivity matrix

    activation = space["activation"]
    width = space["width"]
    dropout_p = space["dropout_p"]
    layer_i = 1

    for _ in range(2):
        A = Dense(width, name="dense_%d" % layer_i, activation=activation)(A)
        A = A if dropout_p is None else SpatialDropout1D(dropout_p, name="dropout_%d" % layer_i)(A)
        layer_i += 1

    neigh_wts = space["neigh_wts"]
    for _ in range(space["n_conv_layers"]):
        A = GraphConv(width, name="conv_%d" % layer_i, activation=activation, neigh_wts=neigh_wts)([A, D])
        A = A if dropout_p is None else SpatialDropout1D(dropout_p, name="dropout_%d" % layer_i)(A)
        layer_i += 1

    for _ in range(2):
        A = Dense(width, name="dense_%d" % layer_i, activation=activation)(A)
        A = A if dropout_p is None else SpatialDropout1D(dropout_p, name="dropout_%d" % layer_i)(A)
        layer_i += 1

    A = Dense(n_ys_per_atom, name="dense_last")(A)
    #A = A if not dropout else SpatialDropout1D(dropout_p, name="dropout_last")(A)

    output = Activation(activation="linear", name="output")(A)

    with tf.device("/cpu:0"):
        model = Model(inputs=[X, D], outputs=[output])
    if n_gpus:
        model = multi_gpu_model(model, gpus=n_gpus)
    model.compile(optimizer=Adam(lr=space["lr"], decay=space["decay"]), loss=mean_absolute_error_nans)

    return model


def custom_load_model(model_file):
    return load_model(model_file, custom_objects={
            "mean_absolute_error_nans": mean_absolute_error_nans,
            "GraphConv": GraphConv})


def save_model(model, output_file, weights_only=False):
    ":type model: Model"
    if weights_only:
        model.save_weights(output_file)
    else:
        model.save(output_file, overwrite=True, include_optimizer=True)


def load_weights(model, wts_file):
    ":type model: Model"
    model.load_weights(wts_file)


def clone_model(model, new_max_hac):
    # this is no more required as we can pass shape of the arrays ar None * n_features
    ":type model: Model"
    fd, wts_file = mkstemp()
    try:
        model.save_weights(wts_file)
        print "here"
        X_shape = (None, new_max_hac, int(model.input[0].shape[2]))
        D_shape = (None, new_max_hac, new_max_hac)
        Y_shape = (None, new_max_hac, int(model.output.shape[2]))

        new_model = build_model(X_shape, D_shape, Y_shape)
        new_model.load_weights(wts_file)
        return new_model
    except:
        print "Some error occuered while loading cloning the model"
        raise
    finally:
        os.remove(wts_file)
