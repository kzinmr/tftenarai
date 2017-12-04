# Instead of sub-classing Estimator,
# we simply provide Estimator a function `model_fn`
# that tells `tf.estimator` how it can evaluate pred, loss and opt.
# https://www.tensorflow.org/api_docs/python/tf/estimator


import numpy as np
import tensorflow as tf
# Check that we have correct TensorFlow version installed
tf_version = tf.__version__
print("TensorFlow version: {}".format(tf_version))
assert "1.4" <= tf_version, "TensorFlow r1.4 or later is needed"


def huber_loss(labels, predictions, delta=1.0):
    residual = tf.abs(predictions - labels)
    condition = tf.less(residual, delta)
    small_res = 0.5 * tf.square(residual)
    large_res = delta * residual - 0.5 * tf.square(delta)
    return tf.where(condition, small_res, large_res)


def linear_regressor_huber(features, labels, mode):
    # Build a linear model and predict values
    W = tf.get_variable("W", [1], dtype=tf.float32)
    b = tf.get_variable("b", [1], dtype=tf.float32)
    y = W*features['x'] + b
    # Loss sub-graph using Huber loss instead of MSE
    loss = tf.reduce_sum(huber_loss(labels, y)) # tf.square(y - labels)
    # Training sub-graph
    global_step = tf.train.get_global_step()
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = tf.group(optimizer.minimize(loss), tf.assign_add(global_step, 1))
    # EstimatorSpec connects subgraphs we built to the appropriate functionality
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=y,
        loss=loss,
        train_op=train)

# Declare list of feature instead of feature_columns
estimator = tf.estimator.Estimator(model_fn=linear_regressor_huber)

# We have to tell the input function `num_epochs` and `batch_size`
x_train = np.array([1., 2., 3., 4.]).astype(np.float32)
y_train = np.array([0., -1., -2., -3.]).astype(np.float32)
x_eval = np.array([2., 5., 8., 1.]).astype(np.float32)
y_eval = np.array([-1.01, -4.1, -7., 0.]).astype(np.float32)

input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_train}, y_train, batch_size=4, num_epochs=None, shuffle=True)
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_train}, y_train, batch_size=4, num_epochs=1000, shuffle=False)
eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_eval}, y_eval, batch_size=4, num_epochs=1, shuffle=False)

# train (invoke 1000 training steps while `num_epochs=None` in input_fn)
estimator.train(input_fn=input_fn, steps=1000)

# evaluate (num_epochs is specified in train/eval_input_fn)
train_metrics = estimator.evaluate(input_fn=train_input_fn)
eval_metrics = estimator.evaluate(input_fn=eval_input_fn)
print("train metrics: %r"% train_metrics)
print("eval metrics: %r"% eval_metrics)