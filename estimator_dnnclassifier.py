# `tf.estimator.inputs` provides numpy_input_fn and pandas_input_fn
# refer `tf.data` to read data from file
# https://www.tensorflow.org/api_docs/python/tf/estimator/inputs/numpy_input_fn
# https://www.tensorflow.org/api_docs/python/tf/estimator
# https://www.tensorflow.org/extend/estimators


import numpy as np
import tensorflow as tf
import os
import sys
if sys.version_info < (3, 0, 0):
    from urllib import urlopen
else:
    from urllib.request import urlopen

# Check that we have correct TensorFlow version installed
tf_version = tf.__version__
print("TensorFlow version: {}".format(tf_version))
assert "1.4" <= tf_version, "TensorFlow r1.4 or later is needed"

PATH = "/tmp/iris_model"

# Fetch and store Data sets
PATH_DATASET = PATH + os.sep + "dataset"
FILE_TRAIN = PATH_DATASET + os.sep + "iris_training.csv"
FILE_TEST = PATH_DATASET + os.sep + "iris_test.csv"
URL_TRAIN = "http://download.tensorflow.org/data/iris_training.csv"
URL_TEST = "http://download.tensorflow.org/data/iris_test.csv"

def downloadDataset(url, file):
    if not os.path.exists(PATH_DATASET):
        os.makedirs(PATH_DATASET)
    if not os.path.exists(file):
        data = urlopen(url).read()
        with open(file, "wb") as f:
            f.write(data)
            f.close()
downloadDataset(URL_TRAIN, FILE_TRAIN)
downloadDataset(URL_TEST, FILE_TEST)

tf.logging.set_verbosity(tf.logging.INFO)


# Specify that all features have real-value data
feature_columns = [tf.feature_column.numeric_column("x", shape=[4])]

# Build 3 layer DNN with 10, 20, 10 units respectively.
classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns,
                                        hidden_units=[10, 20, 10],
                                        n_classes=3,
                                        model_dir=PATH)

# input_fn
# Load training datasets as a namedtuple(on memory)
training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
    filename=FILE_TRAIN,
    target_dtype=np.int,
    features_dtype=np.float32)
# Get a function that generates training inputs
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": np.array(training_set.data)},
    y=np.array(training_set.target),
    batch_size=32,
    num_epochs=None,
    shuffle=True)

# Train model. 
# 2000 steps for which to train model.
classifier.train(input_fn=train_input_fn, steps=2000)


# Load test datasets as a namedtuple(on memory)
test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
    filename=FILE_TEST,
    target_dtype=np.int,
    features_dtype=np.float32)
# Define the test inputs
test_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": np.array(test_set.data)},
    y=np.array(test_set.target),
    num_epochs=1,
    shuffle=False)

# Evaluate accuracy.
accuracy_score = classifier.evaluate(input_fn=test_input_fn)["accuracy"]

print("\nTest Accuracy: {0:f}\n".format(accuracy_score))


# Classify two new flower samples.
new_samples = np.array(
    [[6.4, 3.2, 4.5, 1.5],
     [5.8, 3.1, 5.0, 1.7]], dtype=np.float32)
predict_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": new_samples},
    num_epochs=1,
    shuffle=False)

predictions = list(classifier.predict(input_fn=predict_input_fn))
predicted_classes = [p["classes"] for p in predictions]

print("New Samples, Class Predictions:    {}\n".format(predicted_classes))
