# This is the complete code for the following blogpost:
# https://developers.googleblog.com/2017/09/introducing-tensorflow-datasets.html
# Create an input function reading a file using the Dataset API
# Then provide the results to the Estimator API
# https://www.tensorflow.org/api_docs/python/tf/data
# https://www.tensorflow.org/get_started/input_fn
# https://www.tensorflow.org/api_docs/python/tf/feature_column
# https://www.tensorflow.org/api_docs/python/tf/estimator


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


PATH = "/tmp/tf_dataset_and_estimator_apis"

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


# The CSV features in our training & test data
FEATURE_NAMES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']

# Create the feature_columns, which specifies the input to our model
feature_columns = [tf.feature_column.numeric_column(k) for k in FEATURE_NAMES]

# Use the DNNClassifier pre-made estimator
classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns,
                                        hidden_units=[10, 10],
                                        n_classes=3,
                                        model_dir=PATH)

# input_fn
# The user can do feature engineering or pre-processing inside the input_fn.
# sess.run(next_batch) returns 32 random elements for
# next_batch = my_input_fn(FILE_TRAIN, True)
def my_input_fn(file_path, perform_shuffle=False, repeat_count=1, batch_size=32):
    def decode_csv(line):
        parsed_line = tf.decode_csv(line, [[0.], [0.], [0.], [0.], [0]])
        label = parsed_line[-1:]  # Last element is the label
        del parsed_line[-1]  # Delete last element
        features = parsed_line
        d = dict(zip(FEATURE_NAMES, features)), label
        return d

    dataset = (tf.data.TextLineDataset(file_path)  # Read text file
               .skip(1)  # Skip header row
               .map(decode_csv))  # Transform each elem by applying decode_csv fn
    if perform_shuffle:
        # Randomizes input using a window of 256 elements (read into memory)
        dataset = dataset.shuffle(buffer_size=256)
    dataset = dataset.repeat(repeat_count)  # Repeats dataset this # times
    dataset = dataset.batch(batch_size)  # Batch size to use
    # make an iterator that provides access to one element of the dataset
    iterator = dataset.make_one_shot_iterator()
    batch_features, batch_labels = iterator.get_next()
    return batch_features, batch_labels


# Train our model
# Train until the OutOfRange error or StopIteration exception occurs.
# Stop training after 8 iterations of train data (epochs)
n_epochs = 8
classifier.train(input_fn=lambda: my_input_fn(FILE_TRAIN, True, n_epochs))


# Evaluate our model
# Return value will contain evaluation_metrics such as: loss & average_loss
n_epochs = 4
evaluate_result = classifier.evaluate(
    input_fn=lambda: my_input_fn(FILE_TEST, False, n_epochs))
print("Evaluation results")
for key in evaluate_result:
    print("   {}, was: {}".format(key, evaluate_result[key]))


# Predict the type of some Iris flowers in FILE_TEST
n_epochs = 1
predict_results = classifier.predict(
    input_fn=lambda: my_input_fn(FILE_TEST, False, n_epochs))
print("Predictions on test file")
for prediction in predict_results:
    print(prediction["class_ids"][0])

# Let create a dataset for prediction
# We've taken the first 3 examples in FILE_TEST
prediction_input = [[5.9, 3.0, 4.2, 1.5],  # -> 1, Iris Versicolor
                    [6.9, 3.1, 5.4, 2.1],  # -> 2, Iris Virginica
                    [5.1, 3.3, 1.7, 0.5]]  # -> 0, Iris Sentosa

def new_input_fn(input_slices):
    def decode(x):
        n_features = len(FEATURE_NAMES)
        x = tf.split(x, n_features)
        return dict(zip(FEATURE_NAMES, x))

    dataset = tf.data.Dataset.from_tensor_slices(input_slices)
    dataset = dataset.map(decode)
    iterator = dataset.make_one_shot_iterator()
    next_feature_batch = iterator.get_next()
    return next_feature_batch, None

# Predict all our prediction_input
predict_results = classifier.predict(input_fn=lambda: new_input_fn(prediction_input))

# Print results
print("Predictions on memory")
for idx, prediction in enumerate(predict_results):
    type = prediction["class_ids"][0]  # Get the predicted class (index)
    if type == 0:
        print("I think: {}, is Iris Sentosa".format(prediction_input[idx]))
    elif type == 1:
        print("I think: {}, is Iris Versicolor".format(prediction_input[idx]))
    else:
        print("I think: {}, is Iris Virginica".format(prediction_input[idx]))