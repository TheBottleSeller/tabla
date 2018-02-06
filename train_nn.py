"""Tabla Neural Network"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import argparse
import sys
import pandas as pd
import sklearn.model_selection as sk

tf.logging.set_verbosity(tf.logging.INFO)

parser = argparse.ArgumentParser(description='Neural network for pneumonia likelihood')
parser.add_argument('--input', type=str, help='input file')
args = parser.parse_args()

if not input:
    print("Missing input data")
    sys.exit(-1)

# Configurable network parameters
training_epochs = 100
learning_rate = 0.1
n_hidden_1 = 5  # 1st layer number of neurons
n_hidden_2 = 3  # 2nd layer number of neurons
test_percent = 0.5
diagnosis_threshold = 0.6

def nn_model_fn(features, labels, mode):
  """Model function for NN."""
  # Hidden fully connected layer
  layer_1 = tf.layers.dense(features["x"], n_hidden_1)

  # Hidden fully connected layer
  layer_2 = tf.layers.dense(layer_1, n_hidden_2)

  # Use a sigmoid activiation function to get output from 0 to 1
  diagnosis_probabilities = tf.layers.dense(layer_2, 1, activation=tf.nn.sigmoid, name="diagnosis_probabilities")

  diagnosis_pred_comparison = tf.greater(diagnosis_probabilities, tf.constant(diagnosis_threshold, dtype=tf.float64))
  diagnosis_pred = tf.where(diagnosis_pred_comparison,
    tf.ones_like(diagnosis_probabilities, dtype=tf.float64),
    tf.zeros_like(diagnosis_probabilities, dtype=tf.float64),
    name="predictions"
  )

  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "diagnosis": diagnosis_pred,
      "probabilities": diagnosis_probabilities
  }
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
  diff = tf.subtract(labels, diagnosis_probabilities, name='diff')
  loss = tf.reduce_sum(diff * diff)

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
        labels=labels, predictions=predictions["diagnosis"]
      )
  }
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(unused_argv):
    # Load training and eval data
    data = pd.read_csv(args.input, header=None)
    num_cols = len(data.columns)

    x_cols = range(0, num_cols - 1)
    y_cols = [num_cols - 1]

    healthy_data = data.loc[data[num_cols - 1] == 0]
    healthy_data_xs = healthy_data.drop(labels=y_cols, axis=1)
    healthy_data_ys = healthy_data.drop(labels=x_cols, axis=1)
    healthy_x_train, healthy_x_test, healthy_y_train, healthy_y_test = \
        sk.train_test_split(healthy_data_xs,healthy_data_ys,test_size=1, random_state = 42, shuffle=True)

    sick_data = data.loc[data[num_cols - 1] == 1]
    sick_data_xs = sick_data.drop(labels=y_cols, axis=1)
    sick_data_ys = sick_data.drop(labels=x_cols, axis=1)
    sick_x_train, sick_x_test, sick_y_train, sick_y_test = \
        sk.train_test_split(sick_data_xs,sick_data_ys,test_size=1, random_state = 42, shuffle=True)

    x_train = healthy_x_train.append(sick_x_train).as_matrix()
    x_test = healthy_x_test.append(sick_x_test).as_matrix()
    y_train = healthy_y_train.append(sick_y_train).as_matrix()
    y_test = healthy_y_test.append(sick_y_test).as_matrix()

    # Create the Estimator
    tabla_classifier = tf.estimator.Estimator(
      model_fn=nn_model_fn, model_dir="./tabla_nn_model")

    # Set up logging for predictions
    # Log the values in the "diagnosis_probabilities" tensor with label "probabilities"
    tensors_to_log = {
        "probabilities": "diagnosis_probabilities/Sigmoid",
        # "predictions": "predictions",
        # "diff": "diff"
    }
    logging_hook = tf.train.LoggingTensorHook(
      tensors=tensors_to_log, every_n_iter=5)

    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": x_train},
      y=y_train,
      batch_size=x_train.shape[0],
      num_epochs=training_epochs,
      shuffle=True
    )
    tabla_classifier.train(
      input_fn=train_input_fn,
      hooks=[logging_hook]
    )

    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": x_test},
      y=y_test,
      num_epochs=1,
      shuffle=False
    )
    eval_results = tabla_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)

if __name__ == "__main__":
  tf.app.run()
