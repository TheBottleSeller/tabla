""" Tabla Neural Network.
A 2-Hidden Layers Fully Connected Neural Network (a.k.a Multilayer Perceptron)
implementation with TensorFlow.
"""

from __future__ import print_function

import tensorflow as tf
import argparse
import sys
import pandas as pd
import numpy as np
import sklearn.model_selection as sk

parser = argparse.ArgumentParser(description='Neural network for pneumonia likelihood')
parser.add_argument('--input', type=str, help='input file')
args = parser.parse_args()

if not input:
    print("Missing input data")
    sys.exit(-1)

# Configurable network parameters
training_epochs = 50
learning_rate = 0.1
n_hidden_1 = 8  # 1st layer number of neurons
n_hidden_2 = 4  # 2nd layer number of neurons
test_percent = 0.1

# Read the data
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

x_train = healthy_x_train.append(sick_x_train)
x_test = healthy_x_test.append(sick_x_test)
y_train = healthy_y_train.append(sick_y_train)
y_test = healthy_y_test.append(sick_y_test)

# Unconfigurable network parameters
num_input = num_cols - 1   # input vector size, number of columns - 1 (last is output label)
num_classes = 1                         # output classes (lung disease or not)

x = tf.placeholder("float", [None, num_input])
y = tf.placeholder("float", [None, num_classes])

# Define the neural network
def neural_net():
    # Hidden fully connected layer
    layer_1 = tf.layers.dense(x, n_hidden_1)

    # Hidden fully connected layer
    layer_2 = tf.layers.dense(layer_1, n_hidden_2)

    # Use a sigmoid activiation function to get output from 0 to 1
    out_layer = tf.layers.dense(layer_2, num_classes, activation=tf.nn.sigmoid)
    return out_layer

# Build the neural network
pred = neural_net()

# Define loss and optimizer
loss_func = tf.nn.l2_loss(pred-y,name="squared_error_cost")
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
error = optimizer.minimize(loss_func)

# Launch the graph
init = tf.initialize_all_variables()
with tf.Session() as sess:
    sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        e = sess.run([error], feed_dict={
            x: x_train,
            y: y_train,
        })
        print(e)
        print("Epoch: %d. Error: %v" % (epoch, e))
