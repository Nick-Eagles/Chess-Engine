#   Following preprocessing in 'pgn_to_test_data.py', this script trains a
#   neural network on the generated semi-synthetic data to validate that
#   the general approach works (a policy and value network of the
#   architecture used throughout this repo can effectively learn
#   primitive patterns)

from pyhere import here
import sys
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pickle
import gzip

sys.path.append(str(here()))

data_path = here('external', 'tensor_list.pkl.gz')
hidden_layer_lens = [200, 100, 50]
policy_weight = 0.5
optimizer = 'adam'
batch_size = 100
epochs = 20

#   Load semi-synthetic data into training and test sets
with gzip.open(data_path, 'rb') as f:
    X_train, X_test, y_train, y_test = pickle.load(f)


################################################################################
#   Construct a basic neural network
################################################################################

#   Input layer
input_lay = keras.Input(shape = (839,), name = "game_position")
x = input_lay

#   Dense hidden layers
for hidden_layer_len in hidden_layer_lens:
    x = layers.Dense(
            hidden_layer_len,
            activation = "relu"
        )(x)

#   Output layer
policy_start_sq = layers.Dense(
        64, activation = "softmax", name = "policy_start_square"
    )(x)
policy_end_sq = layers.Dense(
        64, activation = "softmax", name = "policy_end_square"
    )(x)
policy_end_piece = layers.Dense(
        6, activation = "softmax", name = "policy_end_piece"
    )(x)
value = layers.Dense(
        1, activation = "sigmoid", name = "value",
    )(x)

net = keras.Model(
    inputs = input_lay,
    outputs = [policy_start_sq, policy_end_sq, policy_end_piece, value],
    name = "network"
)

################################################################################
#   Compile and fit the model on the data
################################################################################

loss_weights = [
    policy_weight / 3, policy_weight / 3, policy_weight / 3, 1 - policy_weight
]
loss = [
    tf.keras.losses.CategoricalCrossentropy(), # policy: start sq
    tf.keras.losses.CategoricalCrossentropy(), # policy: end sq
    tf.keras.losses.CategoricalCrossentropy(), # policy: end piece
    tf.keras.losses.BinaryCrossentropy()       # value
]

net.compile(
    optimizer = optimizer,
    loss = loss,
    loss_weights = loss_weights,
    metrics = [tf.keras.metrics.CategoricalAccuracy() for i in range(4)]
)

history = net.fit(
    X_train,
    y_train,
    batch_size = batch_size,
    epochs = epochs,
    validation_data = (X_test, y_test)
)
