#   Following preprocessing in 'pgn_to_test_data.py', this script trains a
#   neural network on the generated real data to see if a policy-value
#   network of the current architecture can learn effectively

from pyhere import here
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pickle
import gzip
import pandas as pd
import re
from plotnine import ggplot, aes, geom_line, coord_cartesian, theme_bw, ggsave, facet_wrap, labs

data_path = here('external', 'tensor_list_real.pkl.gz')
model_path = here('nets', 'first.keras')
metrics_plot_path = here('visualization', 'real_metrics.pdf')
hidden_layer_lens = [200, 100, 50]
policy_weight = 0.5
optimizer = 'adam'
batch_size = 100
epochs = 5

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
policy_move_sq = layers.Dense(
        4096, activation = "softmax", name = "policy_move_square"
    )(x)
policy_end_piece = layers.Dense(
        6, activation = "softmax", name = "policy_end_piece"
    )(x)
value = layers.Dense(
        1, activation = "sigmoid", name = "value",
    )(x)

net = keras.Model(
    inputs = input_lay,
    outputs = [policy_move_sq, policy_end_piece, value],
    name = "network"
)

################################################################################
#   Compile and fit the model on the data
################################################################################

loss_weights = [policy_weight / 2, policy_weight / 2, 1 - policy_weight]
loss = [
    tf.keras.losses.CategoricalCrossentropy(), # policy: move sq
    tf.keras.losses.CategoricalCrossentropy(), # policy: end piece
    tf.keras.losses.BinaryCrossentropy()       # value
]

net.compile(
    optimizer = optimizer,
    loss = loss,
    loss_weights = loss_weights,
    metrics = [
        tf.keras.metrics.CategoricalAccuracy(),
        tf.keras.metrics.CategoricalAccuracy(),
        None
    ]
)

history = net.fit(
    X_train,
    y_train,
    batch_size = batch_size,
    epochs = epochs,
    validation_data = (X_test, y_test)
)

################################################################################
#   Plot history info
################################################################################

#-------------------------------------------------------------------------------
#   Gather history.history into a tidy DataFrame
#-------------------------------------------------------------------------------

#   Convert history info to DataFrame, making sure column names start with
#   either 'val_' or 'train_'
history_df = pd.DataFrame(history.history)
history_df.rename(
    lambda x: re.sub(r'^(?!val_)', 'train_', x), axis = 1, inplace = True
)

#   Add epoch as a column
history_df.index.name = 'epoch'
history_df.reset_index(inplace = True)

#   Reformat longer, such that as columns we have 'epoch', 'data_group' (train
#   or val), metric name, and value for that metric
history_df = pd.melt(
    history_df,
    id_vars = 'epoch',
    value_vars = history_df.drop('epoch', axis = 1).columns,
    value_name = 'categorical_accuracy'
)
history_df['data_group'] = history_df['variable'].str.extract('^(train|val)')
history_df['metric'] = history_df['variable'].apply(
    lambda x: re.sub('^(train|val)_', '', x)
)
history_df.drop('variable', axis = 1, inplace = True)

#   Now only take the categorical accuracy metrics and construct a column
#   replacing 'metric' that just has the output-layer name (e.g. 
#   'policy_end_piece')
history_df = history_df[
    history_df['metric'].str.match(r'.*_categorical_accuracy$')
]
history_df['output_type'] = history_df['metric'].apply(
    lambda x: re.sub(r'_categorical_accuracy$', '', x)
)
history_df.drop('metric', axis = 1, inplace = True)

#-------------------------------------------------------------------------------
#   Plot categorical accuracy for each output type across epochs
#-------------------------------------------------------------------------------

p = (
    ggplot(
            history_df,
            aes(
                x = 'epoch', y = 'categorical_accuracy', color = 'output_type',
                group = 'output_type'
            )
        ) +
        geom_line() +
        facet_wrap('data_group') +
        # coord_cartesian(ylim = [0, 1]) +
        theme_bw(base_size = 15) +
        labs(
            x = 'Epoch Number', y = 'Categorical Accuracy',
            color = 'Output Type'
        )
)
ggsave(p, filename = metrics_plot_path, width = 10, height = 5)

net.save(model_path)
