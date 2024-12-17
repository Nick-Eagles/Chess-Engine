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
from plotnine import ggplot, aes, geom_line, theme_bw, ggsave, facet_wrap, labs

train_paths = [
    here('external', 'preprocessed_games', f'tensor_list_train_real{i}.pkl.gz')
    for i in range(1, 20)
]
test_path = here(
    'external', 'preprocessed_games', 'tensor_list_test_real.pkl.gz'
)
model_path = here('nets', 'first.keras')
metrics_plot_path = here('visualization', 'real_metrics.pdf')
hidden_layer_lens = [200, 100, 50]
policy_weight = 0.5
optimizer = 'adam'
batch_size = 100
epochs = 1

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
#   Compile the model
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

################################################################################
#   Fit the model
################################################################################

#   Because the full training set is too large to fit in memory on a 8GB laptop,
#   training is performed in 1000-game batches

#   Load the 1000-game test set
with gzip.open(test_path, 'rb') as f:
    X_test, y_test = pickle.load(f)

history_df_list = []
for epoch_num in range(epochs):
    for batch_num, train_path in enumerate(train_paths[:2]):
        #   Load 1000-game training batch
        with gzip.open(train_path, 'rb') as f:
            X_train, y_train = pickle.load(f)
        
        #   Train
        history = net.fit(
            X_train,
            y_train,
            batch_size = batch_size,
            epochs = 1,
            validation_data = (X_test, y_test)
        )
        
        #   Append history, with the appropriate training step number, to the
        #   ongoing list
        history_df = pd.DataFrame(history.history)
        history_df['training_step'] = [
            epoch_num * len(train_paths) + batch_num + i + 1
            for i in range(history_df.shape[0])
        ]
        history_df_list.append(history_df)

################################################################################
#   Plot history info
################################################################################

#-------------------------------------------------------------------------------
#   Gather history.history into a tidy DataFrame
#-------------------------------------------------------------------------------

history_df = pd.concat(history_df_list, axis = 0)

#   Make sure column names start with either 'val_' or 'train_'
history_df.rename(
    lambda x: re.sub(r'^(?!val_|training_step)', 'train_', x),
    axis = 1, inplace = True
)

#   Reformat longer, such that as columns we have 'training_step', 'data_group'
#   (train or val), metric name, and value for that metric
history_df = pd.melt(
    history_df,
    id_vars = 'training_step',
    value_vars = history_df.drop('training_step', axis = 1).columns,
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
#   Plot categorical accuracy for each output type across training steps
#-------------------------------------------------------------------------------

p = (
    ggplot(
            history_df,
            aes(
                x = 'training_step', y = 'categorical_accuracy',
                color = 'output_type', group = 'output_type'
            )
        ) +
        geom_line() +
        facet_wrap('data_group') +
        theme_bw(base_size = 15) +
        labs(
            x = 'Training Step Number (1000 games each)',
            y = 'Categorical Accuracy', color = 'Output Type'
        )
)
ggsave(p, filename = metrics_plot_path, width = 10, height = 5)

net.save(model_path)
