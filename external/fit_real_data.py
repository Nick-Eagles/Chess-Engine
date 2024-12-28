#   Following preprocessing in 'pgn_to_test_data.py', this script trains a
#   neural network on the generated real data to see if a policy-value
#   network of the current architecture can learn effectively

from pyhere import here
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.models import load_model
import numpy as np
import pickle
import gzip
import pandas as pd
import re
from plotnine import ggplot, aes, geom_line, theme_bw, labs, ggsave

net_name = "res_deeper"

#   If True, load the existing model and history. If False, construct a new
#   model
resume = True

#   Hyperparameters (ignored if 'resume' is True)
hidden_layer_lens = [500, 500, 500, 500]
loss_weights = [0.08, 0.02, 0.9]
optimizer = 'adam'
batch_size = 100
epochs = 1

train_paths = [
    here(
        'external', 'preprocessed_games', 'g75_no_expit',
        f'tensor_list_train_real{i}.pkl.gz'
    )
    for i in range(1, 20)
]
test_path = here(
    'external', 'preprocessed_games', 'g75_no_expit',
    'tensor_list_test_real.pkl.gz'
)
model_path = Path(here('nets', net_name, 'model.keras'))
history_path = here('nets', net_name, 'history.csv')
metrics_plot_paths = [
    here('nets', net_name, f'{x}.pdf')
    for x in ['policy_square', 'policy_end_piece', 'value']
]

model_path.parent.mkdir(exist_ok = True)

#   Given input and output tensors and a model 'net', return the cosine
#   similarity of the model predictions of reward against the labels.
#   Each sample composes one component of the vectors (e.g. if X is a batch
#   of 100 samples, cosine similarity is taken across two 100-dimensional
#   vectors)
def cos_sim(X, y, net):
    exp_rew = net(X, training = False)[-1].numpy().flatten()
    act_rew = y[-1].numpy().flatten()
    result = float(
        (exp_rew @ act_rew) /
        (np.linalg.norm(exp_rew) * np.linalg.norm(act_rew))
    )
    
    return result

################################################################################
#   Construct a basic neural network (or load an existing one)
################################################################################

if resume:
    net = load_model(model_path)
else:
    #   Input layer
    input_lay = keras.Input(shape = (774,), name = "game_position")
    x = input_lay

    #   Dense hidden layers
    # for hidden_layer_len in hidden_layer_lens:
    #     x = layers.Dense(
    #             hidden_layer_len,
    #             activation = "relu"
    #         )(x)
    #     x = layers.BatchNormalization()(x)

    #   Linear projection to match block input and output lengths
    x = layers.Dense(400, name='linear_projection')(x)

    for i in range(3):
        block_input = x
        x = layers.Dense(
            400,
            activation="relu",
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(
            400,
            activation="relu"
        )(x)

        #   Residual connection, with batch norm afterward
        x = layers.add([x, block_input], name = f'residual_conn{i}')
        x = layers.BatchNormalization()(x)

    #   Output layer
    policy_move_sq = layers.Dense(
            4096, activation = "softmax", name = "policy_move_square"
        )(x)
    policy_end_piece = layers.Dense(
            6, activation = "softmax", name = "policy_end_piece"
        )(x)
    
    for i in range(3):
        x = layers.Dense(50, activation = "relu")(x)
        x = layers.BatchNormalization()(x)
    value = layers.Dense(
            1, kernel_regularizer=regularizers.l2(0.01), name = "value",
        )(x)

    net = keras.Model(
        inputs = input_lay,
        outputs = [policy_move_sq, policy_end_piece, value],
        name = "network"
    )

################################################################################
#   Compile the model
################################################################################

loss = [
    tf.keras.losses.CategoricalCrossentropy(), # policy: move sq
    tf.keras.losses.CategoricalCrossentropy(), # policy: end piece
    tf.keras.losses.MeanSquaredError()         # value
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

#   If resuming, read in the training history. Offset the training step number
#   later by how many training steps were already performed
if resume:
    history_df_old = pd.read_csv(history_path)
    step_offset = int(history_df_old['training_step'].max())
else:
    step_offset = 0

history_df_list = []
for epoch_num in range(epochs):
    print(f'Starting epoch {epoch_num+1}.')
    for batch_num, train_path in enumerate(train_paths):
        print(f'Starting batch {batch_num+1}.')
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
        
        #   Append history to the ongoing list. Add the appropriate training
        #   step number, and evaluate cosine similarity of expected and actual
        #   rewards (not sure how to do this with keras metrics during fitting)
        history_df = pd.DataFrame(history.history)
        history_df['training_step'] = epoch_num * len(train_paths) + \
            batch_num + step_offset + 1
        history_df['value_cos_sim'] = cos_sim(X_train, y_train, net)
        history_df['val_value_cos_sim'] = cos_sim(X_test, y_test, net)
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
    value_name = 'metric_val'
)
history_df['data_group'] = history_df['variable'].str.extract('^(train|val)')
history_df['metric'] = history_df['variable'].apply(
    lambda x: re.sub('^(train|val)_', '', x)
)
history_df.drop('variable', axis = 1, inplace = True)

#   Now pivot wider so that each metric is a column in addition to the
#   'training_step' and 'data_group' columns
history_df = (
    history_df
        .pivot(
            columns = 'metric',
            index = ['training_step', 'data_group'],
            values = 'metric_val'
        )
        .reset_index()
)
history_df.columns.name = None

if resume:
    history_df = pd.concat([history_df, history_df_old], axis = 0)

history_df.to_csv(history_path, index = False)

#-------------------------------------------------------------------------------
#   Plot interesting metrics for policy and value components of outputs
#-------------------------------------------------------------------------------

#   Define parameters to ggplot for each of the three metrics to show
plot_params = [
    {
        'y': 'policy_move_square_categorical_accuracy',
        'lab_y': 'Categorical Accuracy',
        'title': 'Policy Square'
    },
    {
        'y': 'policy_end_piece_categorical_accuracy',
        'lab_y': 'Categorical Accuracy',
        'title': 'Policy End Piece'
    },
    {
        'y': 'value_cos_sim',
        'lab_y': 'Cosine Similarity',
        'title': 'Value'
    }
]

#   Produce the plots
p_list = [
    ggplot(
            history_df,
            aes(
                x = 'training_step', y = x['y'], color = 'data_group',
                group = 'data_group'
            )
        ) +
        geom_line() +
        theme_bw(base_size = 15) +
        labs(
            x = 'Training Step (1000 games each)',
            y = x['lab_y'], color = 'Data Group',
            title = x['title']
        )
    for x in plot_params
]

#   Save plots for each metric separately
for i in range(len(metrics_plot_paths)):
    ggsave(p_list[i], filename = metrics_plot_paths[i], width = 10, height = 5)

net.save(model_path)
