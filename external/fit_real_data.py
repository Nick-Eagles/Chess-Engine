#   Following preprocessing in 'pgn_to_test_data.py' and
#   'tactics_to_real_data.py', this script trains a neural network on the
#   generated real data to see if a policy-value network of the current
#   architecture can learn effectively

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

net_name = "first_conv"

#   If True, load the existing model and history. If False, construct a new
#   model
resume = False

#   Per batch
num_tactics_positions = 20000

#   Hyperparameters (ignored if 'resume' is True)
num_filters = 40
num_residual_blocks = 4
loss_weights = [0.08, 0.02, 0.9]
optimizer = 'adam'
batch_size = 64
epochs = 1
num_batches = 8

#   How many training batches pass before cosine similarity is computed on
#   the training and validation data
calc_train_similarity_period = 4
calc_val_similarity_period = 2

train_game_paths = [
    here(
        'external', 'preprocessed_games', 'g75_conv',
        f'train{i+1}.pkl.gz'
    )
    for i in range(num_batches)
]
test_game_path = here(
    'external', 'preprocessed_games', 'g75_conv', 'test.pkl.gz'
)
train_tactics_paths = [
    here('external', 'preprocessed_tactics', 'g75_conv', f'train{i+1}.pkl.gz')
    for i in range(num_batches)
]
test_tactics_path = here(
    'external', 'preprocessed_tactics', 'g75_conv', 'test.pkl.gz'
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
#   vectors). Perform and average across batches to save memory
def cos_sim(X, y, net, num_batches = 5):
    result = 0
    batch_size = X.shape[0] // num_batches
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        exp_rew = net(X[start_index: start_index + batch_size], training = False)[-1].numpy().flatten()
        act_rew = y[-1][start_index: start_index + batch_size].numpy().flatten()
        result += float(
            (exp_rew @ act_rew) /
            (np.linalg.norm(exp_rew) * np.linalg.norm(act_rew))
        )
    
    return result / num_batches

################################################################################
#   Construct a basic neural network (or load an existing one)
################################################################################

if resume:
    net = load_model(model_path)
else:
    #---------------------------------------------------------------------------
    #   Input layer
    #---------------------------------------------------------------------------

    input_lay = keras.Input(shape = (8, 8, 15), name = "game_position")
    x = input_lay

    #   Perform one convolution so residual block inputs and outputs match
    #   shape
    x = layers.Conv2D(
        num_filters,
        kernel_size = (3, 3),
        padding = "same",
        data_format = "channels_last"
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    #---------------------------------------------------------------------------
    #   Residual blocks
    #---------------------------------------------------------------------------

    for i in range(num_residual_blocks):
        block_input = x

        #   Two dense layers
        for j in range(2):
            x = layers.Conv2D(
                num_filters,
                kernel_size = (3, 3),
                padding = "same",
                data_format = "channels_last"
            )(x)
            x = layers.BatchNormalization()(x)
            x = layers.ReLU()(x)

        #   Residual connection
        x = layers.add([x, block_input], name = f'residual_conn{i}')
    
    #---------------------------------------------------------------------------
    #   Data compression
    #---------------------------------------------------------------------------

    #   Perform a convolution with fewer filters, pool to a 4x4 board, and
    #   flatten to control the number of weights (mostly between the first flat
    #   layer and the policy square)
    # x = layers.Conv2D(
    #     20,
    #     kernel_size = (3, 3),
    #     padding = "same",
    #     data_format = "channels_last"
    # )(x)
    # x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(pool_size = (2, 2))(x)
    x = layers.Flatten()(x)

    #---------------------------------------------------------------------------
    #   Output layer
    #---------------------------------------------------------------------------

    policy_move_sq = layers.Dense(
            4096, activation = "softmax", name = "policy_move_square"
        )(x)
    policy_end_piece = layers.Dense(
            6, activation = "softmax", name = "policy_end_piece"
        )(x)

    for i in range(2):
        x = layers.Dense(50)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
    value = layers.Dense(
            1, kernel_regularizer=regularizers.l2(0.01), name = "value",
        )(x)

    net = keras.Model(
        inputs = input_lay,
        outputs = [policy_move_sq, policy_end_piece, value],
        name = "network"
    )

    net.summary()

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
        tf.keras.metrics.TopKCategoricalAccuracy(
            k = 3, name = 'categorical_accuracy'
        ),
        tf.keras.metrics.CategoricalAccuracy(),
        None
    ]
)

################################################################################
#   Fit the model
################################################################################

#   Because the full training set is too large to fit in memory on a 8GB laptop,
#   training is performed in batches consisting of 1000 games and up to 40000
#   tactical positions

#   Load the 1000-game test set and tactical test set, then combine
print('Loading test positions...')
with gzip.open(test_game_path, 'rb') as f:
    X_test, y_test = pickle.load(f)

with gzip.open(test_tactics_path, 'rb') as f:
    X_test_t, y_test_t = pickle.load(f)

X_test = tf.concat([X_test, X_test_t[:num_tactics_positions, :]], axis = 0)
y_test = [
    tf.concat([x, y[:num_tactics_positions, :]], axis = 0)
    for x, y in zip(y_test, y_test_t)
]

#   If resuming, read in the training history. Offset the training step number
#   later by how many training steps were already performed
if resume:
    history_df_old = pd.read_csv(history_path)
    step_offset = int(history_df_old['training_step'].max())
else:
    step_offset = 0

history_df_list = []
for epoch_num in range(epochs):
    for batch_num in range(num_batches):
        print(f'Starting epoch {epoch_num+1}, batch {batch_num+1}.')
        #   Load 1000-game training batch and tactical training batch, then
        #   combine
        with gzip.open(train_game_paths[batch_num], 'rb') as f:
            X_train, y_train = pickle.load(f)
        
        with gzip.open(train_tactics_paths[batch_num], 'rb') as f:
            X_train_t, y_train_t = pickle.load(f)
        
        X_train = tf.concat(
            [X_train, X_train_t[:num_tactics_positions, :]], axis = 0
        )
        y_train = [
            tf.concat([x, y[:num_tactics_positions, :]], axis = 0)
            for x, y in zip(y_train, y_train_t)
        ]
        
        #   Train
        history = net.fit(
            X_train,
            y_train,
            batch_size = batch_size,
            epochs = 1,
            validation_data = (X_test, y_test),
            shuffle = True
        )
        
        #   Append history to the ongoing list. Add the appropriate training
        #   step number, and evaluate cosine similarity of expected and actual
        #   rewards (not sure how to do this with keras metrics during fitting)
        history_df = pd.DataFrame(history.history)
        history_df['training_step'] = epoch_num * num_batches + \
            batch_num + step_offset + 1
        
        if (epoch_num * num_batches + batch_num) % calc_train_similarity_period == 0:
            print("Calculating training cos. similarity...")
            history_df['value_cos_sim'] = cos_sim(X_train, y_train, net)
        else:
            history_df['value_cos_sim'] = np.nan
        
        if (epoch_num * num_batches + batch_num) % calc_val_similarity_period == 0:
            print("Calculating validation cos. similarity...")
            history_df['val_value_cos_sim'] = cos_sim(X_test, y_test, net)
        else:
            history_df['val_value_cos_sim'] = np.nan

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
        'lab_y': 'Top-3 Categorical Accuracy',
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
