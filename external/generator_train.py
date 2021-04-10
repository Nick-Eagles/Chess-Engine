import sys
import random
import _pickle as pickle
import gzip
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras.callbacks import CSVLogger
sys.path.append('./')
sys.path.append('./external/')
sys.path.append('./experimental/')

import input_handling
import policy_net
import read_pgn

t_dir = 'external/2019_games_t'
v_x_path = 'external/2019_games_tensor_v_x.pkl.gz'
v_y_path = 'external/2019_games_tensor_v_y.pkl.gz'
loss_path = 'visualization/external_generator_costs.csv'
num_groups = 2
blocks_per_group = 1
block_width = 2
n = 308400

def load_v_data(path):
    with gzip.open(path, 'rb') as f:
        data = pickle.load(f)

    return data

p = input_handling.readConfig()
net = policy_net.InitializeNet(num_groups,
                               blocks_per_group,
                               block_width,
                               p,
                               'policy_value')

print('Loading validation data...')
v_data = (load_v_data(v_x_path), load_v_data(v_y_path))

print('Compiling model...')
optim = tf.keras.optimizers.Adam(learning_rate=p['nu'])
policy_net.CompileNet(net, p, optim, 'policy_value')

print('Fitting model...')
csv_logger = CSVLogger(loss_path, append=False)

train_generator = read_pgn.RawDataGen(p['batchSize'], t_dir, n)

new_history = net.fit(
    train_generator,
    epochs = p['epochs'],
    validation_data = v_data,
    callbacks=[csv_logger]
)
