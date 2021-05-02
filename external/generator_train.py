import sys
import csv
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
import Session
import q_learn
import buffer as Buffer

t_dir = 'external/2019_games_t'
v_x_path = 'external/2019_games_tensor_v_x.pkl.gz'
v_y_path = 'external/2019_games_tensor_v_y.pkl.gz'
loss_path = 'visualization/external_generator_costs.csv'
hyper_path = 'external/gen_hyperparams.csv'
net_dir = 'nets/tf_ex_gen'
make_new_net = False
num_groups = 2
blocks_per_group = 1
block_width = 2
n = 944800

def load_data(path):
    with gzip.open(path, 'rb') as f:
        data = pickle.load(f)

    return data

p = input_handling.readConfig()

if make_new_net:
    net = policy_net.InitializeNet(num_groups,
                                   blocks_per_group,
                                   block_width,
                                   p,
                                   'policy_value')
else:
    session = Session.Session([[]], [[]])
    session.Load(net_dir, lazy=True)
    net = session.net

print('Loading validation data...')
v_data = (load_data(v_x_path), load_data(v_y_path))

print('Compiling model...')
optim = tf.keras.optimizers.Adam(learning_rate=p['nu'])
policy_net.CompileNet(net, p, optim, 'policy_value')

print('Fitting model...')
csv_logger = CSVLogger(loss_path, append = not make_new_net)

train_generator = read_pgn.RawDataGen(p['batchSize'], t_dir, n)

new_history = net.fit(
    train_generator,
    epochs = p['epochs'],
    validation_data = v_data,
    callbacks=[csv_logger]
)

print('Reshaping data for computing certainty...')

#   Format external data as a valid buffer
buffer = [[]]
for i in range(v_data[0].shape[0]):
    #   Get this particular example as a tuple (input, label)
    x = tf.reshape(v_data[0][i, :], [1, 839])
    y = [tf.reshape(v_data[1][0][i, :], [1, 64]),
         tf.reshape(v_data[1][1][i, :], [1, 64]),
         tf.reshape(v_data[1][2][i, :], [1, 6]),
         tf.reshape(v_data[1][3][i, :], [1, 1])]
    
    buffer[0].append((x, y))

Buffer.verify(buffer, p, numBuffs=1)

#   Compute certainty and update model attributes
print('Computing certainty...')
p['persist'] = 0
q_learn.getCertainty(net, buffer, p, greedy=False)
net.certaintyRate = 0

#   Save network without any associated data
print('Saving model...')
session = Session.Session([[]], [[]], net)
session.Save(net_dir)

#   Record hyperparams and latest accuracy achieved
print('Recording results...')
lay_indices = [1 + (2 + 2 * blocks_per_group * block_width) * i
               for i in range(num_groups)]
lay_lens = [net.layers[i].output_shape[-1] for i in lay_indices]
lay_lens_str = ','.join([str(x) for x in lay_lens])

last_acc = round(new_history.history['val_policy_end_square_categorical_accuracy'][-1],
                 4)

hyperparams = [[num_groups,
               blocks_per_group,
               block_width,
               lay_lens_str,
               p['nu'],
               p['batchSize'],
               p['weightDec'],
               p['epochs'],
               last_acc]]

with open(hyper_path, 'a') as f:
    writer = csv.writer(f)
    writer.writerows(hyperparams)

print('Done.')
