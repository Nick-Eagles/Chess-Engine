import sys
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import CSVLogger
import _pickle as pickle
import gzip

sys.path.append('./')
sys.path.append('./external/')
sys.path.append('./experimental/')
import input_handling
import network_helper
import policy_net

#   Experiment with different architectures and optimizers to determine a
#   reasonable setup, given an 100k-example dataset.

t_x_path = 'external/2019_games_tensor_t_x.pkl.gz'
t_y_path = 'external/2019_games_tensor_t_y.pkl.gz'
v_x_path = 'external/2019_games_tensor_v_x.pkl.gz'
v_y_path = 'external/2019_games_tensor_v_y.pkl.gz'

def load_data(path):
    with gzip.open(path, 'rb') as f:
        data = pickle.load(f)

    return data

def copy_model(model):
    model_copy = keras.models.clone_model(model)
    model_copy.set_weights(model.get_weights())

    return model_copy

def fit_model(model, p, t_x, t_y, v_data, loss_path):
    csv_logger = CSVLogger(loss_path, append=False)
        
    new_history = model.fit(
        t_x,
        t_y,
        batch_size = p['batchSize'],
        epochs = p['epochs'],
        validation_data = v_data,
        callbacks=[csv_logger]
    )

def simple_model(num_layers, lens):
    inputs = keras.Input(shape=(839,), name="game_position")
    x = inputs

    #   Hidden layers
    for i in range(num_layers):
        x = layers.Dense(
            lens[i],
            activation="relu",
            kernel_regularizer=regularizers.l2(p['weightDec'])
        )(x)
        x = layers.BatchNormalization(momentum=p['popPersist'])(x)

    #   Output layer (3 pieces of a policy vector and a scalar value)
    policy_start_sq = layers.Dense(
        64, activation="softmax", name="policy_start_square"
    )(x)
    policy_end_sq = layers.Dense(
        64, activation="softmax", name="policy_end_square"
    )(x)
    policy_end_piece = layers.Dense(
        6, activation="softmax", name="policy_end_piece"
    )(x)

    value = layers.Dense(
        1,
        activation="sigmoid",
        name="output",
        kernel_regularizer=regularizers.l2(p['weightDec'])
    )(x)

    net = keras.Model(
        inputs=inputs,
        outputs=[policy_start_sq, policy_end_sq, policy_end_piece, value],
        name="network"
    )

    return net

#   Load dataset
print("Loading example dataset...")
t_x = load_data(t_x_path)
t_y = load_data(t_y_path)
v_data = (load_data(v_x_path), load_data(v_y_path))

p = input_handling.readConfig()
#p['weightDec'] = 0.0

################################################################################
#   First compare optimizers for what is expected to be a fairly reasonable,
#   "average", architecture choice
################################################################################

#   Initialize 3 identical models
print("Initializing models...")
net2 = policy_net.InitializeNet(1, 2, 2, p, 'policy_value')
#net2 = simple_model(4, [200, 200, 100, 100])
#net2 = copy_model(net)
#net3 = copy_model(net)

#   Compile each model with a different optimizer
print("Compiling models...")
#optim = tf.keras.optimizers.SGD(learning_rate=p['nu'], momentum=p['mom'])
#policy_net.CompileNet(net, p, optim, 'policy_value')

optim = tf.keras.optimizers.Adam(learning_rate=p['nu'])
policy_net.CompileNet(net2, p, optim, 'policy_value')

#optim = tf.keras.optimizers.RMSprop(learning_rate=p['nu'])
#policy_net.CompileNet(net3, p, optim, 'policy_value')

#   Train each model and write losses
print("Fitting models...")
#fit_model(net, p, t_x, t_y, v_data, 'visualization/external_sgd_costs.csv')
fit_model(net2, p, t_x, t_y, v_data, 'visualization/external_adam_costs.csv')
#fit_model(net3, p, t_x, t_y, v_data, 'visualization/external_rmsprop_costs.csv')

print("Saving...")
net2.save('nets/tf_ex_compare/model')

net_attributes = {
    'certainty': 0.25,
    'certaintyRate': 0
}

with open('nets/tf_ex_compare/net_attributes.pkl', 'wb') as f:
    pickle.dump(net_attributes, f)
    
print("Done all tasks.")
