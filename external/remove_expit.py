from pyhere import here
from pathlib import Path
import tensorflow as tf
from scipy.special import logit
import pickle
import gzip
import random

train_paths_in = [
    here(
        'external', 'preprocessed_games', 'g9_sorted',
        f'tensor_list_train_real{i}.pkl.gz'
    )
    for i in range(1, 20)
]
train_paths_out = [
    here(
        'external', 'preprocessed_games', 'g9_no_expit',
        f'tensor_list_train_real{i}.pkl.gz'
    )
    for i in range(1, 20)
]
test_path_in = here(
    'external', 'preprocessed_games', 'g9_sorted', 'tensor_list_test_real.pkl.gz'
)
test_path_out = Path(
    here(
        'external', 'preprocessed_games', 'g9_no_expit',
        'tensor_list_test_real.pkl.gz'
    )
)

test_path_out.parent.mkdir(exist_ok = True)

#   First apply logit to test data and re-write
with gzip.open(test_path_in, 'rb') as f:
    X_test, y_test = pickle.load(f)

permute = list(range(y_test[-1].shape[0]))
random.shuffle(permute)

#   First shuffle examples
X_test = tf.constant(
    X_test.numpy()[permute, :], shape = X_test.shape, dtype = tf.float32
)
y_test = [
    tf.constant(x.numpy()[permute, :], shape = x.shape, dtype = tf.float32)
    for x in y_test
]

#   Now apply logit to the reward component of the output
y_test[-1] = tf.constant(
    logit(y_test[-1].numpy()), shape = y_test[-1].shape, dtype = tf.float32
)

with gzip.open(test_path_out, 'wb') as f:
    pickle.dump((X_test, y_test), f)

#   Now apply logit to training data and re-write
for i in range(len(train_paths_in)):
    with gzip.open(train_paths_in[i], 'rb') as f:
        X_train, y_train = pickle.load(f)

    #   First shuffle examples
    X_train = tf.constant(
        X_train.numpy()[permute, :], shape = X_train.shape, dtype = tf.float32
    )
    y_train = [
        tf.constant(x.numpy()[permute, :], shape = x.shape, dtype = tf.float32)
        for x in y_train
    ]

    #   Now apply logit to the reward component of the output
    y_train[-1] = tf.constant(
        logit(y_train[-1].numpy()),
        shape = y_train[-1].shape,
        dtype = tf.float32
    )

    with gzip.open(train_paths_out[i], 'wb') as f:
        pickle.dump((X_train, y_train), f)
