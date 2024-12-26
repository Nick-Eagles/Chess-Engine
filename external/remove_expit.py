from pyhere import here
from pathlib import Path
import tensorflow as tf
from scipy.special import logit
import pickle
import gzip

train_paths_in = [
    here(
        'external', 'preprocessed_games', 'g75',
        f'tensor_list_train_real{i}.pkl.gz'
    )
    for i in range(1, 20)
]
train_paths_out = [
    here(
        'external', 'preprocessed_games', 'g75_no_expit',
        f'tensor_list_train_real{i}.pkl.gz'
    )
    for i in range(1, 20)
]
test_path_in = here(
    'external', 'preprocessed_games', 'g75', 'tensor_list_test_real.pkl.gz'
)
test_path_out = Path(
    here(
        'external', 'preprocessed_games', 'g75_no_expit',
        'tensor_list_test_real.pkl.gz'
    )
)

test_path_out.parent.mkdir(exist_ok = True)

#   First apply logit to test data and re-write
with gzip.open(test_path_in, 'rb') as f:
    X_test, y_test = pickle.load(f)

y_test[-1] = tf.constant(
    logit(y_test[-1].numpy()), shape = y_test[-1].shape, dtype = tf.float32
)

with gzip.open(test_path_out, 'wb') as f:
    pickle.dump((X_test, y_test), f)

#   Now apply logit to training data and re-write
for i in range(len(train_paths_in)):
    with gzip.open(train_paths_in[i], 'rb') as f:
        X_train, y_train = pickle.load(f)

    y_train[-1] = tf.constant(
        logit(y_train[-1].numpy()),
        shape = y_train[-1].shape,
        dtype = tf.float32
    )

    with gzip.open(train_paths_out[i], 'wb') as f:
        pickle.dump((X_train, y_train), f)
