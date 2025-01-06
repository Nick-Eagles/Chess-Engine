#   Try training a policy-value neural network on real data: predicting the move
#   chosen by high-rated players, and predict the expected reward from each
#   position 

from pyhere import here
from pathlib import Path
import sys
import pickle
import gzip

sys.path.append(str(here('external')))
sys.path.append(str(here()))

import input_handling
import read_pgn

train_paths = [
    here('external', 'preprocessed_games', f'train_games{i}.txt.gz')
    for i in range(26, 27)
]
test_path = here('external', 'preprocessed_games', 'test_games.txt.gz')
out_train_paths = [
    here(
        'external', 'preprocessed_games', 'g75_conv',
        f'train{i}.pkl.gz'
    )
    for i in range(26, 27)
]
out_test_path = Path(
    here('external', 'preprocessed_games', 'g75_conv', 'test.pkl.gz')
)

out_test_path.parent.mkdir(exist_ok = True)

p = input_handling.readConfig()

for i in range(len(train_paths) + 1):
    in_path = (train_paths + [test_path])[i]
    out_path = (out_train_paths + [out_test_path])[i]

    with gzip.open(in_path, 'rt') as f:
        games = f.read().splitlines()

    in_vecs = []
    out_vecs = []
    for j, game in enumerate(games):
        temp = read_pgn.game_to_pairs_real(game, p, j)
        in_vecs.append(temp[0])
        out_vecs.append(temp[1])
        if j % 100 == 99:
            print(f'Done processing batch {i+1}, game {j+1}')

    data = read_pgn.games_to_tensors(in_vecs, out_vecs)

    with gzip.open(out_path, 'wb') as f:
        pickle.dump(data, f)
