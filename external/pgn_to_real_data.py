#   Try training a policy-value neural network on real data: predicting the move
#   chosen by high-rated players, and predict the expected reward from each
#   position 

from pyhere import here
import sys
import pickle
import gzip

sys.path.append(str(here('external')))

import input_handling
import read_pgn

pgn_path = here('external', '6956_games.txt')
out_path = here('external', 'tensor_list_real.pkl.gz')
test_size = 0.1
random_state = 0

p = input_handling.readConfig()

with open(pgn_path, 'r') as f:
    games = f.read().splitlines()

in_vecs = []
out_vecs = []
for i, game in enumerate(games):
    temp = read_pgn.game_to_pairs_real(game, p, i)
    in_vecs.append(temp[0])
    out_vecs.append(temp[1])
    if i % 100 == 0:
        print(f'Done processing game {i}')

data = read_pgn.games_to_tensors(
    in_vecs, out_vecs, test_size, random_state
)

with gzip.open(out_path, 'wb') as f:
    pickle.dump(data, f)
