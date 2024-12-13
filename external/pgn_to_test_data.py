#   The goal here is to read in 6980_games.txt and produce tensors for training
#   a neural network. The output tensors are defined in an intentionally simple
#   and artificial way: if the current color has a queen, the policy should be
#   moving it to A1, and the evaluation should be that the current color wins.
#   Otherwise, move the current king to B2 and the opposite color wins. The idea
#   is that training a network to find the optimal chess move is highly
#   difficult, and an easier case should be attempted to ensure the foundation
#   is programmed correctly (the tensorflow piece and game.to_NNvecs())

from pyhere import here
import sys
import pickle
import gzip

sys.path.append(str(here('external')))

import read_pgn

pgn_path = here('external', '6956_games.txt')
test_size = 0.1
random_state = 0
out_path = here('external', 'tensor_list_synthetic.pkl.gz')

with open(pgn_path, 'r') as f:
    games = f.read().splitlines()

in_vecs = []
out_vecs = []
for i, game in enumerate(games[:500]):
    temp = read_pgn.game_to_pairs_synthetic(game, i)
    in_vecs.append(temp[0])
    out_vecs.append(temp[1])
    if i % 100 == 0:
        print(f'Done processing game {i}')

data = read_pgn.games_to_tensors(
    in_vecs, out_vecs, test_size, random_state
)

with gzip.open(out_path, 'wb') as f:
    pickle.dump(data, f)
