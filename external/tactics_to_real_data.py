import pandas as pd
from pyhere import here
from pathlib import Path
import sys
import pickle
import gzip

sys.path.append(str(here()))
sys.path.append(str(here('external')))
import input_handling
import read_tactics
import read_pgn

batch_size = 40000
num_batches = 30

tactics_path = here('external', 'preprocessed_tactics', 'all_tactics.csv.gz')
train_paths = [
    here('external', 'preprocessed_tactics', 'g75_conv', f'train{i+1}.pkl.gz')
    for i in range(num_batches-1)
]
test_path = Path(here('external', 'preprocessed_tactics', 'g75_conv', 'test.pkl.gz'))

test_path.parent.mkdir(exist_ok = True)

p = input_handling.readConfig()
tactics = pd.read_csv(tactics_path)

for batch_num in range(num_batches):
    print(f'Starting batch {batch_num} of {num_batches}...')
    in_vecs = []
    out_vecs = []
    for tactic_num in range(batch_num * batch_size, (batch_num + 1) * batch_size):
        #   Process this tactical position and append tensors to the ongoing
        #   lists
        temp = read_tactics.tactics_to_pairs(
            fen_str = tactics.iloc[tactic_num]['FEN'],
            moves = tactics.iloc[tactic_num]['Moves'].split(' '),
            p = p
        )
        in_vecs.append(temp[0])
        out_vecs.append(temp[1])
    
    data = read_pgn.games_to_tensors(in_vecs, out_vecs)
    assert data[0].shape[0] == batch_size, data[0].shape[0]
    assert data[1][0].shape[0] == batch_size, data[1][0].shape[0]

    out_path = ([test_path] + train_paths)[batch_num]
    with gzip.open(out_path, 'wb') as f:
        pickle.dump(data, f)
