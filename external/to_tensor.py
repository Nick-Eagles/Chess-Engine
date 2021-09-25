#   This script is to be called from the repository's base directory:
#       python3 external/to_tensor.py

import gzip
import _pickle as pickle
import sys
sys.path.append('./')
sys.path.append('./experimental')

import input_handling
import policy_net
import buffer
import read_pgn

t_filename = 'external/2019_games_processed_t.txt'
v_filename = 'external/2019_games_processed_v.txt'
t_file_base_out = 'external/2019_games_tensor_t_'
v_file_base_out = 'external/2019_games_tensor_v_'
num_t_games = 1400
num_v_games = 201 # this is all of them
max_t_positions = 100000
max_v_positions = 15000

def save_tensor(tensor, filename):
    with gzip.open(filename, 'wb') as f:
        pickle.dump(tensor, f)

#   Initialize dummy parameters and a dummy network
print('Reading config and net...')
p = input_handling.readConfig()
net = policy_net.InitializeNet(p, 'policy_value', 1, 1, 2)

p['gamma'] = 0.9

#   Read validation games into a buffer
print('Reading validation games...')
v_games = read_pgn.load_games(
    v_filename, p, list(range(num_v_games)), net, certainty=False
)
v_games = [v_games[0][:max_v_positions]]
print(len(v_games[0]), 'positions loaded.')

#   Collapse into input and label tensors, then save
print('Collapsing and saving...')
x, y = buffer.collapse(v_games)
save_tensor(x, v_file_base_out + 'x.pkl.gz')
save_tensor(y, v_file_base_out + 'y.pkl.gz')

#   Read training games into a buffer
print('Reading training games...')
t_games = read_pgn.load_games(
    t_filename, p, list(range(num_t_games)), net, certainty=False
)
t_games = [t_games[0][:max_t_positions]]
print(len(t_games[0]), 'positions loaded.')

#   Collapse into input and label tensors, then save
print('Collapsing and saving...')
x, y = buffer.collapse(t_games)
save_tensor(x, t_file_base_out + 'x.pkl.gz')
save_tensor(y, t_file_base_out + 'y.pkl.gz')

print('Done.')
