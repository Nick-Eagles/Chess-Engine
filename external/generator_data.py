import os
import sys
import random
import _pickle as pickle
sys.path.append('./')

import input_handling
import read_pgn
import policy_net

out_dir = 'external/2019_games_t'
in_file = 'external/2019_games_processed_t.txt'
num_groups = 1
blocks_per_group = 1
block_width = 2
total_lines = 2000
start_line = 10000
net_type = 'policy_value'

p = input_handling.readConfig()

buffer = [[]]
net = policy_net.InitializeNet(
    num_groups, blocks_per_group, block_width, p, net_type
)

with open(in_file, 'r') as pgn_file:
    all_lines = pgn_file.readlines()

if start_line == 0:
    first_index = 0
    os.mkdir(out_dir)
else:
    with open(out_dir + '/num_examples.txt', 'r') as f:
        first_index = int(f.readlines()[0])
        
data_index = first_index

#   Write each training example individually in order
print('Converting training examples...')
for i in range(start_line, start_line + total_lines):
    if i % 100 == 0:
        print('On line ' + str(i+1) + '...')
        
    buffer = [[]]

    read_pgn.process_line(buffer, all_lines[i], p, False, net_type)

    for x in buffer[0]:
        assert len(x) == 2, len(x)
        assert x[0].shape == (1, 839), x[0].shape
        with open(out_dir + '/' + str(data_index) + '.pkl', 'wb') as f:
            pickle.dump(x, f)

        data_index += 1

print('Converted ' + str(data_index + 1 - first_index) + ' examples.')

with open(out_dir + '/num_examples.txt', 'w') as f:
    f.write(str(data_index))
