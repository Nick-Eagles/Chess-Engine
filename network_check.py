#   Ran interactively to test the network's competence: namely, performance
#   on both tactical and game-based test data, and looking at policy suggestions
#   in the opening 

from pyhere import here
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
import pickle
import gzip

import Game
import board_helper
import policy_net
import Move

net_name = "first"
model_path = here('nets', net_name, 'model.keras')
test_game_path = here(
    'external', 'preprocessed_games', 'g75_new',
    'tensor_list_test_real.pkl.gz'
)
test_tactics_path = here(
    'external', 'preprocessed_tactics', 'g75', 'test.pkl.gz'
)

#   Given input and output tensors and a model 'net', return the cosine
#   similarity of the model predictions of reward against the labels.
#   Each sample composes one component of the vectors (e.g. if X is a batch
#   of 100 samples, cosine similarity is taken across two 100-dimensional
#   vectors)
def cos_sim(X, y, net):
    exp_rew = net(X, training = False)[-1].numpy().flatten()
    act_rew = y[-1].numpy().flatten()
    result = float(
        (exp_rew @ act_rew) /
        (np.linalg.norm(exp_rew) * np.linalg.norm(act_rew))
    )
    
    return result

def tensor_cor(X, y, net):
    exp_rew = net(X, training = False)[-1].numpy().flatten()
    act_rew = y[-1].numpy().flatten()
    
    return np.corrcoef(exp_rew, act_rew)[0, 1]

def to_eval_df(game, net):
    outputs = net(game.toNN_vecs(), training=False)[:2]
    moves = board_helper.getLegalMoves(game)
    evals = policy_net.AdjustPolicy(outputs, moves, game)
    
    df = pd.DataFrame(
        {
            'move': [x.getMoveName(game) for x in moves],
            'evaluation': evals
        }
    ).sort_values('evaluation', ascending = False)
    
    return df

net = load_model(model_path)

with gzip.open(test_game_path, 'rb') as f:
    X_test_game, y_test_game = pickle.load(f)

with gzip.open(test_tactics_path, 'rb') as f:
    X_test_tact, y_test_tact = pickle.load(f)

sim_game = cos_sim(X_test_game, y_test_game, net)
sim_tact = cos_sim(X_test_tact, y_test_tact, net)
cor_game = tensor_cor(X_test_game, y_test_game, net)
cor_tact = tensor_cor(X_test_tact, y_test_tact, net)

df = pd.DataFrame(
    {
        'dataset': ['tactics', 'games'],
        'cos_sim': [sim_tact, sim_game],
        'cor': [cor_tact, cor_game]
    }
)
print('Performance on test data:')
print(df)

#   Check that reasonable openings are suggested by the network
game = Game.Game()
print('From the starting position:')
print(to_eval_df(game, net))

game.doMove(Move.Move((4, 1), (4, 3), 1))
print('After 1. e4:')
print(to_eval_df(game, net))
