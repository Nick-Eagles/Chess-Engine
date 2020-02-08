import sys
sys.path.append('../')
from multiprocessing import Pool
import numpy as np

import policy
import Game
import Network
import network_helper
import input_handling
import board_helper

#   Random decision
iters = 15
sep = 0.15    # distance between alpha0 and alpha1
nu = 0.1     # size of first step in both alphas if mate occurs
persist = 0.7 # momentum-like term for exponentially-weighted average
decay = 0.6   # how quickly convergence is forced. Must be in (0,2)
resume = True

p1 = input_handling.readConfig(1, '../config.txt')
p1.update(input_handling.readConfig(3, '../config.txt'))
#p1['epsSearch'] = 0.0
p2 = p1.copy()

assert p1['epsGreedy'] > 0, "Games may be too deterministic to accurately converge."

#net = Network.Network([20, 1])
net = Network.load('../nets/8deep7', True)[0]
pool = Pool()

if resume:
    p1['alpha'] = 0.433
    p2['alpha'] = 0.283
    i_start = 180
    moving_result = -0.44
else:
    p1['alpha'] = 0.5 + sep / 2
    p2['alpha'] = 0.5 - sep / 2
    i_start = 0
    moving_result = 0

center_vals = []
for i in range(i_start, i_start + iters):
    print("On game", i+1, "of", i_start + iters, "...")
    center_vals.append(p1['alpha'] - sep / 2)
    print(center_vals[-1])
    
    game = Game.Game()
    while (game.gameResult == 17):
        #   This ensures there isn't a bias for white winning, for example
        if (game.whiteToMove + i) % 2 == 1:
            bestMove = policy.getBestMoveTreeEG(net, game, p1, pool)
        else:
            bestMove = policy.getBestMoveTreeEG(net, game, p2, pool)
        game.doMove(bestMove)
        
    #   Flip the game result if the supposedly better policy is being played
    #   by black each turn instead of white
    moving_result = persist * moving_result + (1 - persist) * game.gameResult * (1 - 2 * (i % 2))

    p1['alpha'] += (moving_result / (i + 1)**decay) * nu
    p2['alpha'] += (moving_result / (i + 1)**decay) * nu

pool.close()
print("p1['alpha']:", p1['alpha'])
print("p2['alpha']:", p2['alpha'])
print("moving_result:", moving_result)

if resume:
    mode = "a"
else:
    mode = "w"
    
result_file = open("../visualization/alpha_tests.txt", mode)
result_file.writelines([str(x) + "\n" for x in center_vals])
result_file.close()
