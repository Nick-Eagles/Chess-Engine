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

#   This script verifies as a sanity check that an agent using getBestMoveTreeEG
#   performs better than one using getBestMoveEG, which should perform better
#   than one playing randomly.

repeats = 10

pool = Pool()
net = Network.load('../nets/8deep5', True)[0]
p = input_handling.readConfig(1, '../config.txt')
p.update(input_handling.readConfig(3, '../config.txt'))

print("Using", repeats, "games for each policy comparison.")

###############################################################
#   getBestMoveTreeEG vs. getBestMoveEG
###############################################################

results = 0     # how many times "better" policy wins
for i in range(repeats):
    game = Game.Game()
    while (game.gameResult == 17):
        #   This ensures there isn't a bias for white winning, for example
        if (game.whiteToMove + i) % 2 == 1:
            bestMove = policy.getBestMoveTreeEG(net, game, p, pool)
        else:
            bestMove = policy.getBestMoveEG(net, game, p)
        game.doMove(bestMove)
        
    #   Flip the game result if the supposedly better policy is being played
    #   by black each turn instead of white
    results += game.gameResult * (1 - 2 * (i % 2))

print("Average game result for tree search vs. simple EG policy:", round(results/ repeats, 4))

###############################################################
#   getBestMoveEG vs. random
###############################################################

results = 0     # how many times "better" policy wins
for i in range(repeats):
    game = Game.Game()
    while (game.gameResult == 17):
        #   This ensures there isn't a bias for white winning, for example
        if (game.whiteToMove + i) % 2 == 1:
            bestMove = policy.getBestMoveEG(net, game, p)
        else:
            legalMoves = board_helper.getLegalMoves(game)
            bestMove = legalMoves[np.random.randint(len(legalMoves))]
        game.doMove(bestMove)
        
    #   Flip the game result if the supposedly better policy is being played
    #   by black each turn instead of white
    results += game.gameResult * (1 - 2 * (i % 2))

print("Average game result for simple EG policy vs. random:", round(results/ repeats, 4))
