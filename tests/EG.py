import sys
import numpy as np
sys.path.append('../')

import Network
import Game
import board_helper
import input_handling
import network_helper
import policy

gameLen = 83

p = input_handling.readConfig(3, "../config.txt")
net = Network.load('../nets/8deep')

game = Game.Game()
game2 = Game.Game()
step = 0
equalCount = 0
while (game.gameResult == 17 and step < gameLen):
    #   Get best move (epsilon-greedy policy)
    legalMoves = board_helper.getLegalMoves(game)
    bestMove = policy.getBestMoveEG(game, legalMoves, net, p['epsGreedy'], p['mateReward'])
    bestMove2 = policy.getBestMoveEG(game, legalMoves, net, p['epsGreedy'], p['mateReward'])

    game.doMove(bestMove)
    game2.doMove(bestMove2)

    #   Moves are equal iff their resulting boards from the same position are equal
    equalCount += np.array_equal(game.board, game2.board)
    
    game2 = game.copy()
    step += 1

expConc = (1 - p['epsGreedy'])**2
print("Based on epsGreedy, 2 moves chosen should be the same", expConc, "of the time.")
print("An actual experiment showed the same moves were selected", round(equalCount / gameLen, 5), "of the time.")
