import sys
import numpy as np
sys.path.append('../')

import Network
import Game
import board_helper
import input_handling
import network_helper

#   *Tested with print statements inserted in network_helper.getBestMove,
#   to examine raw eval vector, noise vector, and mix*
p = input_handling.readConfig(1, "../config.txt")
net = Network.load('../nets/7deep')
game = Game.Game(quiet=False)
for i in range(5):
    legalMoves = board_helper.getLegalMoves(game)
    game.doMove(network_helper.getBestMove(game, legalMoves, net, p))
