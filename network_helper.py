import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import CSVLogger

import input_handling
import Game
import misc
import board_helper
import policy

#################################################################################
#   Utilities that are used by Model objects, specific to the chess engine
#################################################################################

def train(net, tData, vData, p):
    mom = p['mom']
    learn_rate = p['nu']

    #   Compile model using current hyperparameters
    optim = tf.keras.optimizers.SGD(learning_rate=learn_rate, momentum=mom)

    net.compile(
        optimizer = optim,
        loss = tf.keras.losses.BinaryCrossentropy(),
        metrics = []
    )
    
    #   Train the model
    csv_logger = CSVLogger('visualization/costs.csv', append=True)
    
    new_history = net.fit(
        tData[0],
        tData[1],
        batch_size = p['batchSize'],
        epochs = p['epochs'],
        validation_data = vData,
        callbacks=[csv_logger]
    )


def net_activity(net, tData):
    #   TODO
    return

def display_evaluation(game, bestMoves, r):
    if game.whiteToMove:
        print(str(game.moveNum) + '.')
        
        print('    White to move; playing ' + \
              bestMoves[0].getMoveName(game.board) + \
              '.')
    else:
        print('    Black to move; playing ' + \
              bestMoves[0].getMoveName(game.board) + \
              '.')

    for i in range(len(bestMoves)):
        print('        ' + str(i+1) + '.',
              bestMoves[i].getMoveName(game.board),
              '(' + str(r[i]) + ')')


def bestGame(net):
    #   Get parameters, but use a fully greedy policy
    p = input_handling.readConfig(3)
    p.update(input_handling.readConfig(1))
    p['epsGreedy'] = 0
    p['epsSearch'] = 0

    if p['mode'] >= 2:
        num_lines = input_handling.getUserInput("Display how many lines? ",
                                                "Not a valid number.",
                                                "int",
                                                "var >= 2 and var < 50")
    else:
        num_lines = 1

    game = Game.Game(quiet=False)

    while (game.gameResult == 17):
        if num_lines == 1:
            bestMove = policy.getBestMoveTreeEG(net, game, p)
            game.doMove(bestMove)
        else:
            bestMoves, r = policy.getBestMoveTreeEG(net, game, p, num_lines)
            display_evaluation(game, bestMoves, r)
            game.doMove(bestMoves[0])

    if p['mode'] < 2:
        print(game.annotation)
    
    game.toPGN()
