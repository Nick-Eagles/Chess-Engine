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

#   A helper function for Network.showGame. Returns the string that is output for
#   a game at its current state (the list of evaluations for whichever player is
#   to move). These are printed when running the program in mode >= 2.
def generateAnnLine(evalList, game):
    if game.whiteToMove:
        line = "#####  Move " + str(game.moveNum) + ":  #####\n\n-- White: --\n"
    else:
        line = "-- Black: --\n"
        
    for i, e in enumerate(evalList):
        line += str(i) + ". " + e[0] + ", overall: " + str(round(e[1] + e[2], 4))
        line += " | r = " + str(round(e[2], 4)) + " | NN eval = " + str(round(e[1], 4)) + "\n"
    line += "\n"

    return line

def bestGame(net):
    p = input_handling.readConfig(3)
    p.update(input_handling.readConfig(1))
    p['epsGreedy'] = 0
    p['epsSearch'] = 0

    game = Game.Game(quiet=False)

    while (game.gameResult == 17):
        bestMove = policy.getBestMoveTreeEG(net, game, p)
        game.doMove(bestMove)

    print(game.annotation)
    game.toPGN()
