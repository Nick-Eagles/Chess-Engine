import numpy as np
from scipy.special import expit, logit
import time

import Game
import board_helper
import Network
import input_handling
import Traversal
import Move
import misc
import policy

def parseInput(game, moveNames):
    cond = 'var in auxVars'
    messDef = 'Enter your move: '
    messOnErr = 'Not a legal move; the following moves are legal:\n' + \
                ', '.join(moveNames)
    return input_handling.getUserInput(messDef,
                                       messOnErr,
                                       'str',
                                       cond,
                                       auxVars=moveNames)

def interact(net):
    p = input_handling.readConfig()
    p['epsGreedy'] = 0
    p['epsSearch'] = 0

    game = Game.Game(quiet=False)

    #   Randomly select who plays white
    userStarts = bool(np.random.randint(2))
    if userStarts:
        print("You will play white.")
    else:
        print("The game starts now; you will play black.")

    actRew, expRew = [], []
    
    while (game.gameResult == 17):
        if game.whiteToMove == userStarts:
            moves = board_helper.getLegalMoves(game)
            moveNames = [m.getMoveName(game.board) for m in moves]
            userChoice = parseInput(game, moveNames)
            bestMove = moves[misc.match(userChoice, moveNames)]
            
            if p['mode'] >= 2:
                #   Do move and observe reward
                r, vec = game.getReward(bestMove,
                                        p['mateReward'],
                                        simple=False,
                                        copy=False)
                actRew.append(r)

                #   Predict the reward of the next move
                if game.gameResult == 17:
                    expRew.append(logit(net(vec, training=False)))
                    print("Expected reward from the current position is", 
                          round(float(expRew[-1]), 4))
            else:
                game.doMove(bestMove)
        else:
            print('Calculating...', end='')
            bestMove = policy.getBestMoveTreeEG(net, game, p)
            print('Chose ', bestMove.getMoveName(game.board), '.', sep='')

            if p['mode'] >= 2:
                #   Do move and observe reward
                r, vec = game.getReward(bestMove,
                                        p['mateReward'],
                                        simple=False,
                                        copy=False)
                actRew.append(r)

                #   Predict the reward of the next move
                if game.gameResult == 17:
                    expRew.append(logit(net(vec, training=False)))
                    print("Expected reward from the current position is", 
                          round(float(expRew[-1]), 4))
            else:
                game.doMove(bestMove)


    #   Compute certainty
    if p['mode'] >= 2:
        assert len(actRew) == len(expRew), len(actRew)-len(expRew)
        actRew = np.array(actRew)
        expRew = np.array(expRew)

        #   Pass back rewards
        for i in range(1,actRew.shape[0]):
            actRew[-i-1] += p['gamma'] * actRew[-i]
            expRew[-i-1] += p['gamma'] * expRew[-i]
        
        actNorm = np.linalg.norm(actRew)
        if round(float(actNorm), 5) == 0:
            certainty = 0
        else:
            certainty = np.dot(expRew, actRew) / \
                           (np.linalg.norm(expRew) * np.linalg.norm(actRew))

        print("Certainty was", certainty)


    if game.gameResult == 2 * userStarts - 1:
        print("This is checkmate; you won.")
    elif game.gameResult == 1 - 2 * userStarts:
        print("You lost to a weak engine.")
    else:
        print("This is a draw. You can probably do better than that?")
            
