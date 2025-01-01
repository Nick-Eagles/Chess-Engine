import numpy as np

import Game
import board_helper
import input_handling
import misc
import Traversal

def parseInput(moveNames):
    cond = 'var in auxVars'
    messDef = 'Enter your move: '
    messOnErr = 'Not a legal move; the following moves are legal:\n' + \
                ', '.join(moveNames)
    return input_handling.getUserInput(
        messDef,
        messOnErr,
        'str',
        cond,
        auxVars=moveNames
    )

def describeTraversal(trav, game, p):
    if p['mode'] > 1:
        #   Print evaluation before performing move
        expRew = (2 * game.whiteToMove - 1) * \
            trav.net(game.toNN_vecs(), training=False)[-1]
        print(
            "Expected reward from the current position is", 
            round(float(expRew), 3)
        )

        print(f'Chose {trav.bestLine[0]}. Top lines:')

        #   Indices of best lines and rewards based on evaluation
        indices = [
            misc.match(x, trav.rewards)
            for x in sorted(trav.rewards, reverse = game.whiteToMove)
        ]

        #   Print top lines and associated rewards in order of evaluation
        for i in indices:
            print(f'  ({round(trav.rewards[i], 2)}) {" ".join(trav.bestLines[i])}')
    else:
        print(f'Chose {trav.bestLine[0]}.')

def interact(net):
    p = input_handling.readConfig()

    game = Game.Game(quiet=False)

    #   Randomly select who plays white
    userStarts = bool(np.random.randint(2))
    if userStarts:
        print("You will play white.")
    else:
        print("The game starts now; you will play black.")

    while (game.gameResult == 17):
        if game.whiteToMove == userStarts:
            moves = board_helper.getLegalMoves(game)
            moveNames = [m.getMoveName(game) for m in moves]
            userChoice = parseInput(moveNames)
            
            game.doMove(moves[misc.match(userChoice, moveNames)])
        else:
            trav = Traversal.Traversal(game, net, p)
            trav.traverse()
            describeTraversal(trav, game, p)
            game.doMove(trav.bestMove)

    if game.gameResult == 2 * userStarts - 1:
        print("This is checkmate; you won.")
    elif game.gameResult == 1 - 2 * userStarts:
        print("You lost to a weak engine.")
    else:
        print("This is a draw. You can probably do better than that?")

def bestGame(net):
    p = input_handling.readConfig()

    game = Game.Game(quiet=False)

    while (game.gameResult == 17):
        trav = Traversal.Traversal(game, net, p)
        trav.traverse()
        describeTraversal(trav, game, p)
        game.doMove(trav.bestMove)

    game.toPGN()
            
