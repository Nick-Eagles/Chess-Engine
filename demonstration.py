import numpy as np

import Game
import board_helper
import input_handling
import misc

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
            print('Calculating...', end='')
            trav = Traversal.Traversal(game, net, p)
            trav.traverse()
            bestMove = trav.bestMove
            print('Chose ', bestMove.getMoveName(game), '.', sep='')
            game.doMove(bestMove)

            if p['mode'] >= 1:
                expRew = net(game.toNN_vecs(every=False)[0], training=False)[-1]
                
                print(
                    "Expected reward from the current position is", 
                    round(float(expRew), 4)
                )


    if game.gameResult == 2 * userStarts - 1:
        print("This is checkmate; you won.")
    elif game.gameResult == 1 - 2 * userStarts:
        print("You lost to a weak engine.")
    else:
        print("This is a draw. You can probably do better than that?")
            
