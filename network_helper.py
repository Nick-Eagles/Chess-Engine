import input_handling
import Game

def display_evaluation(game, bestMoves, r, bestLine):
    if game.whiteToMove:
        print(f'{str(game.moveNum)}. White to move; playing {bestMoves[0].getMoveName(game)}.')
    else:
        print(f'    Black to move; playing {bestMoves[0].getMoveName(game)}.')

    if not(bestLine is None):
        print('    Considered ' + ' '.join(bestLine))

    for i in range(len(bestMoves)):
        print(f'        {str(i+1)}. {bestMoves[i].getMoveName(game)} ({str(round(r[i], 3))})')

def bestGame(net, policy_function):
    p = input_handling.readConfig()

    if p['mode'] >= 2:
        num_lines = input_handling.getUserInput(
            "Display how many lines? ",
            "Not a valid number.",
            "int",
            "var >= 2 and var < 50"
        )
    else:
        num_lines = 1

    game = Game.Game(quiet=False)

    while (game.gameResult == 17):
        if num_lines == 1:
            bestMove = policy_function(net, game, p)
            game.doMove(bestMove)
        else:
            bestMoves, r, bestLine = policy_function(
                net, game, p, True, num_lines
            )
            display_evaluation(game, bestMoves, r, bestLine)
            game.doMove(bestMoves[0])

    if p['mode'] < 2:
        print(game.annotation)
    
    game.toPGN()
