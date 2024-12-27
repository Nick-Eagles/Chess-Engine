import numpy as np
import tensorflow as tf

#   Given the compressed game representation used for storage in files,
#   write the corresponding .fen file representation
def toFEN(NN_vec, filename, verbose=True):
    assert NN_vec.shape == (1, 839), NN_vec.shape
    NN_vec = tf.reshape(NN_vec, [839])
    
    #   Reformat NN_vec so that accessing the piece at a particular square is
    #   intuitive
    inds = [i for i in range(832) if NN_vec[i]]
    assert len(inds) == 64, "NN_vec is invalid/ does not describe a position."
    
    outList = [i % 13 for i in inds] + NN_vec[832:].numpy().tolist()
    
    ###########################################################
    #   Pieces on board
    ###########################################################
    
    letters = 'PNBRQK'
    game_str = ''
    board = np.array(outList[:64]).reshape(8,8)
    
    # Loop through ranks backward
    for i in range(8):
        j = 0
        empty_count = 0
        while j < 8:
            piece = int(board[j][7-i]) - 6
            if piece >= 1:
                game_str += letters[piece - 1]
            elif piece == 0:
                empty_count += 1
                # If on the last of some group of contiguous empty squares
                if j == 7 or board[j+1][7-i] - 6 != 0:
                    game_str += str(empty_count)
                    empty_count = 0
            else:
                game_str += letters[-1*piece - 1].lower()

            j += 1

        if i < 7:
            game_str += '/'
    game_str += ' '

    ###########################################################
    #   Game info
    ###########################################################

    #   whiteToMove
    if outList[64]:
        game_str += 'w '
    else:
        game_str += 'b '

    #   castling
    if not any(outList[66:70]):
        game_str += '- '
    else:
        letters = 'KQkq'
        for i, val in enumerate(outList[66:70]):
            if val:
                game_str += letters[i]
        game_str += ' '

    #   En passant square
    if outList[65]:
        file = -1
        wToMove = outList[64]
        rank = 3 + wToMove
        for i in range(7):
            thisSq = board[i][rank]
            nextSq = board[i+1][rank]
            if thisSq * nextSq == -1:
                file = i + (wToMove == (thisSq == -1))
        if file != -1:
            game_str += 'abcdefgh'[file] + str(rank + 1 - 2 * wToMove) + ' '
        else:
            game_str += '- '
    else:
        game_str += '- '

    #   Moves since action/ halfmove counter
    game_str += str(int(outList[70] * 25)) + ' '

    #   Full move counter, fabricated since my move representations lose this
    #   info, and this info is unimportant for my purposes
    game_str += str(int(25 * outList[70] / 2) + 10)

    ###########################################################
    #   Write to .fen file
    ###########################################################

    if verbose:
        print("Writing position to " + filename + "...")
    with open(filename, 'w') as fenFile:
        fenFile.write(game_str)
    if verbose:
        print("Done.")
