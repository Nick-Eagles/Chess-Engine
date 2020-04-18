import csv
import os
import numpy as np
from scipy.special import expit, logit

#   Take np array used as input to the NN and return a list more suitable for
#   writing to a csv: the sparse board representation is condensed into the
#   64 integer rep, and the booleans are appended "as is".
def compressNNinput(NN_vec):
    assert NN_vec.shape == (839, 1), NN_vec.shape
    
    inds = [i for i in range(832) if NN_vec[i][0]]
    assert len(inds) == 64, "NN_input to compress encodes a faulty board; num squares encoded: " + str(len(inds))
    outList = [i % 13 for i in inds] + NN_vec[832:].flatten().tolist()
    assert len(outList) == 71, len(outList)
    return outList

#   Given games in the format they are read from readGames(), produce a list
#   of tuples in the format required for training examples for the network
def decompressGames(games):
    cGames = []
    for g in games:
        NN_vec = []
        result = g.pop()
        board = np.array(g[:64]).reshape(8,8).tolist()
        for file in range(8):
            for rank in range(8):
                piece = board[file][rank]
                for i in range(13):
                    if piece == i:
                        NN_vec.append(1)
                    else:
                        NN_vec.append(0)
        NN_vec += g[64:]
        final_vec = np.array(NN_vec).reshape(-1,1)
        assert final_vec.shape[0] == 839, final_vec.shape[0]
        assert abs(result - 0.5) < 0.5, result 
        cGames.append((final_vec, result))

    return cGames

def writeGames(games, filepath, compress=False, append=True):
    if compress:
        if len(games) > 0:
            assert games[0][0].shape == (839, 1), "Is data already compressed?"
            assert float(games[0][1]) <= 1 and float(games[0][1]) >= 0, "Data label is not in 'expit' form"
        else:
            print("Warning: writing an empty file (no games to write for current call to file_IO.writeGames).")
        games = [compressNNinput(g[0]) + [float(g[1])] for g in games]
    else:
        assert type(games[0][0]) is int, "Is data actually compressed?"
    
    if os.path.exists(filepath) and append:
        gameFile = open(filepath, 'a')
    else:
        gameFile = open(filepath, 'w')

    with gameFile:
        writer = csv.writer(gameFile)
        writer.writerows(games)
    
#   Reads game from a file as specified by the user; returns data as a list of tuples with
#   the first entry (the game) in compressed form
def readGames(filepath, p):
    flatData = []

    #   Open and read file
    with open(filepath, 'r') as gameFile:
        reader = csv.reader(gameFile, delimiter=',', quotechar=',', quoting=csv.QUOTE_MINIMAL)

        for line in reader:
            flatData.append(line)

    if p['mode'] >= 2:
        print("Found", len(flatData), "position(s) in", os.path.basename(filepath) + ".")

    #   Convert data to list of tuples as required
    games = []
    for row in range(len(flatData)):
        games.append([np.array([[float(i)]]) for i in flatData[row]])  # csv reader returns a line as list of strings

    return games

#   Return only the positions in newGames that are not present in the file specified by filepath
def filterByNovelty(newGames, filepath, p):
    if not os.path.exists(filepath):
        return newGames

    fileGames = readGames(filepath, p)
    novelGames = []
    for nG in newGames:
        gNum = 0
        match = False
        while not match and gNum < len(fileGames):
            i = 0
            while i < len(nG)-2 and nG[i] == fileGames[i]:
                i += 1
            match = i == len(nG)-2
            gNum += 1
        if gNum == len(fileGames):
            novelGames.append(nG)

    return novelGames

#   Given the compressed game representation used for storage in files,
#   write the corresponding .fen file representation
def toFEN(NN_vec, filename, verbose=True):
    ###########################################################
    #   Pieces on board
    ###########################################################

    letters = 'PNBRQK'
    game_str = ''
    board = np.array(NN_vec[:64]).reshape(8,8)
    
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
    if NN_vec[64]:
        game_str += 'w '
    else:
        game_str += 'b '

    #   castling
    if not any(NN_vec[66:70]):
        game_str += '- '
    else:
        letters = 'KQkq'
        for i, val in enumerate(NN_vec[66:70]):
            if val:
                game_str += letters[i]
        game_str += ' '

    #   En passant square
    if NN_vec[65]:
        file = -1
        wToMove = NN_vec[64]
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
    game_str += str(int(NN_vec[70])) + ' '

    #   Full move counter, fabricated since my move representations lose this info,
    #   and this info is unimportant for my purposes
    game_str += str(int(NN_vec[70] / 2) + 10)

    ###########################################################
    #   Write to .fen file
    ###########################################################

    if verbose:
        print("Writing position to " + filename + "...")
    with open(filename, 'w') as fenFile:
        fenFile.write(game_str)
    if verbose:
        print("Done.")
