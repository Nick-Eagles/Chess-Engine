import csv
import os
import numpy as np
from scipy.special import expit, logit

import input_handling

#   Take np array used as input to the NN and return a list more suitable for
#   writing to a csv: the sparse board representation is condensed into the
#   64 integer rep, and the booleans are appended "as is".
def compressNNinput(NN_vec):
    inds = [i for i in range(832) if NN_vec[i]]
    return [i % 13 for i in inds] + NN_vec[832:].flatten().tolist()

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

def writeCheckmates(games, filepath, compress=False):
    if os.path.exists(filepath):
        gameFile = open(filepath, 'a')
    else:
        gameFile = open(filepath, 'w')

    with gameFile:
        writer = csv.writer(gameFile)
        writer.writerows(games)
    
#   Reads game from a file as specified by the user; returns data as a list of tuples with
#   the first entry (the game) in compressed form
def readGames(filepath):
    flatData = []

    #   Open and read file
    with open(filepath, 'r') as gameFile:
        reader = csv.reader(gameFile, delimiter=',', quotechar=',', quoting=csv.QUOTE_MINIMAL)

        for line in reader:
            flatData.append(line)

    print("Found", len(flatData), "position(s) in this file.")

    #   Convert data to list of tuples as required
    games = []
    for row in range(len(flatData)):
        games.append([float(i) for i in flatData[row]])  # csv reader returns a line as list of strings

    return games

#   Return only the positions in newGames that are not present in the file specified by filepath
def filterByNovelty(newGames, filepath):
    fileGames = readGames(filepath)

    novelGames = []
    for nG in newGames:
        gNum = 0
        match = False
        while not match and gNum < len(fileGames):
            i = 0
            while i < len(nG) and nG[i] == fileGames[i]:
                i += 1
            match = i == len(nG)
            gNum += 1
        if gNum == len(fileGames):
            novelGames.append(nG)

    return novelGames

#   Given the compressed game representation used for storage in files,
#   write the corresponding .fen file representation
def toFEN(NN_vec, filename):
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

    print("Writing position to .fen file...")
    with open(filename, 'w') as fenFile:
        fenFile.write(game_str)
    print("Done.")

def normalize(xBatch, gamma, beta, eps):
    assert len(xBatch.shape) == 2
    assert len(gamma.shape) == 2 and gamma.shape[1] == 1, gamma.shape
    assert len(beta.shape) == 2 and beta.shape[1] == 1, beta.shape
    layLen = xBatch.shape[0]
    batchSize = xBatch.shape[1]
    
    mean = (np.sum(xBatch, axis = 1) / batchSize).reshape((-1,1))
    dev = xBatch - np.dot(mean, np.ones((1, batchSize)))
    var_w_eps = np.sum(dev * dev, axis = 1) / batchSize + np.full(layLen, eps)
    var_w_eps = np.sqrt(var_w_eps)

    assert len((gamma.flatten() * dev[:,0] / var_w_eps).shape) == 1, (gamma.flatten() * dev[:,0] / var_w_eps).shape
    xNorm = np.array([gamma.flatten() * dev[:,i] / var_w_eps for i in range(batchSize)]).T + np.full((layLen, batchSize), beta)
    assert xNorm.shape == (layLen, batchSize), xNorm.shape
    return xNorm

#   For a batch (2d np array w/ axis 0 being one example's activations), return a
#   tuple: (variance, sum of deviations, mean) as np arrays of expected dim
def batchStats(xBatch):
    mean = np.sum(xBatch, axis = 1) / xBatch.shape[1]
    dev = xBatch - np.dot(mean.reshape((-1,1)), np.ones((1, xBatch.shape[1])))
    return (np.sum(dev * dev, axis = 1) / xBatch.shape[1], dev * dev, mean)

#   Intended to provide numerical stability when computing costs. y is the actual
#   reward for a given example, which can only be outside of [0,1] due to floating
#   point imprecision. 'tol' sets the threshold for acceptable level of deviation
#   from this range. Returns a value restricted to the range (0,1), where cutoff
#   determines how far in from the endpoints the function changes from linear to
#   asymptotic.
def stabilize(y, cutoff, tol):
    assert y < 1 + tol, y
    assert y > -1*tol, y
    if y > 1 - cutoff:
        return float(1 - cutoff * np.exp((1 - cutoff - y)/ cutoff))
    if y < cutoff:
        return float(cutoff * np.exp((y - cutoff)/ cutoff))
    return y

def dead_neurons(net, tData):
    p = input_handling.readConfig(0)
    bigBatch = np.array([g[0].flatten() for g in tData]).T
    z, zNorm, a = net.ff_track(bigBatch)

    deadNeurons = []
    for lay in zNorm:
        temp = []
        for neuron in lay:
            temp.append(int(all([x < 0 for x in neuron]))) # across batch for 1 neuron
        deadNeurons.append(temp)

    print("Dead neurons by layer:")
    for i, lay in enumerate(deadNeurons):
        print("Layer" + str(i+1) + ": " + str(100 * sum(lay)/len(lay)) + "% dead")

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

def getBestMove(game, legalMoves, net, p):
    eps = p['epsilon']
    if eps == 1:
        #   Shortcut for completely random move choice
        return legalMoves[np.random.randint(len(legalMoves))]
    else:
        #   Get the expected future reward
        vals = np.zeros(len(legalMoves), dtype=np.float32)
        for i, m in enumerate(legalMoves):
            rTuple = game.getReward(m, p['mateReward'])
            vals[i] = rTuple[0] + float(logit(net.feedForward(rTuple[1])))

        #   Return the best move as the legal move maximizing the linear combination of:
        #       1. The expected future reward vector
        #       2. A noise vector matching the first 2 moments of the reward vector
        noise = np.random.normal(np.mean(vals), np.std(vals), vals.shape[0])
        if game.whiteToMove:
            bestMove = legalMoves[np.argmax((1 - eps) * vals + eps * noise)]
        else:
            bestMove = legalMoves[np.argmin((1 - eps) * vals + eps * noise)]
            
        return bestMove
