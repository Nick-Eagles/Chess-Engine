import csv
import os
import gzip
import _pickle as pickle
import numpy as np
from scipy.special import expit, logit
import tensorflow as tf

import input_handling
import buffer

#   Write a buffer (the standard in-memory representation of data for this
#   chess engine) to a compressed file
#
#   buff:   A list whose elements are lists of 2-tuples: (input, label) pairs
#
#   filepath: a string giving the filename to write the data in buff to. This
#           should end in '.pkl.gz', since it will be a gzipped pickled
#           representation of the data
def writeBuffer(buff, filepath, append=False):
    #   Appending works by loading, combining, then writing
    if append:
        buff = buffer.combine(readBuffer(filepath, p), buff)
        
    #   First restructure data to use numpy arrays instead of tf.Tensors. The
    #   idea is that numpy arrays seem to be more 'lightweight', and it should
    #   be far simpler to serialize 8 numpy arrays than thousands of tf.Tensors
    simple_buff = []
    for b in buff:
        x = tf.stack([tf.reshape(e[0], [839]) for e in b]).numpy()
        y = tf.stack([tf.reshape(e[1], [1]) for e in b]).numpy()
        simple_buff.append((x, y))

    #   Write the simpler representation to a compressed file
    with gzip.open(filepath, 'wb') as gz_file:
        pickle.dump(simple_buff, gz_file)

#   Read a restructured "buffer" of data (in the format produced by
#   writeBuffer), returning a list whose elements are lists of 2-tuples:
#   (input, label) pairs. Typically this list has 4 elements, when reading in
#   'buffers' like session.tBuffer and session.vBuffer, or 1 element to read in
#   a generic list of data (such as the list of pre-checkmate positions).
def readBuffer(filepath, p):
    #   Load the simplified representation of the data
    with gzip.open(filepath, 'rb') as gz_file:
        simple_buff = pickle.load(gz_file)

    #   Reformat into the standard in-memory representation
    buff = [[] for i in range(len(simple_buff))]
    for i, b in enumerate(simple_buff):
        for j in range(b[0].shape[0]):
            buff[i].append(
                (tf.constant(b[0][j,:], shape=[1,839], dtype=tf.float32),
                tf.constant(b[1][j,:], shape=[1,1], dtype=tf.float32))
            )

    buffer.verify(buff, p, len(simple_buff))
    
    return buff


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
