import numpy as np
from scipy.special import expit

import Move

#   returns true iff white is in check; to simplify its applications, the function also
#   considers adjacent kings to be a form of check
def inCheck(board):
    assert len(board) == 8 and len(board[0]) == 8, "inCheck() was passed an illegitimate board"
        
    #   find the rank and file of the white king
    for file in range(8):
        for rank in range(8):
            if board[file][rank] == 6:
                kingRank = rank
                kingFile = file
                
    #   CHECK BY PAWN
    if kingRank < 7:
        #   pawn or king to the "top" right? (if applicable)
        if (kingFile < 7 and board[kingFile+1][kingRank+1] == -1):
            #print("check by pawn to top right")
            return True

        #   pawn or king to the "top" left? (if applicable)
        if (kingFile > 0 and board[kingFile-1][kingRank+1] == -1):
            #print("check by pawn to top left")
            return True

    #   CHECK BY KING (allows for convenient computation of illegal moves):
    #   define a box around the king and iterate through to check if the
    #   black king is in it (including, for simplicity, the case where kings share a square)
    ranks = list(range(max(0,kingRank-1),min(8,kingRank+2)))
    files = list(range(max(0,kingFile-1),min(8,kingFile+2)))
    for r in ranks:
        for f in files:
            if board[f][r] == -6:
                #print("check by king (illegal move)")
                return True

    #   x and y are perturbations in file and rank, respectively, relative to the king's square.
    #   Their outer product (by set) represents all diagonal steps of 1 square distance
    for x in [-1, 1]:
        for y in [-1, 1]:
            i = 1
            while inBounds((kingFile, kingRank), (i*x, i*y)):
                piece = board[kingFile+i*x][kingRank+i*y]
                if piece == -3 or piece == -5:
                    return True
                #   if the square is occupied then it blocks any farther piece from giving check,
                #   so the remaining diagonal need not be iterated through
                elif piece != 0:
                    i = 7
                i += 1

    #   Similarly, we step in the cardinal directions to test for the presence of a rook or queen
    for x, y in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        i = 1
        while inBounds((kingFile, kingRank), (i*x, i*y)):
            piece = board[kingFile+i*x][kingRank+i*y]
            if piece == -4 or piece == -5:
                return True
            elif piece != 0:
                i = 7
            i += 1
      
    #   Check by knight
    horizontals = [-2, -2, -1, -1, 1, 1, 2, 2]
    verticals = [-1, 1, -2, 2, -2, 2, -1, 1]
    for x,y in zip(horizontals, verticals):
        #   As long as this square is in bounds, check if it's a knight
        if inBounds((kingFile, kingRank), (x,y)) and board[kingFile+x][kingRank+y] == -2:
            return True

    return False

#   Returns (bool,bool) based on whether the current player can castle kingside
#   and queenside, respectively. 
def canCastle(game):
    coeff = 2 * game.whiteToMove - 1
    if (game.whiteToMove and inCheck(game.board)) or (not game.whiteToMove and inCheck(game.invBoard)):
        return (False, False)

    if game.whiteToMove:
        castleVars = (game.canW_K_Castle, game.canW_Q_Castle)
        backRank = 0
    else:
        castleVars = (game.canB_K_Castle, game.canB_Q_Castle)
        backRank = 7
    #   Kingside check
    if castleVars[0] and game.board[5][backRank] == 0 and game.board[6][backRank] == 0:
        newBoard = [x.copy() for x in game.board]
        newBoard2 = [x.copy() for x in game.board]
        newBoard[4][backRank] = 0
        newBoard2[4][backRank] = 0
        newBoard[5][backRank] = 6 * coeff
        newBoard2[6][backRank] = 6 * coeff
        whiteCond = game.whiteToMove and not inCheck(newBoard) and not inCheck(newBoard2)
        blackCond = not game.whiteToMove and not inCheck(invert(newBoard)) and not inCheck(invert(newBoard2))
        if whiteCond or blackCond:
            kingside = True
        else:
            kingside = False
    else:
        kingside = False

    #   Queenside check
    if castleVars[1] and game.board[1][backRank] == 0 and game.board[2][backRank] == 0 and game.board[3][backRank] == 0:
        newBoard = [x.copy() for x in game.board]
        newBoard2 = [x.copy() for x in game.board]
        newBoard[4][backRank] = 0
        newBoard2[4][backRank] = 0
        newBoard[2][backRank] = 6 * coeff
        newBoard2[3][backRank] = 6 * coeff
        whiteCond = game.whiteToMove and not inCheck(newBoard) and not inCheck(newBoard2)
        blackCond = not game.whiteToMove and not inCheck(invert(newBoard)) and not inCheck(invert(newBoard2))
        if whiteCond or blackCond:
            queenside = True
        else:
            queenside = False
    else:
        queenside = False

    return (kingside, queenside)
            
            
#   Given a game, returns a list of legal Moves for whichever player is to move.
def getLegalMoves(game):
    moves = []
    coeff = 2 * game.whiteToMove - 1
    if game.whiteToMove:
        backRank = 0
    else:
        backRank = 7
        game.invBoard = invert(game.board)

    #   Castling
    castling = canCastle(game)
    if castling[0]:
        moves.append(Move.Move((4,backRank),(6,backRank),6*coeff))
    if castling[1]:
        moves.append(Move.Move((4,backRank),(2,backRank),6*coeff))

    #   En passant: looks for any white pawns next to the black pawn that just moved
    if game.enPassant:
        #   Look to the left for a white pawn
        if game.lastMove.endSq[0] > 0 and game.board[game.lastMove.endSq[0]-1][game.lastMove.endSq[1]] == coeff:
            newBoard = [x.copy() for x in game.board]
            #   Move above the black pawn
            newBoard[game.lastMove.endSq[0]][game.lastMove.endSq[1]+coeff] = coeff
            #   Capture the black pawn
            newBoard[game.lastMove.endSq[0]][game.lastMove.endSq[1]] = 0
            #   Remove the white pawn from its starting position
            newBoard[game.lastMove.endSq[0]-1][game.lastMove.endSq[1]] = 0
            if (game.whiteToMove and not inCheck(newBoard)) or (not game.whiteToMove and not inCheck(invert(newBoard))):
                moves.append(Move.Move((game.lastMove.endSq[0]-1, game.lastMove.endSq[1]),(game.lastMove.endSq[0],game.lastMove.endSq[1]+coeff),coeff))
        #   Look to the right for a white pawn
        if game.lastMove.endSq[0] < 7 and game.board[game.lastMove.endSq[0]+1][game.lastMove.endSq[1]] == coeff:
            newBoard = [x.copy() for x in game.board]
            #   Move above the black pawn
            newBoard[game.lastMove.endSq[0]][game.lastMove.endSq[1]+coeff] = coeff
            #   Capture the black pawn
            newBoard[game.lastMove.endSq[0]][game.lastMove.endSq[1]] = 0
            #   Remove the white pawn from its starting position
            newBoard[game.lastMove.endSq[0]+1][game.lastMove.endSq[1]] = 0
            if (game.whiteToMove and not inCheck(newBoard)) or (not game.whiteToMove and not inCheck(invert(newBoard))):
                moves.append(Move.Move((game.lastMove.endSq[0]+1, game.lastMove.endSq[1]),(game.lastMove.endSq[0],game.lastMove.endSq[1]+coeff),coeff))

    #   Anything else
    for file in range(8):
        for rank in range(8):
            piece = game.board[file][rank]
            assert abs(piece) <= 6, "Tried to get legal moves with a corrupt board"
            if piece*coeff <= 0:
                continue

            #   Pawn --------------------------------------------------------------
            if piece*coeff == 1:
                #   Offset is the change in file for the potential pawn move; cond is the corresponding condition
                #   to ensure legitimacy of an attempted change in file (will the move's end sqaure be in bounds)
                for offset in range(-1, 2):
                    if inBounds((file, rank), (offset, coeff)) \
                       and ((offset == 0 and game.board[file][rank+coeff]*coeff == 0) \
                       or (offset != 0 and game.board[file+offset][rank+coeff]*coeff < 0)):
                        assert game.board[file+offset][rank+coeff] != -6*coeff, "Had the option to capture king to promote" + game.verbosePrint()
                        #   Pawn promotion
                        if rank == 1 + 5*game.whiteToMove:
                            canPromote = False
                            #   Tries promotion to each possible piece 
                            for i in range(2,6):
                                #   Piece type doesn't affect whether promotion puts
                                #   white in check
                                move = Move.Move((file,rank),(file+offset,rank+coeff),i*coeff)
                                if canPromote or tryMove(game, move):
                                    canPromote = True
                                    moves.append(move)
                        else:
                            move = Move.Move((file,rank),(file+offset,rank+coeff),coeff)
                            if tryMove(game, move):
                                moves.append(move)  

                #   Also an empty square two spaces ahead for a pawn that
                #   hasn't moved
                if rank == 6 - 5*game.whiteToMove and game.board[file][rank+coeff] == 0 and game.board[file][rank+2*coeff] == 0:
                    move = Move.Move((file,rank),(file,rank+2*coeff),coeff)
                    if tryMove(game, move):
                        moves.append(move)                            

            #   Knight --------------------------------------------------------------
            elif piece*coeff == 2:
                horizontals = [-2, -2, -1, -1, 1, 1, 2, 2]
                verticals = [-1, 1, -2, 2, -2, 2, -1, 1]
                for x,y in zip(horizontals, verticals):
                    if inBounds((file, rank), (x, y)):
                        #   Empty square or black piece (includes king, shouldn't matter)
                        if game.board[file+x][rank+y]*coeff <= 0:
                            assert game.board[file+x][rank+y] != -6*coeff, "Had the option for knight to capture king" + game.verbosePrint()
                            move = Move.Move((file,rank),(file+x, rank+y), 2*coeff)
                            if tryMove(game, move):
                                moves.append(move)
            #   King
            elif piece*coeff == 6:
                ranks = list(range(max(0,rank-1),min(8,rank+2)))
                files = list(range(max(0,file-1),min(8,file+2)))
                for r in ranks:
                    for f in files:
                        if game.board[f][r]*coeff <= 0:
                            assert game.board[f][r] != -6*coeff, "Kings were adjacent and one had the option to capture the other" + game.verbosePrint()
                            move = Move.Move((file,rank),(f,r), 6*coeff)
                            if tryMove(game, move):
                                moves.append(move)

            #   Rook, bishop, and queen are more complicated to deal with individually
            #   ----------------------------------------------------------------------
            else:
                #   Diagonals (bishop or queen)
                if piece*coeff == 3 or piece*coeff == 5:
                    for x in [-1, 1]:
                        for y in [-1, 1]:
                            i = 1
                            while inBounds((file, rank), (i*x, i*y)):
                                pieceTemp = game.board[file+i*x][rank+i*y]
                                if pieceTemp*coeff <= 0:
                                    move = Move.Move((file,rank),(file+i*x,rank+i*y), piece)
                                    if tryMove(game, move):
                                        moves.append(move)
                                if pieceTemp == 0:
                                    i += 1
                                else:
                                    i = 8
                                
                #   Lateral, up/down movement (rook or queen)
                if piece*coeff == 4 or piece*coeff == 5:
                    for x, y in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        i = 1
                        while inBounds((file, rank), (i*x, i*y)):
                            pieceTemp = game.board[file+i*x][rank+i*y]
                            if pieceTemp*coeff <= 0:
                                move = Move.Move((file,rank),(file+i*x,rank+i*y), piece)
                                if tryMove(game, move):
                                    moves.append(move)
                            if pieceTemp == 0:
                                i += 1
                            else:
                                i = 8
             
    return moves              

#   Inverts board in the expected sense (by rank, file and color)
def invert(board):
    b = np.zeros((8,8))
    for f in range(8):
        for r in range(8):
            b[f][r] = -1*board[7-f][7-r]
    return b

#   A simple check that a square, when perturbed a distance in both
#   dimensions, remains in bounds on a chess board. Both are 2-tuples of ints.
def inBounds(square, perturb):
    h = (square[0] + perturb[0] >= 0) and (square[0] + perturb[0]) <= 7
    v = (square[1] + perturb[1] >= 0) and (square[1] + perturb[1]) <= 7
    return h and v

#   Return if a legitimate move is legal (ie. doesn't end with the player in
#   check). The passed move is relative to game.board, and should not be a
#   castling move or en passant.
def tryMove(game, move):
    if game.whiteToMove:
        newBoard = [x.copy() for x in game.board]
        startSq = move.startSq
        endSq = move.endSq
        endPiece = move.endPiece
    else:
        newBoard = [x.copy() for x in game.invBoard]
        startSq = (7 - move.startSq[0], 7 - move.startSq[1])
        endSq = (7 - move.endSq[0], 7 - move.endSq[1])
        endPiece = -1 * move.endPiece

    newBoard[startSq[0]][startSq[1]] = 0
    newBoard[endSq[0]][endSq[1]] = endPiece
    return not inCheck(newBoard)

#   Take a game board and initialized numpy array (netInput), and fill in the
#   numpy array with a representation of the board used for input to the NN.
def piece_to_vector(netInput, piece, c):
    for i in range(-6, 7):
        if piece == i:
            netInput[c][0] = 1
        else:
            netInput[c][0] = 0
        c += 1
        
    return netInput

#   invert: (bool) determines whether to invert board by color (white and
#           black switch)
#   flip0: (bool) reflect the board about the first axis (which is file, by
#           default, or rank if swap is true
#   flip1: (bool) reflect the board about the second axis
#   swap: (bool) switch the first and second axis (this corresponds to some
#           rotation of the board, and its exact effect depends on flip0 and flip1)
def generate_NN_vec(game, invert, flip0, flip1, swap):
    netInput = np.zeros((839,1))

    coeff = 1 - 2 * invert
    c = 0
    if not flip0 and not flip1 and not swap:
        for file in range(8):
            for rank in range(8):
                piece = coeff * game.board[file][rank]
                netInput = piece_to_vector(netInput, piece, c)
                c += 13
    elif not flip0 and not flip1 and swap:
        for file in range(8):
            for rank in range(8):
                piece = coeff * game.board[rank][file]
                netInput = piece_to_vector(netInput, piece, c)
                c += 13
    elif not flip0 and flip1 and not swap:
        for file in range(8):
            for rank in range(8):
                piece = coeff * game.board[file][7-rank]
                netInput = piece_to_vector(netInput, piece, c)
                c += 13
    elif not flip0 and flip1 and swap:
        for file in range(8):
            for rank in range(8):
                piece = coeff * game.board[rank][7-file]
                netInput = piece_to_vector(netInput, piece, c)
                c += 13
    elif flip0 and not flip1 and not swap:
        for file in range(8):
            for rank in range(8):
                piece = coeff * game.board[7-file][rank]
                netInput = piece_to_vector(netInput, piece, c)
                c += 13
    elif flip0 and not flip1 and swap:
        for file in range(8):
            for rank in range(8):
                piece = coeff * game.board[7-rank][file]
                netInput = piece_to_vector(netInput, piece, c)
                c += 13
    elif flip0 and flip1 and not swap:
        for file in range(8):
            for rank in range(8):
                piece = coeff * game.board[7-file][7-rank]
                netInput = piece_to_vector(netInput, piece, c)
                c += 13
    else: # all are true
        for file in range(8):
            for rank in range(8):
                piece = coeff * game.board[7-rank][7-file]
                netInput = piece_to_vector(netInput, piece, c)
                c += 13

    netInput[832][0] = int(invert + game.whiteToMove == 1)
    netInput[833][0] = int(game.enPassant)
    if not invert and not flip0:
        netInput[834][0] = int(game.canW_K_Castle)
        netInput[835][0] = int(game.canW_Q_Castle)
        netInput[836][0] = int(game.canB_K_Castle)
        netInput[837][0] = int(game.canB_Q_Castle)
    elif not invert and flip0:
        netInput[834][0] = int(game.canW_Q_Castle)
        netInput[835][0] = int(game.canW_K_Castle)
        netInput[836][0] = int(game.canB_Q_Castle)
        netInput[837][0] = int(game.canB_K_Castle)
    elif invert and not flip0:
        netInput[834][0] = int(game.canB_K_Castle)
        netInput[835][0] = int(game.canB_Q_Castle)
        netInput[836][0] = int(game.canW_K_Castle)
        netInput[837][0] = int(game.canW_Q_Castle)
    else: # both are true
        netInput[834][0] = int(game.canB_Q_Castle)
        netInput[835][0] = int(game.canB_K_Castle)
        netInput[836][0] = int(game.canW_Q_Castle)
        netInput[837][0] = int(game.canW_K_Castle)
    netInput[838][0] = game.movesSinceAction

    return netInput

def verify_data(data, withMates=True):
    if withMates:
        numBuffs = 4
    else:
        numBuffs = 3

    #   All buffers exist
    assert len(data) == numBuffs, len(data)
    for i in range(numBuffs):
        if len(data[i]) > 0:
            # the first example consists of an input and output
            assert len(data[i][0]) == 2, len(data[i][0])

            # the input is of proper shape
            assert data[i][0][0].shape == (839, 1), data[i][0][0].shape
        else:
            print("Warning: buffer", i, "was empty.")
