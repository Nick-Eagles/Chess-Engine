import numpy as np
from scipy.special import expit
import tensorflow as tf

import Move

#   A helper function for tryMove. Given the original board, a move that is not
#   capturing en passant or castling, and integers together representing the
#   direction along which to look for checking pieces, returns True iff in
#   check after performing the move. For example, df = 1 and dr = -1 would
#   indicate that we should look for checking pieces downward and to the right
#   of the moving piece's start square
def check_along_line(board, move, df, dr, kingFile, kingRank):
    #   Determine whether to look for (rooks and queens) or (bishops and queens)
    pieces = [-5]
    if df == 0 or abs(dr / df) != 1:
        pieces.append(-4)
    else:
        pieces.append(-3)

    newBoard = [x.copy() for x in board]
    newBoard[move.startSq[0]][move.startSq[1]] = 0
    newBoard[move.endSq[0]][move.endSq[1]] = move.endPiece

    #   Look for pieces beyond the moving piece's starting square
    i = 1
    while inBounds((kingFile, kingRank), (i*df, i*dr)):
        piece = newBoard[kingFile + i*df][kingRank + i*dr]
        if piece in pieces:
            return True
        elif piece != 0:
            i = 7
        i += 1

    return False

#   A helper function for tryMove, calling check_along_line for the appropriate
#   direction given the move's starting square relative to the king's square
def nonVerticalCheck(board, move, slope_start, delta_f_start, kingFile, kingRank):
    if slope_start == 1:
        if delta_f_start > 0:
            return check_along_line(board, move, 1, 1, kingFile, kingRank)
        else:
            return check_along_line(board, move, -1, -1, kingFile, kingRank)
    elif slope_start == -1:
        if delta_f_start > 0:
            return check_along_line(board, move, 1, -1, kingFile, kingRank)
        else:
            return check_along_line(board, move, -1, 1, kingFile, kingRank)
    elif slope_start == 0:
        if delta_f_start > 0:
            return check_along_line(board, move, 1, 0, kingFile, kingRank)
        else:
            return check_along_line(board, move, -1, 0, kingFile, kingRank)
    else:
        #   The moving piece is not "lined up" with the king, and therefore
        #   can't "discover" a check upon moving
        return False


#   returns true iff white is in check; to simplify its applications, the function also
#   considers adjacent kings to be a form of check
def inCheck(board):
    assert len(board) == 8 and len(board[0]) == 8, "inCheck() was passed an illegitimate board"
    
    #   find the rank and file of the white king
    kingFile = -1
    found = False
    
    while kingFile < 7 and not found:
        kingFile += 1
        kingRank = -1
        while kingRank < 7 and not found:
            kingRank += 1
            if board[kingFile][kingRank] == 6:
                found = True
                
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

    #   CHECK BY ADJACENT QUEEN OR KING (allows for convenient computation of
    #   illegal moves): define a box around the king and iterate through to
    #   check if the black king is in it (including, for simplicity, the case
    #   where kings share a square)
    ranks = range(max(0,kingRank-1),min(8,kingRank+2))
    files = range(max(0,kingFile-1),min(8,kingFile+2))
    for r in ranks:
        for f in files:
            if board[f][r] <= -5:
                #print("check by king (illegal move) or queen")
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
    if game.currentlyCheck:
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
        elif game.lastMove.endSq[0] < 7 and game.board[game.lastMove.endSq[0]+1][game.lastMove.endSq[1]] == coeff:
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
                ranks = range(max(0, rank-1), min(8, rank+2))
                files = range(max(0, file-1), min(8, file+2))
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
                if game.currentlyCheck:
                    #   Diagonals (bishop or queen)
                    if piece*coeff == 3 or piece*coeff == 5:
                        for diagonal in [-1, 1]:
                            num_legal = 0
                            num_illegal = 0
                            for direction in [-1, 1]:
                                i = 1
                                #   If the second condition is False, we have found the
                                #   unique legal move along this diagonal already
                                while inBounds((file, rank), (direction*diagonal*i, direction*i)) and (num_legal < 1 or num_illegal < 1):
                                    x = file + direction*diagonal*i
                                    y = rank + direction*i
                                    
                                    pieceTemp = game.board[x][y]
                                    if pieceTemp*coeff <= 0:
                                        move = Move.Move((file, rank),(x, y), piece)
                                        #   If 2 moves are legal, all are
                                        if num_legal >= 2 or tryMove(game, move):
                                            num_legal += 1
                                            moves.append(move)
                                        else:
                                            num_illegal += 1
                                    if pieceTemp == 0:
                                        i += 1
                                    else:
                                        i = 8

                    #   Lateral, up/down movement (rook or queen)
                    if piece*coeff == 4 or piece*coeff == 5:
                        for line in [0, 1]:
                            num_legal = 0
                            num_illegal = 0
                            for direction in [-1, 1]:
                                i = 1
                                #   If the second condition is False, we have found the
                                #   unique legal move along this diagonal already
                                while inBounds((file, rank), (line*direction*i, (1-line)*direction*i)) and (num_legal < 1 or num_illegal < 1):
                                    x = file + line*direction*i
                                    y = rank + (1-line)*direction*i
                                    
                                    pieceTemp = game.board[x][y]
                                    if pieceTemp*coeff <= 0:
                                        move = Move.Move((file, rank),(x, y), piece)
                                        #   If 2 moves are legal, all are
                                        if num_legal >= 2 or tryMove(game, move):
                                            num_legal += 1
                                            moves.append(move)
                                        else:
                                            num_illegal += 1
                                    if pieceTemp == 0:
                                        i += 1
                                    else:
                                        i = 8
                                        
                #   If not currently in check
                else:
                    #   Diagonals (bishop or queen)
                    if piece*coeff == 3 or piece*coeff == 5:
                        for diagonal in [-1, 1]:
                            num_legal = 0
                            any_illegal = False
                            for direction in [-1, 1]:
                                i = 1
                                while not any_illegal and inBounds((file, rank), (direction*diagonal*i, direction*i)):
                                    x = file + direction*diagonal*i
                                    y = rank + direction*i
                                    
                                    pieceTemp = game.board[x][y]
                                    if pieceTemp*coeff <= 0:
                                        move = Move.Move((file, rank),(x, y), piece)
                                        #   If 1 move is legal, all are
                                        if num_legal >= 1 or tryMove(game, move):
                                            num_legal += 1
                                            moves.append(move)
                                        #   If any are illegal, all are
                                        else:
                                            any_illegal = True
                                    if pieceTemp == 0:
                                        i += 1
                                    else:
                                        i = 8

                    #   Lateral, up/down movement (rook or queen)
                    if piece*coeff == 4 or piece*coeff == 5:
                        for line in [0, 1]:
                            num_legal = 0
                            any_illegal = False
                            for direction in [-1, 1]:
                                i = 1
                                while not any_illegal and inBounds((file, rank), (line*direction*i, (1-line)*direction*i)):
                                    x = file + line*direction*i
                                    y = rank + (1-line)*direction*i
                                    
                                    pieceTemp = game.board[x][y]
                                    if pieceTemp*coeff <= 0:
                                        move = Move.Move((file, rank),(x, y), piece)
                                        #   If 1 move is legal, all are
                                        if num_legal >= 1 or tryMove(game, move):
                                            num_legal += 1
                                            moves.append(move)
                                        #   If any are illegal, all are
                                        else:
                                            any_illegal = True
                                    if pieceTemp == 0:
                                        i += 1
                                    else:
                                        i = 8                
    
    
    return moves              


#   A variant of getLegalMoves which returns True iff at least one legal move
#   exists for the current player
def anyLegalMoves(game):
    coeff = 2 * game.whiteToMove - 1
    
    #   Check normal moves (not en passant captures or castling)
    for file in range(8):
        for rank in range(8):
            piece = game.board[file][rank]
            assert abs(piece) <= 6, "Tried to get legal moves with a corrupt board"
            if piece*coeff <= 0:
                continue

            #   Pawn -----------------------------------------------------------
            if piece*coeff == 1:
                #   Offset is the change in file for the potential pawn move
                for offset in range(-1, 2):
                    if inBounds((file, rank), (offset, coeff)) \
                       and ((offset == 0 and game.board[file][rank+coeff]*coeff == 0) \
                       or (offset != 0 and game.board[file+offset][rank+coeff]*coeff < 0)):
                        assert game.board[file+offset][rank+coeff] != -6*coeff, \
                            "Had the option to capture king to promote" + game.verbosePrint()
                        #   Pawn moves possibly involving captures or promotion,
                        #   but only moving up one rank (also note that the
                        #   promotion piece is a pawn, which suffices for the
                        #   purposes of this function)
                        move = Move.Move((file,rank),(file+offset,rank+coeff),coeff)
                        if tryMove(game, move):
                            return True 

                #   Also an empty square two spaces ahead for a pawn that
                #   hasn't moved
                if rank == 6 - 5*game.whiteToMove and game.board[file][rank+coeff] == 0 and game.board[file][rank+2*coeff] == 0:
                    move = Move.Move((file,rank),(file,rank+2*coeff),coeff)
                    if tryMove(game, move):
                        return True

            #   Knight --------------------------------------------------------------
            elif piece*coeff == 2:
                horizontals = [-2, -2, -1, -1, 1, 1, 2, 2]
                verticals = [-1, 1, -2, 2, -2, 2, -1, 1]
                for x, y in zip(horizontals, verticals):
                    if inBounds((file, rank), (x, y)):
                        #   Empty square or black piece (includes king, shouldn't matter)
                        if game.board[file+x][rank+y]*coeff <= 0:
                            assert game.board[file+x][rank+y] != -6*coeff, \
                                "Had the option for knight to capture king" + game.verbosePrint()
                            move = Move.Move((file,rank),(file+x, rank+y), 2*coeff)
                            if tryMove(game, move):
                                return True

            #   King
            elif piece*coeff == 6:
                ranks = range(max(0, rank-1), min(8, rank+2))
                files = range(max(0, file-1), min(8, file+2))
                for r in ranks:
                    for f in files:
                        if game.board[f][r]*coeff <= 0:
                            assert game.board[f][r] != -6*coeff, \
                                "Kings were adjacent and one had the option to capture the other" + game.verbosePrint()
                            move = Move.Move((file,rank),(f,r), 6*coeff)
                            if tryMove(game, move):
                                return True

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
                                        return True
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
                                    return True
                            if pieceTemp == 0:
                                i += 1
                            else:
                                i = 8

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
                return True
        #   Look to the right for a white pawn
        elif game.lastMove.endSq[0] < 7 and game.board[game.lastMove.endSq[0]+1][game.lastMove.endSq[1]] == coeff:
            newBoard = [x.copy() for x in game.board]
            #   Move above the black pawn
            newBoard[game.lastMove.endSq[0]][game.lastMove.endSq[1]+coeff] = coeff
            #   Capture the black pawn
            newBoard[game.lastMove.endSq[0]][game.lastMove.endSq[1]] = 0
            #   Remove the white pawn from its starting position
            newBoard[game.lastMove.endSq[0]+1][game.lastMove.endSq[1]] = 0
            if (game.whiteToMove and not inCheck(newBoard)) or (not game.whiteToMove and not inCheck(invert(newBoard))):
                return True

    #   Note that ability to castle need not be checked, since castling is
    #   possible iff the king and rook have other legal moves
    
    return False


#   Inverts board in the expected sense (by rank, file and color)
def invert(board):
    b = [[0 for i in range(8)] for j in range(8)]
    for f in range(8):
        for r in range(8):
            b[f][r] = -1*board[7-f][7-r]
    return b

#   A simple check that a square, when perturbed a distance in both
#   dimensions, remains in bounds on a chess board. Both are 2-tuples of ints.
def inBounds(square, perturb):
    return (square[0] + perturb[0] >= 0) and (square[0] + perturb[0]) <= 7 and \
           (square[1] + perturb[1] >= 0) and (square[1] + perturb[1]) <= 7

#   Return if a legitimate move is legal (ie. doesn't end with the player in
#   check). The passed move is relative to game.board, and should not be a
#   castling move or en passant.
def tryMove(game, move_orig):
    #   Make the current player appear as white
    if game.whiteToMove:
        board = game.board
        move = move_orig
    else:
        board = game.invBoard
        move = move_orig.invert()
    
    #   In these particular cases, we don't have useful shortcuts and
    #   will use the generic inCheck function to test legality
    if move.endPiece == 6 or game.currentlyCheck:
        newBoard = [x.copy() for x in board]
        newBoard[move.startSq[0]][move.startSq[1]] = 0
        newBoard[move.endSq[0]][move.endSq[1]] = move.endPiece
        return not inCheck(newBoard)
        
    #   Find the rank and file of the white king
    kingFile = -1
    found = False

    while kingFile < 7 and not found:
        kingFile += 1
        kingRank = -1
        while kingRank < 7 and not found:
            kingRank += 1
            if board[kingFile][kingRank] == 6:
                found = True
                
    delta_f_start = move.startSq[0] - kingFile
    delta_r_start = move.startSq[1] - kingRank
    if delta_f_start == 0: # Then king and moving piece are on same file
        #   If the moving piece doesn't change files
        if move.endSq[0] - move.startSq[0] == 0:
            return True

        #   Now we must check if there is a checking piece (i.e. rook or
        #   queen) in the same direction on the king's file which would be
        #   "discovered" by the move
        if delta_r_start > 0:
            return not check_along_line(board, move, 0, 1, kingFile, kingRank)
        else:
            return not check_along_line(board, move, 0, -1, kingFile, kingRank)
    else:
        slope_start = delta_r_start / delta_f_start

        delta_f_end = move.endSq[0] - kingFile
        if delta_f_end == 0:
            return not nonVerticalCheck(board, move, slope_start, delta_f_start, kingFile, kingRank)
        else:
            slope_end = (move.endSq[1] - kingRank) / delta_f_end

            #   Movement along the same line cannot discover a check
            if slope_start == slope_end:
                return True

            return not nonVerticalCheck(board, move, slope_start, delta_f_start, kingFile, kingRank)


#   Take a game board and initialized numpy array (netInput), and fill in the
#   numpy array with a representation of the board used for input to the NN.
def piece_to_vector(netInput, piece, c):
    for i in range(-6, 7):
        if piece == i:
            netInput[c] = 1
        c += 1


#   invert: (bool) determines whether to invert board by color (white and
#           black switch)
#   flip0: (bool) reflect the board about the first axis (which is file, by
#           default, or rank if swap is true
#   flip1: (bool) reflect the board about the second axis
#   swap: (bool) switch the first and second axis (this corresponds to some
#           rotation of the board, and its exact effect depends on flip0 and flip1)
def generate_NN_vec(game, invert, flip0, flip1, swap):
    netInput = np.zeros((839,))

    coeff = 1 - 2 * invert
    c = 0
    if not flip0 and not flip1 and not swap:
        for file in range(8):
            for rank in range(8):
                piece = coeff * game.board[file][rank]
                piece_to_vector(netInput, piece, c)
                c += 13
    elif not flip0 and not flip1 and swap:
        for file in range(8):
            for rank in range(8):
                piece = coeff * game.board[rank][file]
                piece_to_vector(netInput, piece, c)
                c += 13
    elif not flip0 and flip1 and not swap:
        for file in range(8):
            for rank in range(8):
                piece = coeff * game.board[file][7-rank]
                piece_to_vector(netInput, piece, c)
                c += 13
    elif not flip0 and flip1 and swap:
        for file in range(8):
            for rank in range(8):
                piece = coeff * game.board[rank][7-file]
                piece_to_vector(netInput, piece, c)
                c += 13
    elif flip0 and not flip1 and not swap:
        for file in range(8):
            for rank in range(8):
                piece = coeff * game.board[7-file][rank]
                piece_to_vector(netInput, piece, c)
                c += 13
    elif flip0 and not flip1 and swap:
        for file in range(8):
            for rank in range(8):
                piece = coeff * game.board[7-rank][file]
                piece_to_vector(netInput, piece, c)
                c += 13
    elif flip0 and flip1 and not swap:
        for file in range(8):
            for rank in range(8):
                piece = coeff * game.board[7-file][7-rank]
                piece_to_vector(netInput, piece, c)
                c += 13
    else: # all are true
        for file in range(8):
            for rank in range(8):
                piece = coeff * game.board[7-rank][7-file]
                piece_to_vector(netInput, piece, c)
                c += 13

    netInput[832] = int(invert + game.whiteToMove == 1)
    netInput[833] = int(game.enPassant)
    if not invert and not flip0:
        netInput[834] = int(game.canW_K_Castle)
        netInput[835] = int(game.canW_Q_Castle)
        netInput[836] = int(game.canB_K_Castle)
        netInput[837] = int(game.canB_Q_Castle)
    elif not invert and flip0:
        netInput[834] = int(game.canW_Q_Castle)
        netInput[835] = int(game.canW_K_Castle)
        netInput[836] = int(game.canB_Q_Castle)
        netInput[837] = int(game.canB_K_Castle)
    elif invert and not flip0:
        netInput[834] = int(game.canB_K_Castle)
        netInput[835] = int(game.canB_Q_Castle)
        netInput[836] = int(game.canW_K_Castle)
        netInput[837] = int(game.canW_Q_Castle)
    else: # both are true
        netInput[834] = int(game.canB_Q_Castle)
        netInput[835] = int(game.canB_K_Castle)
        netInput[836] = int(game.canW_Q_Castle)
        netInput[837] = int(game.canW_K_Castle)
    netInput[838] = game.movesSinceAction

    return tf.constant(netInput, shape=(1,839), dtype=tf.float32)


def verify_data(data, p, withMates=True):
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
            assert data[i][0][0].shape == (1,839), data[i][0][0].shape
            assert data[i][0][1].shape == (1, 1), data[i][0][1].shape
        elif p['mode'] >= 2:
            print("Warning: buffer", i, "was empty.")
