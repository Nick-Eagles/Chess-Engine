def canUncastle(game):
    if inCheck(game.board) or inCheck(game.invBoard):
        return (False, False)
    
    #   For white -------
    if game.whiteToMove:
        #   Kingside check
        if game.board[4][0] == 0 and game.board[5][0] == 4 and game.board[6][0] == 6 and game.board[7][0] == 0:
            newBoard = [b.copy() for b in game.board]
            newBoard2 = [b.copy() for b in game.board]
            newBoard[6][0] = 0
            newBoard2[6][0] = 0
            newBoard[5][0] = 6
            newBoard2[4][0] = 6
            if not inCheck(newBoard) and not inCheck(newBoard2):
                kingside = True
            else:
                kingside = False
        else:
            kingside = False

        #   Queenside check
        if game.board[0][0] == 0 and game.board[1][0] == 0 and game.board[2][0] == 6 and game.board[3][0] == 4 and game.board[4][0] == 0:
            newBoard = [b.copy() for b in game.board]
            newBoard2 = [b.copy() for b in game.board]
            newBoard[2][0] = 0
            newBoard2[2][0] = 0
            newBoard[3][0] = 6
            newBoard2[4][0] = 6
            if not inCheck(newBoard) and not inCheck(newBoard2):
                queenside = True
            else:
                queenside = False
        else:
            queenside = False
    else:
        #   Kingside check
        if game.invBoard[3][0] == 0 and game.invBoard[2][0] == 4 and game.invBoard[1][0] == 6 and game.invBoard[0][0] == 0:
            newBoard = [b.copy() for b in game.invBoard]
            newBoard2 = [b.copy() for b in game.invBoard]
            newBoard[1][0] = 0
            newBoard2[1][0] = 0
            newBoard[2][0] = 6
            newBoard2[3][0] = 6
            if not inCheck(newBoard) and not inCheck(newBoard2):
                kingside = True
            else:
                kingside = False
        else:
            kingside = False

        #   Queenside check
        if game.invBoard[7][0] == 0 and game.invBoard[6][0] == 0 and game.invBoard[5][0] == 6 and game.invBoard[4][0] == 4 and game.invBoard[3][0] == 0:
            newBoard = [b.copy() for b in game.invBoard]
            newBoard2 = [b.copy() for b in game.invBoard]
            newBoard[5][0] = 0
            newBoard2[5][0] = 0
            newBoard[3][0] = 6
            newBoard2[4][0] = 6
            if not inCheck(newBoard) and not inCheck(newBoard2):
                queenside = True
            else:
                queenside = False
        else:
            queenside = False
            
    return (kingside, queenside)


def processRevMove(game, moves, m):
    #   Copy original game and step back one move
    g = copy.deepcopy(game)
    g.whiteToMove = not g.whiteToMove
    g.doMove(m)
    g.whiteToMove = not g.whiteToMove

    if g.whiteToMove:
        board = game.board
        invBoard = game.invBoard
    else:
        board = game.invBoard
        invBoard = game.board
        
    thisInCheck = board_helper.inCheck(board)
    thatInCheck = board_helper.inCheck(invBoard)
    #   If this position is stalemate, or if the player who loses by checkmate is
    #   in check this position (check 2 turns in a row), the reverse move is illegitimate.
    if (not thisInCheck and len(board_helper.getLegalMoves(self)) == 0) or thatInCheck:
        return moves
    
    #   Otherwise we can add the reverse move to the list of legal reverse moves
    moves.append(m)
    return moves
        
#   For now, it is assumed that the game passed has ended by checkmate.
def getLegalMoves(game):
    moves = []

    if game.whiteToMove:
        board = game.invBoard
    else:
        board = game.board
    for file in range(8):
        for rank in range(8):
            piece = board[file][rank]
            assert piece >= 0 and piece <= 12, "Tried to get legal moves with a corrupt board"
            if piece == 0 or piece > 6:
                continue

            #   Pawn --------------------------------------------------------------
            if piece == 1 and rank > 1 and board[file][rank-1] == 0:
                #   Moving backward one square
                m = Move.Move((file, rank), (file, rank-1), 1)
                moves = processRevMove(game, moves, m)
                    
                #   Moving back 2 squares to starting position
                if rank == 3 and board[file][1] == 0:
                    m = Move.Move((file, 3), (file, 1), 1)
                    moves = processRevMove(game, moves, m)
                    
            elif piece == 2:
                horizontals = [-2, -2, -1, -1, 1, 1, 2, 2]
                verticals = [-1, 1, -2, 2, -2, 2, -1, 1]
                #   Movement in any legal direction to an empty square
                for x,y in zip(horizontals, verticals):
                    if inBounds((file,rank), (x,y)) and board[file+x][rank+y] == 0:
                        m = Move.Move((file, rank), (file+x, rank+y), 2)
                        moves = processRevMove(game, moves, m)
                        
            #   King
            elif piece == 6:
                ranks = list(range(max(0,rank-1),min(8,rank+2)))
                files = list(range(max(0,file-1),min(8,file+2)))
                for r in ranks:
                    for f in files:
                        if board[f][r] == 0:
                            m = Move.Move((file,rank),(f,r), 6)
                            moves = processRevMove(game, moves, m)
                                
            #   Rook, bishop, and queen are more complicated to deal with individually
            #   ----------------------------------------------------------------------
            else:
                #   Diagonals (bishop or queen)
                if piece == 3 or piece == 5:
                    #   MOVEMENT ALONG NE DIAGONAL
                    i = 1
                    while file+i <= 7 and rank+i <= 7:
                        if board[file+i][rank+i] == 0:
                            m = Move.Move((file,rank),(file+i,rank+i), piece)
                            moves = processRevMove(game, moves, m)
                            i += 1
                        else:
                            i = 8
                    #   MOVEMENT ALONG NW DIAGONAL
                    i = 1
                    while file-i >= 0 and rank+i <= 7:
                        if board[file-i][rank+i] == 0:
                            m = Move.Move((file,rank),(file-i,rank+i), piece)
                            moves = processRevMove(game, moves, m)
                            i += 1
                        else:
                            i = 8
                    #   MOVEMENT ALONG SE DIAGONAL
                    i = 1
                    while file+i <= 7 and rank-i >= 0:
                        if board[file+i][rank-i] == 0:
                            m = Move.Move((file,rank),(file+i,rank-i), piece)
                            moves = processRevMove(game, moves, m)
                            i += 1
                        else:
                            i = 8
                    #   MOVEMENT ALONG SW DIAGONAL
                    i = 1
                    while file-i >= 0 and rank-i >= 0:
                        if board[file-i][rank-i] == 0:
                            m = Move.Move((file,rank),(file-i,rank-i), piece)
                            moves = processRevMove(game, moves, m)
                            i += 1
                        else:
                            i = 8
                            
                #   Lateral, up/down movement (rook or queen)
                if piece == 4 or piece == 5:
                    #   MOVEMENT UPWARD IN A FILE
                    i = 1
                    while rank+i <= 7:
                        if board[file][rank+i] == 0:
                            m = Move.Move((file,rank),(file,rank+i), piece)
                            moves = processRevMove(game, moves, m)
                            i += 1
                        else:
                            i = 8
                    #   MOVEMENT DOWNWARD IN A FILE
                    i = 1
                    while rank-i >= 0:
                        if board[file][rank-i] == 0:
                            m = Move.Move((file,rank),(file,rank-i), piece)
                            moves = processRevMove(game, moves, m)
                            i += 1
                        else:
                            i = 8
                    #   MOVEMENT TO THE RIGHT
                    i = 1
                    while file+i <= 7:
                        if board[file+i][rank] == 0:
                            m = Move.Move((file,rank),(file+i,rank), piece)
                            moves = processRevMove(game, moves, m)
                            i += 1
                        else:
                            i = 8
                    #   MOVEMENT TO THE LEFT
                    i = 1
                    while file-i >= 0:
                        if board[file-i][rank] == 0:
                            m = Move.Move((file,rank),(file-i,rank), piece)
                            moves = processRevMove(game, moves, m)
                            i += 1
                        else:
                            i = 8
    return moves              
