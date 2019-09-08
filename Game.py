import numpy as np
import datetime

import Move
import board_helper

class Game:
    def __init__(self, quiet=True):
        self.board = np.array([[4, 1, 0, 0, 0, 0, -1, -4], [2, 1, 0, 0, 0, 0, -1, -2], \
                          [3, 1, 0, 0, 0, 0, -1, -3], [5, 1, 0, 0, 0, 0, -1, -5], \
                          [6, 1, 0, 0, 0, 0, -1, -6], [3, 1, 0, 0, 0, 0, -1, -3], \
                          [2, 1, 0, 0, 0, 0, -1, -2], [4, 1, 0, 0, 0, 0, -1, -4]])
        self.invBoard = np.zeros((8,8), dtype=np.int8)
        
        #   Amount of the pieces on the board: P, N, light B, dark B, R, Q
        self.wPieces = np.array([8, 2, 1, 1, 2, 1])
        self.bPieces = np.array([8, 2, 1, 1, 2, 1])

        #   These only become false when castling has occurred or a rook
        #   or king has moved, making castling illegal *for the game*.
        self.canW_K_Castle = True
        self.canW_Q_Castle = True
        self.canB_K_Castle = True
        self.canB_Q_Castle = True

        self.wValue = 43
        self.bValue = 43

        self.whiteToMove = True
        self.enPassant = False
        self.quiet = quiet

        self.movesSinceAction = 0
        self.gameResult = 17    #   Literally a random number, but will become +/-1 or 0
        self.gameResultStr = ""
        self.moveNum = 1
        self.annotation = ""
        self.lastMove = Move.Move((0, 6), (0, 7), -1) # No particular meaning, simply initialized

    def copy(self):
        g = Game()

        g.board = self.board.copy()
        g.wPieces = self.wPieces.copy()
        g.bPieces = self.bPieces.copy()
        g.canW_K_Castle = self.canW_K_Castle
        g.canW_Q_Castle = self.canW_Q_Castle
        g.canB_K_Castle = self.canB_K_Castle
        g.canB_Q_Castle = self.canB_Q_Castle
        g.wValue = self.wValue
        g.bValue = self.bValue
        g.whiteToMove = self.whiteToMove
        g.enPassant = self.enPassant
        g.quiet = self.quiet
        g.movesSinceAction = self.movesSinceAction
        g.gameResult = self.gameResult
        g.gameResultStr = self.gameResultStr
        g.moveNum = self.moveNum
        g.annotation = self.annotation
        g.lastMove = self.lastMove

        return g

    def updateValues(self):
        vals = np.array([1, 3, 3, 3, 5, 9])

        self.wValue = vals @ self.wPieces.T + 4     
        self.bValue = vals @ self.bPieces.T + 4

        return float(self.wValue / self.bValue)

    #   Return the (absolute reward for doing move "move" (positive means to the
    #   benefit of white), NN input vector for the resulting position) as a tuple
    def getReward(self, move, mateRew):
        g = self.copy()
        g.quiet = True
        g.doMove(move)

        if abs(g.gameResult) == 1:
            return (g.gameResult * mateRew, g.toNN_vecs()[0])
        elif g.gameResult == 0:
            return (np.log(self.bValue / self.wValue), g.toNN_vecs()[0])
        else:
            return (np.log(g.wValue * self.bValue / (self.wValue * g.bValue)), g.toNN_vecs()[0])

    #   Probably can eliminate; np arrays print the correct way I think
    def printBoard(self):
        print("board:  -----")
        for rank in range(8):
            for file in range(8):
                piece = self.board[file][7-rank]
                print(piece, end = ' ')
            print()
        print("-------------")

    #   Converts the board into a standardized input vector for the NN:
    #   for each square, the input vector has 13 entries so that only one
    #   entry has a value of 1 (the remaining are zero), corresponding to
    #   the piece occupying that square. The remaining pieces of information
    #   necessary to specify a unique board are appended to the vector:
    #   game.whiteToMove, game.enPassant, castling permissions, and moves
    #   since a pawn move or capture. All entries are again 0/1 except
    #   game.movesSinceAction: {0,1,...,50}
    #
    #   This process is performed twice; the second time essentially "flips"
    #   the game so that the board is inverted by rank, castling
    #   info and piece values are swapped by color, and the whiteToMove bool
    #   is negated. The idea is that the network can learn from a conceptually
    #   equivalent inverted game (the board is flipped and white pretends
    #   it's black and vice versa), and will learn that most dynamics are
    #   independent of what color you're playing, via this trick. This function
    #   returns a tuple (normal inputVec, inverted game's inputVec)
    def toNN_vecs(self):
        netInput = []
        netInputInv = []
        #   Convert board information: note netInputInv is taking the
        #   original board information and inverting it by rank and file,
        #   then swapping the piece colors.
        for file in range(8):
            for rank in range(8):
                #   Get the piece values at the squares
                piece = self.board[file][rank]
                pieceInv = -1*self.board[file][7-rank]

                #   Encode the piece values as binary sequences
                for i in range(-6, 7):
                    if piece == i:
                        netInput.append(1)
                    else:
                        netInput.append(0)
                    if pieceInv == i:
                        netInputInv.append(1)
                    else:
                        netInputInv.append(0)

        #   Append remaining information about the game
        netInput.append(self.whiteToMove)
        netInput.append(self.enPassant)
        netInput.append(self.canW_K_Castle)
        netInput.append(self.canW_Q_Castle)
        netInput.append(self.canB_K_Castle)
        netInput.append(self.canB_Q_Castle)
        netInput.append(self.movesSinceAction)

        #   Same, but castling bools are swapped by color and
        #   whitetoMove is inverted since colors are swapped
        netInputInv.append(not self.whiteToMove)
        netInputInv.append(self.enPassant)
        netInputInv.append(self.canB_K_Castle)
        netInputInv.append(self.canB_Q_Castle)
        netInputInv.append(self.canW_K_Castle)
        netInputInv.append(self.canW_Q_Castle)
        netInputInv.append(self.movesSinceAction)

        finalInput = np.array(netInput).reshape(-1,1)
        finalInputInv = np.array(netInputInv).reshape(-1,1)
        return (finalInput, finalInputInv)

    #   Returns True/False
    def isCheckmate(self):
        check = (self.whiteToMove and board_helper.inCheck(self.board)) or \
                (not self.whiteToMove and board_helper.inCheck(board_helper.invert(self.board)))
        return check and len(board_helper.getLegalMoves(self)) == 0


    #   Returns a tuple: the first entry is True/False; the second is a string
    #   describing details (such as "Draw by stalemate")
    def isDraw(self):
        coeff = 2 * self.whiteToMove - 1

        ########################################################
        #   Stalemate
        ########################################################
        check = (self.whiteToMove and board_helper.inCheck(self.board)) or \
                (not self.whiteToMove and board_helper.inCheck(board_helper.invert(self.board)))
        if not check and len(board_helper.getLegalMoves(self)) == 0:
            note = "Draw by stalemate."
            return (True, note)

        ########################################################  
        #   Insufficient material
        ########################################################
        
        #   Checks for any pawns, rooks, or queens
        if sum([self.wPieces[i] + self.bPieces[i] for i in [0, 4, 5]]) == 0:  
            #   A bool representing if either side has both bishop colors (which would mean
            #   the game is not a draw)
            bothColors =(self.wPieces[2] > 0 and self.wPieces[3] > 0) or \
                        (self.bPieces[2] > 0 and self.bPieces[3] > 0)
            if not bothColors:
                #   A bool ultimately deciding whether both sides have only bishops
                #   of one color
                colorMatch = (self.wPieces[2] > 0) == (self.bPieces[2] > 0)
                numKnights = self.wPieces[1] + self.bPieces[1]
                
                #   Equivalent to checking if there are bishops on the board and
                #   they are all on the same color
                if colorMatch and self.wPieces[2] + self.wPieces[3] > 0:
                    if numKnights == 0:
                        note = "Draw by insufficient material."
                        return (True, note)
                elif colorMatch:    # in this case meaning no bishops are on the board
                    if numKnights < 2:
                        note = "Draw by insufficient material."
                        return (True, note)
                    
        #   REPETITION
        
        ########################################################
        #   50 MOVE RULE
        ########################################################
        
        if self.movesSinceAction >= 50:
            note = "Draw: no capture or pawn advance in 50 moves."
            return (True, note)

        return (False, "")

    #   Changes the board and relevant class attributes for any move; does not entirely
    #   verify legitimacy of move.
    def doMove(self, move):
        coeff = 2 * self.whiteToMove - 1
        assert move.endPiece != 0, "Piece was likely deleted.\n" + self.verbosePrint()
        assert coeff * move.endPiece > 0, "Tried to move opponent's piece.\n" + self.verbosePrint()
        assert self.wValue >= 4 and self.bValue >= 4, "See what went wrong, saving game to PGN:" + str(self.toPGN()) + self.verbosePrint()
        
        self.lastMove = move
        self.enPassant = False

        if self.whiteToMove:
            self.annotation += str(self.moveNum) + ". " + move.getMoveName(self.board)
            piecesList = self.wPieces
            oppPiecesList = self.bPieces
        else:
            self.annotation += move.getMoveName(self.board) + "\n"
            piecesList = self.bPieces
            oppPiecesList = self.wPieces

        startPiece = self.board[move.startSq[0]][move.startSq[1]]
        pieceRemoved = self.board[move.endSq[0]][move.endSq[1]]
        assert startPiece != 0, "Tried to move a piece from an empty square: " + move.getMoveName(self.board) + self.verbosePrint()
        assert coeff*pieceRemoved <= 0, "Tried to capture own piece: " + move.getMoveName(self.board) + self.verbosePrint()
        assert abs(pieceRemoved) != 6, "About to capture opponent's king: " + move.getMoveName(self.board) + self.verbosePrint()       

        #   Sees if the move is a capture or pawn move
        if pieceRemoved != 0 or abs(startPiece) == 1:
            self.movesSinceAction = 0
        else:
            self.movesSinceAction += 0.5

        #   Identify when a pawn moves 2 squares and creates an
        #   en passant opportunity
        if abs(startPiece) == 1 and abs(move.endSq[1] - move.startSq[1]) == 2:
            #   Opposite color pawn on the left of where this pawn moved
            if move.startSq[0] > 0 and self.board[move.endSq[0]-1][move.endSq[1]] == -1*startPiece:
                self.enPassant = True
            #   To the right
            elif move.startSq[0] < 7 and self.board[move.endSq[0]+1][move.endSq[1]] == -1*startPiece:
                self.enPassant = True
            
        #   In case of promotion, properly update number of same color pieces
        #   of each type
        if startPiece != move.endPiece:
            assert coeff*move.endPiece >= 2 and coeff*move.endPiece <= 5, "Tried to promote to value " + move.endPiece + ".\n" + self.verbosePrint()
            piecesList[0] -= 1
            if abs(move.endPiece) == 3:
                if (move.endSq[0] - move.endSq[1]) % 2 == 0:
                    piecesList[3] += 1
                else:
                    piecesList[2] += 1
            elif abs(move.endPiece) >= 4:
                piecesList[coeff*move.endPiece] += 1
            else:   # == 2
                piecesList[1] += 1

        #   Update number of opposite color pieces of each type                  
        if abs(pieceRemoved) == 3:
            if (move.endSq[0] - move.endSq[1]) % 2 == 0:
                oppPiecesList[3] -= 1
            else:
                oppPiecesList[2] -= 1
        elif abs(pieceRemoved) >= 4:
            oppPiecesList[abs(pieceRemoved)] -= 1
        elif abs(pieceRemoved) == 1 or abs(pieceRemoved) == 2:
            oppPiecesList[abs(pieceRemoved)-1] -= 1
            
        #   Move rook to complete castling, if applicable
        backRank = 7*(not self.whiteToMove)
        if move.endSq[0] - move.startSq[0] == 2 and abs(startPiece) == 6:    # kingside castling
            self.board[5][backRank] = 4*coeff
            self.board[7][backRank] = 0
        elif move.endSq[0] - move.startSq[0] == -2 and abs(startPiece) == 6: # queenside castling
            self.board[3][backRank] = 4*coeff
            self.board[0][backRank] = 0

        #   Set castling permissions to false if king or rooks have moved
        if move.endPiece == 6:
            self.canW_K_Castle = False
            self.canW_Q_Castle = False
        elif move.endPiece == 4 and move.startSq[0] == 0 and move.startSq[1] == 0:
            self.canW_Q_Castle = False
        elif move.endPiece == 4 and move.startSq[0] == 7 and move.startSq[1] == 0:
            self.canW_K_Castle = False
        elif move.endPiece == -6:
            self.canB_K_Castle = False
            self.canB_Q_Castle = False
        elif move.endPiece == -4 and move.startSq[0] == 0 and move.startSq[1] == 7:
            self.canB_Q_Castle = False
        elif move.endPiece == -4 and move.startSq[0] == 7 and move.startSq[1] == 7:
            self.canB_K_Castle = False

        #   Also revoke permissions for opposite color if rook is captured
        if move.endSq == (0, 0):
            self.canW_Q_Castle = False
        elif move.endSq == (7, 0):
            self.canW_K_Castle = False
        elif move.endSq == (0, 7):
            self.canB_Q_Castle = False
        elif move.endSq == (7, 7):
            self.canB_K_Castle = False

        #   Modify board
        self.board[move.startSq[0]][move.startSq[1]] = 0                
        self.board[move.endSq[0]][move.endSq[1]] = move.endPiece

        #   Capture the pawn if en passant
        if pieceRemoved == 0 and abs(startPiece) == 1 and move.startSq[0] != move.endSq[0]:
            self.board[move.endSq[0]][move.endSq[1]-coeff] = 0
            oppPiecesList[0] -= 1

        if not self.whiteToMove:
            self.moveNum += 1
            
        if self.whiteToMove and not self.quiet and self.moveNum % 20 == 0:
            print("Done move", self.moveNum)           
        
        self.whiteToMove = not self.whiteToMove
        self.updateValues()

        #   See if resulting position (after moving) is a checkmate or draw
        if self.isCheckmate():
            if self.whiteToMove:
                self.gameResult = -1
            else:
                self.gameResult = 1
        else:
            draw = self.isDraw()
            if draw[0]:
                self.gameResult = 0
                self.gameResultStr = draw[1]

    #   For debugging; prints info about the current game and returns the empty string,
    #   so that assert statements can easily include a call to this function
    def verbosePrint(self):
        coeff = 2 * self.whiteToMove - 1
        if self.whiteToMove:
            agent = "white"
        else:
            agent = "black"
            
        print("\n$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        print(" Info about the current game:")
        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        print("self.canW_K_Castle: ", self.canW_K_Castle)
        print("self.canW_Q_Castle: ", self.canW_Q_Castle)
        print("self.canB_K_Castle: ", self.canB_K_Castle)
        print("self.canB_Q_Castle: ", self.canB_Q_Castle)
        print("self.enPassant: ", self.enPassant)
        print("self.movesSinceAction: ", self.movesSinceAction)
        print("self.moveNum: ", self.moveNum)
        print("self.gameResult (will be 17 if game isn't done): ", self.gameResult)
        print("self.wValue:", self.wValue)
        print("self.bValue:", self.bValue)
        print("self.annotation: ", self.annotation)
        
        check = (self.whiteToMove and not board_helper.inCheck(self.board)) or \
                (not self.whiteToMove and not board_helper.inCheck(board_helper.invert(self.board)))
        print("It is " + agent + " to move and " + agent + " is" + (check * " not") + " in check.")
        self.printBoard()

        return ""

    def toPGN(self):
        date = datetime.datetime.now()
        dateStr = '"' + str(date.year) + '.' + str(date.month) + '.' + str(date.day) + '"'
        if self.gameResult == -1:
            result = '0-1'
        elif self.gameResult == 0:
            result = '1/2-1/2'
        else:
            result = '1-0'
            
        txt = '[Event "Engine self-play"]\n[Site "Internal"]\n[Date ' + dateStr + ']\n' \
              + '[Round "1"]\n[White "Sculps Engine"]\n[Black "Sculps Engine"]\n' \
              + '[Result "' + result + '"]\n\n'
        txt = txt + self.annotation + result

        #   Write to pgn in current directory
        filename = 'visualization/latest_game.pgn'
        try:
            with open(filename, 'w') as pgn_file:
                pgn_file.write(txt)
        except:
            print("Encountered an error while opening or handling file 'latest_game.pgn'")
        finally:
            pgn_file.close()
