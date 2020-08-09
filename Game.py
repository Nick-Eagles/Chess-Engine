import numpy as np
import datetime

import Move
import board_helper

class Game:
    def __init__(self, quiet=True):
        self.board = [[4, 1, 0, 0, 0, 0, -1, -4], [2, 1, 0, 0, 0, 0, -1, -2], \
                     [3, 1, 0, 0, 0, 0, -1, -3], [5, 1, 0, 0, 0, 0, -1, -5], \
                     [6, 1, 0, 0, 0, 0, -1, -6], [3, 1, 0, 0, 0, 0, -1, -3], \
                     [2, 1, 0, 0, 0, 0, -1, -2], [4, 1, 0, 0, 0, 0, -1, -4]]
        self.invBoard = [[0 for i in range(8)] for j in range(8)]
        
        #   Amount of the pieces on the board: P, N, light B, dark B, R, Q
        self.wPieces = [8, 2, 1, 1, 2, 1]
        self.bPieces = [8, 2, 1, 1, 2, 1]

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
        self.gameResultStr = "Not a terminal position"
        self.moveNum = 1
        self.annotation = ""
        self.lastMove = Move.Move((0, 6), (0, 7), -1) # No particular meaning, simply initialized

    def copy(self):
        g = Game()

        g.board = [x.copy() for x in self.board]
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
        vals = [1, 3, 3, 3, 5, 9]

        self.wValue = sum([vals[i] * self.wPieces[i] for i in range(6)]) + 4
        self.bValue = sum([vals[i] * self.bPieces[i] for i in range(6)]) + 4

    #   Return the (absolute reward for doing move "move" (positive means to the
    #   benefit of white), NN input vector for the resulting position) as a tuple
    def getReward(self, move, mateRew, simple=False, copy=True):
        if copy:
            g = self.copy()
            g.quiet = True
        else:
            g = self

        #   Save original material values, since g may be self
        old_b_value = g.bValue
        old_w_value = g.wValue
        
        g.doMove(move)

        if simple:
            NN_vecs = np.array([])
        else:
            NN_vecs = g.toNN_vecs(every=False)[0]
            
        if abs(g.gameResult) == 1:
            return (g.gameResult * mateRew, NN_vecs)
        elif g.gameResult == 0:
            return (np.log(old_b_value / old_w_value), NN_vecs)
        else:
            return (np.log(g.wValue * old_b_value / (old_w_value * g.bValue)), NN_vecs)

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
    #   This function by default returns additional permutations of the game
    #   board when possible while retaining expected game outcome (possibly
    #   inverted). This behavior is suppressed (and only a single NN input
    #   returned) by setting every=False. In both cases, a list of at least
    #   one numpy array is returned.
    def toNN_vecs(self, every=True):
        #   The original position as-is
        all_vecs = [board_helper.generate_NN_vec(self, False, False, False, False)]

        if every:
            #   The position inverted by rank and color
            all_vecs.append(board_helper.generate_NN_vec(self, True, False, True, False))

            #   If castling is impossible, the board can be reflected by file
            if not any([self.canW_K_Castle, self.canW_Q_Castle, self.canB_K_Castle, self.canB_Q_Castle]):
                #   The positions we've appended so far but inverted by file
                all_vecs.append(board_helper.generate_NN_vec(self, False, True, False, False))
                all_vecs.append(board_helper.generate_NN_vec(self, True, True, True, False))

                #   If there are no pawns on the board, it can also be rotated arbitrarily
                if self.wPieces[0] + self.bPieces[0] == 0:
                    #   All other unique permutations involving reflections and rotations
                    #   of the board- to make 16 total
                    all_vecs.append(board_helper.generate_NN_vec(self, False, False, False, True))
                    all_vecs.append(board_helper.generate_NN_vec(self, False, False, True, False))
                    all_vecs.append(board_helper.generate_NN_vec(self, False, False, True, True))
                    all_vecs.append(board_helper.generate_NN_vec(self, False, True, False, True))
                    all_vecs.append(board_helper.generate_NN_vec(self, False, True, True, False))
                    all_vecs.append(board_helper.generate_NN_vec(self, False, True, True, True))
                    all_vecs.append(board_helper.generate_NN_vec(self, True, False, False, False))
                    all_vecs.append(board_helper.generate_NN_vec(self, True, False, False, True))
                    all_vecs.append(board_helper.generate_NN_vec(self, True, False, True, True))
                    all_vecs.append(board_helper.generate_NN_vec(self, True, True, False, False))
                    all_vecs.append(board_helper.generate_NN_vec(self, True, True, False, True))
                    all_vecs.append(board_helper.generate_NN_vec(self, True, True, True, True))

        return all_vecs
                

    #   Returns a tuple: the first entry is the game result; the second is a string
    #   describing details (such as "Draw by stalemate")
    def updateResult(self):
        coeff = 1 - 2 * self.whiteToMove

        ########################################################
        #   See if the position is checkmate
        ########################################################
        
        check = (self.whiteToMove and board_helper.inCheck(self.board)) or \
                (not self.whiteToMove and board_helper.inCheck(board_helper.invert(self.board)))
        moves = board_helper.getLegalMoves(self)

        if check and len(moves) == 0:
            return (coeff, "Checkmate")

        #   Rule out any draws
        if self.moveNum < 10:
            return (17, "Not a terminal position")

        ########################################################
        #   Stalemate
        ########################################################

        if not check and len(moves) == 0:
            return (0, "Draw by stalemate.")

        ########################################################
        #   50 MOVE RULE
        ########################################################
        
        if self.movesSinceAction >= 50:
            note = "Draw: no capture or pawn advance in 50 moves."
            return (0, note)

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
                if colorMatch and (self.wPieces[2] + self.wPieces[3] > 0) or (self.wPieces[3] + self.wPieces[2] > 0):
                    if numKnights == 0:
                        note = "Draw by insufficient material."
                        return (0, note)
                elif colorMatch:    # in this case meaning no bishops are on the board
                    if numKnights < 2:
                        note = "Draw by insufficient material."
                        return (0, note)

        return (17, "Not a terminal position")

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
            if not self.quiet:
                self.annotation += str(self.moveNum) + ". " + move.getMoveName(self.board) + " "
            piecesList = self.wPieces
            oppPiecesList = self.bPieces
        else:
            if not self.quiet:
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
        self.gameResult, self.gameResultStr = self.updateResult()
        

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

    def toPGN(self, filename='visualization/latest_game.pgn'):
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

        try:
            with open(filename, 'w') as pgn_file:
                pgn_file.write(txt)
        except:
            print("Encountered an error while opening or handling file 'latest_game.pgn'")
        finally:
            pgn_file.close()
