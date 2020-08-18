import time
import numpy as np

repeats = 100000

def new(board):
    kingFile = -1
    kingRank = -1
    found = False

    while kingFile < 7 and not found:
        kingFile += 1
        while kingRank < 7 and not found:
            kingRank += 1
            if board[kingFile][kingRank] == 6:
                found = True

def orig(board):
    for file in range(8):
        for rank in range(8):
            if board[file][rank] == 6:
                kingRank = rank
                kingFile = file

def make_board():
    board = [[0 for i in range(8)] for j in range(8)]

    king_pos = np.random.randint(64)
    rank = king_pos // 8
    file = king_pos % 8

    board[file][rank] = 6
    return board


orig_time = time.time()
for i in range(repeats):
    orig(make_board())
orig_time = time.time() - orig_time

new_time = time.time()
for i in range(repeats):
    new(make_board())
new_time = time.time() - new_time

print("Orig time:", orig_time)
print("New time:", new_time)
