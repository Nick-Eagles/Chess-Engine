import sys
import numpy as np
import random
sys.path.append('../')

import network_helper
import input_handling

positions = network_helper.readGames('../data/checkmates.csv')
keepGoing = True
while keepGoing:
    rint = np.random.randint(len(positions))
    network_helper.toFEN(positions[rint], 'random_position.fen')
    
    messDef = "Generate .fen file for another random position (y/n)? "
    messOnErr = "Not a valid option."
    cond = 'var == "y" or var == "n"'
    choice = input_handling.getUserInput(messDef, messOnErr, 'str', cond)
    keepGoing = choice == 'y'
    
