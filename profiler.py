import Network
import network_helper
import file_IO
import input_handling
import Traversal
import Game
import misc
import q_learn
import main
import board_helper
import policy
import Move

import sys
import os
import shutil
import csv
import random
import numpy as np
from scipy.special import expit, logit
import cProfile

def do_train(arg_list):
    #   Given a network, asks the user for training hyper-parameters,
    #   trains the network, and asks what to do next.
    net, tBuffer, vBuffer = arg_list
    p = input_handling.readConfig(1)
    for i in range(2, 4):
        p.update(input_handling.readConfig(i))

    #   print relevant information about parameters that affect
    #   computational time
    param_names = ['epochs', 'baseBreadth', 'maxSteps', 'breadth', 'depth', 'epsGreedy', 'batchSize']
    for x in param_names:
        print(x + ': ', p[x])

    #   Now do an episode of data generation/training
    main.trainOption(net, tBuffer, vBuffer, 1)
        
    
net, tBuffer, vBuffer = Network.load('nets/8deep6')

net.print()
print('tBuffer and vBuffer sizes: ', sum([len(x) for x in tBuffer]), ',', sum([len(x) for x in vBuffer]))

cProfile.run('do_train([net,tBuffer,vBuffer])')

