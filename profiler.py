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

net_name = 'res3'

def do_train(arg_list):
    random.seed(0)
    #   Given a network, asks the user for training hyper-parameters,
    #   trains the network, and asks what to do next.
    net, tBuffer, vBuffer = arg_list
    p = input_handling.readConfig(1)
    for i in range(2, 4):
        p.update(input_handling.readConfig(i))

    #   print relevant information about parameters that affect
    #   computational time
    param_names = ['epochs', 'baseBreadth', 'maxSteps', 'breadth', 'depth',
                   'epsGreedy', 'epsSearch', 'batchSize']
    for x in param_names:
        print(x + ': ', p[x])

    #   Now do an episode of data generation/training
    main.trainOption(net, tBuffer, vBuffer, 1)

def do_generate_examples(arg_list):
    random.seed(0)
    net, p = arg_list
    data = q_learn.generateExamples(net, p)
        
    
net, tBuffer, vBuffer = Network.load('nets/' + net_name)
p = input_handling.readConfig(1)
p.update(input_handling.readConfig(3))

net.print()
print('tBuffer and vBuffer sizes: ', sum([len(x) for x in tBuffer]), ',',
      sum([len(x) for x in vBuffer]))
print('\n-----------------------')
print('  main.trainOption')
print('-----------------------\n')

cProfile.run('do_train([net,tBuffer,vBuffer])')

#   Reload network since generated examples are in nondeterministic order,
#   subtly affecting network parameters after training
net, tBuffer, vBuffer = Network.load('nets/' + net_name)

print('\n-----------------------')
print('  q_learn.generateExamples')
print('-----------------------\n')

cProfile.run('do_generate_examples([net, p])')
