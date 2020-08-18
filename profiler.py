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

net_name = 'res_profile'
train_option = False
generate_option = True
generate_times = 4

def do_train(arg_list):
    #   Given a network, asks the user for training hyper-parameters,
    #   trains the network, and asks what to do next.
    net, tBuffer, vBuffer = arg_list
    p = input_handling.readConfig(1)
    for i in range(2, 4):
        p.update(input_handling.readConfig(i))

    if p['mode'] >= 3:
        random.seed(0)

    #   Now do an episode of data generation/training
    main.trainOption(net, tBuffer, vBuffer, 1)

def do_generate_examples(arg_list):
    net, p, times = arg_list
    
    if p['mode'] >= 3:
        random.seed(0)
    
    for i in range(times):
        data = q_learn.generateExamples(net, p)
        


p = input_handling.readConfig(1)
for i in range(2, 4):
    p.update(input_handling.readConfig(i))
    
#   print relevant information about parameters that affect
#   computational time
param_names = ['epochs', 'baseBreadth', 'maxSteps', 'breadth', 'depth',
               'epsGreedy', 'epsSearch', 'batchSize', 'mode']
for x in param_names:
    print(x + ': ', p[x])
        
if train_option:
    net, tBuffer, vBuffer = Network.load('nets/' + net_name, data_prefix='profile')

    net.print()
    print('tBuffer and vBuffer sizes: ', sum([len(x) for x in tBuffer]), ',',
          sum([len(x) for x in vBuffer]))
    print('\n-----------------------')
    print('  main.trainOption')
    print('-----------------------\n')

    cProfile.run('do_train([net,tBuffer,vBuffer])')

if generate_option:
    #   Reload network since generated examples are in nondeterministic order,
    #   subtly affecting network parameters after training
    net, tBuffer, vBuffer = Network.load('nets/' + net_name, data_prefix='profile')

    if not train_option:
        net.print()

    print('\n-----------------------')
    print('  q_learn.generateExamples')
    print('-----------------------\n')

    cProfile.run('do_generate_examples([net, p, generate_times])')
