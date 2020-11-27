import Session
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
import tensorflow as tf

net_name = 'tf_profile'
train_option = False
generate_option = True
generate_times = 1

'''
config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=4,
                        inter_op_parallelism_threads=2,
                        allow_soft_placement=True,
                        device_count = {'CPU': 4})
session = tf.compat.v1.Session(config=config)


os.environ["OMP_NUM_THREADS"] = "4"
#os.environ["KMP_BLOCKTIME"] = "30"
#os.environ["KMP_SETTINGS"] = "1"
#os.environ["KMP_AFFINITY"]= "granularity=fine,verbose,compact,1,0"
'''

def do_train(session):
    #   Given a network, asks the user for training hyper-parameters,
    #   trains the network, and asks what to do next.
    p = input_handling.readConfig(1)
    for i in range(2, 4):
        p.update(input_handling.readConfig(i))

    if p['mode'] >= 3:
        random.seed(0)

    #   Now do an episode of data generation/training
    main.trainOption(session, 1)

def do_generate_examples(arg_list):
    session, p, times = arg_list
    
    if p['mode'] >= 3:
        random.seed(0)
    
    for i in range(times):
        data = q_learn.generateExamples(session.net, p)
        


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
    session = Session.Session([], [])
    session.Load('nets/' + net_name, data_prefix='tf_profile')

    session.net.summary()
    print('tBuffer and vBuffer sizes: ', sum([len(x) for x in session.tBuffer]), ',',
          sum([len(x) for x in session.vBuffer]))
    print('\n-----------------------')
    print('  main.trainOption')
    print('-----------------------\n')

    cProfile.run('do_train(session)')

if generate_option:
    #   Reload network since generated examples are in nondeterministic order,
    #   subtly affecting network parameters after training
    session = Session.Session([], [])
    session.Load('nets/' + net_name, data_prefix='tf_profile')

    if not train_option:
        session.net.summary()

    print('\n-----------------------')
    print('  q_learn.generateExamples')
    print('-----------------------\n')

    cProfile.run('do_generate_examples([session, p, generate_times])')
