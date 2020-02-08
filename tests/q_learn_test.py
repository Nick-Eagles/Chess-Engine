import sys
import numpy as np
from multiprocessing import Pool
sys.path.append('../')

import Network
import q_learn

net = Network.load('../nets/6deep')
p = {}
p['maxSteps'] = 50
p['epsGreedy'] = 0.3
p['mateReward'] = 3
p['rDepth'] = 2
p['gamma'] = 0.4

#print("Performing 2 instances of async_q_learn...")
#data1 = q_learn.async_q_learn(net)
#data2 = q_learn.async_q_learn(net)

#if any([np.array_equal(data1[i][0], data2[i][0]) for i in range(10, 20)]):
    #print("Failed test- some positions were equal after 10 moves of play.")
#else:
    #print("Passed test- all positions from different instances were different after 10 moves.")


print("Performing generateExamples 2 times in parallel...")
pool = Pool()
temp_data = pool.starmap(q_learn.generateExamples, [[net, p] for i in range(2)])
print(temp_data[0][5][1])
print(temp_data[1][5][1])
data1 = temp_data[0]
data2 = temp_data[1]

if any([np.array_equal(data1[i][0], data2[i][0]) for i in range(10, 20)]):
    print("Failed test- some positions were equal after 10 moves of play.")
else:
    print("Passed test- all positions from different instances were different after 10 moves.")
