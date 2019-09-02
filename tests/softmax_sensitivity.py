#import Game
import network_helper
import numpy as np

#temp = network_helper.readGames('data/checkmates.csv')
#print(temp[0])
#data = network_helper.decompressGames(temp)
#print(data[0])

#g = Game.Game()
#print(g.updateValues())
#print(g.wValue)
#print(g.bValue)

r = np.array([-1, -0.5, 0, 0.5, 1])
temp = np.exp(r)
r = temp / np.sum(temp)
print(r)
r = np.array([-0.5, -0.1, 0, 0.1, 0.5])
temp = np.exp(r)
r = temp / np.sum(temp)
print(r)
r = np.array([-0.04, -0.02, 0, 0.02, 0.09])
temp = np.exp(10*r)
r = temp / np.sum(temp)
print(r)
