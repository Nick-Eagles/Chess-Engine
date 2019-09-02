import numpy as np
from scipy.special import expit

N = 1000
y = 0.5
cost = 0
outputs = expit(np.random.randn(N,1))
for a in outputs:
    cost -= y * np.log(a) + (1 - y) * np.log(1 - a)
print(cost/N)



