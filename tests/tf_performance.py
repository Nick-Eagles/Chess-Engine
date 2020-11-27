import timeit
import numpy as np
import tensorflow as tf
from scipy.special import logit, expit

size = (100, 300)

#   Test how to most quickly produce a tf.Tensor
def generate_element():
    return float(np.random.normal())

def generate_element_np(size=1):
    return np.random.normal(size)

'''
start = timeit.default_timer()
a = []
for i in range(size[0]):
    a.append([])
    for j in range(size[1]):
        a[i].append(generate_element())

b = tf.constant(a)
diff = timeit.default_timer() - start
print(diff)

start = timeit.default_timer()
a = np.zeros(size)
for i in range(size[0]):
    for j in range(size[1]):
        a[i][j] = generate_element()
    
b = tf.constant(a)
diff = timeit.default_timer() - start
print(diff)

start = timeit.default_timer()
a = np.zeros(size)
for i in range(size[0]):
    for j in range(size[1]):
        a[i][j] = generate_element_np()
    
b = tf.constant(a)
diff = timeit.default_timer() - start
print(diff)

start = timeit.default_timer()
c = generate_element(size)
a = tf.math.sigmoid(c)
diff = timeit.default_timer() - start
print(diff)
'''

start = timeit.default_timer()
for i in range(size[0] * size[1]):
    c = generate_element()
    a = tf.math.sigmoid(tf.constant(c, shape=[1,1]))
diff = timeit.default_timer() - start
print(diff)

start = timeit.default_timer()
for i in range(size[0] * size[1]):
    c = expit(generate_element())
    a = tf.constant(c, shape=[1,1])
diff = timeit.default_timer() - start
print(diff)

'''
start = timeit.default_timer()
for i in range(size[0] * size[1]):
    c = tf.constant(generate_element(), shape=[1,1])
    a = float(logit(c))

diff = timeit.default_timer() - start
print(diff)

start = timeit.default_timer()
for i in range(size[0] * size[1]):
    c = tf.constant(generate_element(), shape=[1,1])
    a = float(tf.math.log(c / (1 - c)))

diff = timeit.default_timer() - start
print(diff)
'''

