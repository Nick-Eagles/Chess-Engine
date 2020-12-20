import sys
sys.path.append('./')
import numpy as np

import misc

################################################################################
#   topN
################################################################################

print('Testing "topN" (typical case 1)...')
a = np.array([1.4, 0, 0, 3, -1, 2, 0])
expected = [3, 5, 0, 1, 2]
actual = misc.topN(a, 5)
assert len(expected) == len(actual), len(actual)

misc.expect_equal(all([expected[i] == actual[i] for i in range(5)]), True)

print('Testing "topN" (typical case 2)...')
a = np.array([-1, -1, 3, 2.15, -0.1])
expected = [2, 3, 4, 0]
actual = misc.topN(a, 4)
assert len(expected) == len(actual), len(actual)

misc.expect_equal(all([expected[i] == actual[i] for i in range(4)]), True)

print('Testing "topN" (typical case 3)...')
a = np.array([0.1, -0.3, 4, -0.3])
expected = [2, 0, 1, 3]
actual = misc.topN(a, 4)
assert len(expected) == len(actual), len(actual)

misc.expect_equal(all([expected[i] == actual[i] for i in range(4)]), True)

print('Testing "topN" (N = 1)...')
a = np.array([0.7, -0.2, 3.5, -0.2])
expected = [2]
actual = misc.topN(a, 1)
assert len(expected) == len(actual), len(actual)

misc.expect_equal(expected[0], actual[0])

################################################################################
#   is_unique
################################################################################

print('Testing "is_unique" (False case)...')
a = np.array([1.5, 4, 2.6, 1.5, 7, -3.98])
misc.expect_equal(misc.is_unique(a), False)

print('Testing "is_unique" (True case)...')
a = np.array([-3.6, -2, 1.77, 0, -9])
misc.expect_equal(misc.is_unique(a), True)

################################################################################
#   match
################################################################################

print('Testing "match" (int and numpy array 1)...')
a = np.array([1, -1, 1.2, 3.8, 7, -4])
misc.expect_equal(misc.match(-1, a), 1)

print('Testing "match" (int and numpy array 2)...')
a = np.array([1.2, -1, 1.2, 3.8, 7, -4])
misc.expect_equal(misc.match(7, a), 4)

print('Testing "match" (duplicated float and list)...')
a = [1.25, -1, 3.8, 3.8, 7, -4]
misc.expect_equal(misc.match(3.800001, a, is_float=True), 2)

