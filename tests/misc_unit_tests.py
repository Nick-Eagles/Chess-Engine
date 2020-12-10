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
