#   Unit tests for functions in buffer.py

import sys
sys.path.append('./')

import buffer
import Session
import input_handling
import misc

net_name = 'tf_profile'

################################################################################
#   filter
################################################################################

session = Session.Session([], [])
session.Load('nets/' + net_name, data_prefix='tf_profile')

p = input_handling.readConfig()
p['memDecay'] = 0.095

orig_lens_t = [len(x) for x in session.tBuffer]
orig_lens_v = [len(x) for x in session.vBuffer]
print("Original lengths:")
print(orig_lens_t)
print(orig_lens_v)

buffer.filter(session.tBuffer, session.vBuffer, p)

new_lens_t = [len(x) for x in session.tBuffer]
new_lens_v = [len(x) for x in session.vBuffer]
print("New lengths:")
print(new_lens_t)
print(new_lens_v)

#   Generally ensure valid structure
buffer.verify(session.tBuffer, p)
buffer.verify(session.vBuffer, p)

#   Accept deviation up to 1% of the original buffer's sizes (due to
#   'quantization')
print("Confirming validation buffers are of correct length after filtering " + \
      "(2 tests)...")
misc.expect_equal(sum(orig_lens_v[:3]) * (1 - p['memDecay']),
                  sum(new_lens_v[:3]),
                  is_float = True,
                  tol = 0.01 * sum(orig_lens_v[:3]))

misc.expect_equal(orig_lens_v[3] * (1 - p['memDecay']),
                  new_lens_v[3],
                  is_float = True,
                  tol = 0.02 * orig_lens_v[3])

print("Confirming training buffers are of correct length after filtering " + \
      "(2 tests)...")
expected_new_len_sum = sum(orig_lens_t[:3]) * (1 - p['memDecay']) + \
                       sum(orig_lens_v[:3]) * p['memDecay']
misc.expect_equal(expected_new_len_sum,
                  sum(new_lens_t[:3]),
                  is_float = True,
                  tol = 0.01 * sum(orig_lens_t[:3]))

misc.expect_equal(orig_lens_t[3] * (1 - p['memDecay']),
                  new_lens_t[3],
                  is_float = True,
                  tol = 0.01 * orig_lens_t[3])


