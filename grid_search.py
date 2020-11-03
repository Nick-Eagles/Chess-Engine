#   An iterative grid search. Given a range of several hyperparameters and a
#   resolution, returns the optimal set of HPs. "Optimal" is defined as the
#   largest drop in held-out validation loss given the constraint that training
#   loss drops by at least some specified percentage (this is intended to rule
#   out "coincidental" drops in validation loss)
#
#   Each iteration, the range of the search space decays by "decay", with the
#   new range centered at the previously best parameter value.

import Network
import input_handling
import main
import board_helper

import random

############################################
#   User-configurable variables
############################################

#   These are traversed through logarithmically
lr_range_orig = (0.000001, 10)
bs_range_orig = (100, 10000)
wd_range_orig = (0.00000001, 0.01)

#   This is traversed through linearly
mom_range_orig = (0, 0.99)

net_name = 'res'
resolution = 3
iterations = 3
decay = 0.5
max_examples = 50000

############################################
#   Functions
############################################

#   Return a smaller range geometrically centered at the previous optimum
def get_range_geom(decay, iter_num, orig_range, optimum):
    #   Geometrically half the difference between the new range max and min
    r = (orig_range[1] / orig_range[0])**(0.5 * decay**iter_num)

    #   The new range is restricted to be within the original range;
    #   compute a geometric offset so that the new range does not stretch
    #   outside
    if optimum / r < orig_range[0]:
        offset = orig_range[0] / (optimum / r)
    elif optimum * r > orig_range[1]:
        offset = orig_range[1] / (optimum * r)
    else:
        offset = 1

    #   Ensure new range falls within original range
    tol = 0.000001
    assert optimum / r * offset + tol > orig_range[0], optimum / r * offset - orig_range[0]
    assert optimum * r * offset - tol < orig_range[1], orig_range[1] - (optimum * r * offset)

    return (optimum / r * offset, optimum * r * offset)

#   Return a smaller range linearly centered at the previous optimum
def get_range_linear(decay, iter_num, orig_range, optimum):
    #   Half the difference between the new range max and min
    r = decay**iter_num * (orig_range[1] - orig_range[0]) / 2

    #   The new range is restricted to be within the original range;
    #   compute an offset so that the new range does not stretch outside
    if optimum - r < orig_range[0]:
        offset = orig_range[0] - (optimum - r)
    elif optimum + r > orig_range[1]:
        offset = orig_range[1] - (optimum + r)
    else:
        offset = 0

    #   Ensure new range falls within original range
    tol = 0.000001
    assert optimum - r + offset + tol > orig_range[0], optimum - r + offset - orig_range[0]
    assert optimum + r + offset - tol < orig_range[1], orig_range[1] - (optimum + r + offset)

    return (optimum - r + offset, optimum + r + offset)

def get_grid_value_geom(param_range, res, i):
    return param_range[0] * (param_range[1] / param_range[0])**(i / (res - 1))

def get_grid_value_linear(param_range, res, i):
    return param_range[0] + (param_range[1] - param_range[0]) * (i / (res - 1))

############################################
#   Main
############################################

lr_range = lr_range_orig
bs_range = bs_range_orig
wd_range = wd_range_orig
mom_range = mom_range_orig

lr_diff = lr_range[1] - lr_range[0]
bs_diff = bs_range[1] - bs_range[0]
wd_diff = wd_range[1] - wd_range[0]
mom_diff = mom_range[1] - mom_range[0]

print('Loading network and data...')
net, tBuffer, vBuffer = Network.load('nets/' + net_name + '.pkl')
p = input_handling.readConfig(2)

#   Ensure there is enough validation data to compute loss at the largest batch
#   size; limit number of examples in tData to maximal amount specified
bigData = main.collapseBuffer(tBuffer) + main.collapseBuffer(vBuffer)
random.shuffle(bigData)

vData = bigData[:bs_range[1]]
tData = bigData[bs_range[1]: max_examples+bs_range[1]]

#   Ensure costs are computed at maximal batch size
if p['costLimit'] < bs_range[1]:
    p['costLimit'] = bs_range[1]

print('len(tData):', len(tData))
print('len(vData):', len(vData))
print('Done.')

counter = 0

for iter_num in range(iterations):
    print('Starting iteration ', iter_num, '...', sep='')
    
    #   Compute new smaller search range
    if iter_num > 0:
        lr_range = get_range_geom(decay, iter_num, lr_range_orig, lr_best)
        bs_range = get_range_geom(decay, iter_num, bs_range_orig, bs_best)
        wd_range = get_range_geom(decay, iter_num, wd_range_orig, wd_best)
        mom_range = get_range_linear(decay, iter_num, mom_range_orig, mom_best)

        print('New ranges:')
        print('lr: ', lr_range)
        print('bs: ', bs_range)
        print('wd: ', wd_range)
        print('mom:', mom_range)

    #   Reset some variables
    vLossMaxDiff = -1
    lr_best = -1
    bs_best = -1
    wd_best = -1
    mom_best = -1
    
    #   Loop through grid
    for i in range(resolution):
        lr = get_grid_value_geom(lr_range, resolution, i)
        for j in range(resolution):
            bs = int(get_grid_value_geom(bs_range, resolution, j))
            for k in range(resolution):
                wd = get_grid_value_geom(wd_range, resolution, k)
                for l in range(resolution):
                    mom = get_grid_value_linear(mom_range, resolution, l)

                    #   Make a copy of 'net' and 'p' appropriate for this iteration
                    net_temp = net.copy()
                    p_temp = p.copy()
                    p['nu'] = lr
                    p['batchSize'] = bs
                    p['weightDec'] = wd
                    p['mom'] = mom

                    net_temp.train(tData, vData, p_temp)
                    
                    #vLossDiff = net_temp.loss.vLoss[-1 - p['epochs']] - net_temp.loss.vLoss[-1]
                    vLossDiff = net.loss.vLoss[-1] - net_temp.loss.vLoss[-1]
                    
                    tLossDiff = net_temp.loss.tLoss[-1 - p['epochs']] - net_temp.loss.tLoss[-1]

                    if vLossDiff > vLossMaxDiff:
                        print('Exceeded best drop in loss with parameters:')
                        print(lr, bs, wd, mom)
                        print('Drop in loss was:', vLossDiff)
                        print('Previous max was:', vLossMaxDiff)
                        
                        lr_best = lr
                        bs_best = bs
                        wd_best = wd
                        mom_best = mom

                        vLossMaxDiff = vLossDiff

                    #   Print progress every 10%
                    perc = int(100 * counter / (iterations * resolution**4))
                    if perc % 10 == 0:
                        print(perc, '% done.', sep='')
                        
                    counter += 1

print('\nBest set of hyperparameters:')
print('Learning rate:', lr_best)
print('Batch size:   ', bs_best)
print('Weight decay: ', wd_best)
print('Momentum:     ', mom_best)

print('\nOriginal validation loss:', net.loss.vLoss[-1])
print('Best drop in loss:       ', vLossMaxDiff)
