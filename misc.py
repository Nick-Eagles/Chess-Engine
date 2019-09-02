import numpy as np
import random
import math

# Given a numpy array "cdf", a cumulative discrete probability distribution function,
# returns n indices, which are random selections proportional to the original probability
# density function (ie. for a cdf [0.1 0.7 1], index 0 would be selected w/ P=0.1, index 1
# w/ P=0.6, and index 2 w/ P=0.3). "exclusive" decides whether an index can be selected
# > 1 time.
def sampleCDF(cdf, n=1, exclusive=True, tol=1e-12):
    assert abs(cdf[-1] - 1) < tol, "Illegitimate CDF received as an argument: probabilities don't add to 1"
    assert not exclusive or n <= len(cdf), "Pigeonhole principle: more ints requested than discrete 'slots'"
    if cdf.shape[0] == n:
        return list(range(n))
    
    cdf = cdf.copy()

    indices = []
    while len(indices) < n:
        r = np.random.uniform()
        i = 0
        while i < (cdf.shape[0])-1 and r > cdf[i]:
            i += 1
        indices.append(i)
        if exclusive and len(indices) < n:
            #   Effectively set the chosen probability to 0 for the future
            if i == 0:
                lostProb = cdf[0]
            else:
                lostProb = cdf[i] - cdf[i-1]
            for j in range(i, len(cdf)):
                cdf[j] -= lostProb
                
            #   Rescale the cdf to compensate for the removed prob
            assert abs(cdf[-1] - 0) > tol, cdf
            cdf /= cdf[-1]

    return indices

#   Given a list "data", randomly partition the list such that the first subset contains a
#   fraction [frac] of the original number of elements. If [both], return the remaining
#   portion as part 2 (otherwise part 2 = []). Return the tuple (part1, part2).
def divvy(data, frac, both=True):
    permute = list(range(len(data)))
    random.shuffle(permute)
    cutoff = int(len(data) * frac)
    part1 = [data[permute[i]] for i in range(cutoff)]
    if both:
        part2 = [data[permute[i+cutoff]] for i in range(len(data) - cutoff)]
    else:
        part2 = []

    return (part1, part2)
