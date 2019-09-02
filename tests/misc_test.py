import sys
import numpy as np
sys.path.append('../')

import misc

#   Check that CDF is sampled w/ expected distribution
n = 100
cdf = np.array([0.1, 0.2, 0.4, 0.9, 1])
actualFreqs = np.zeros(5)
for i in range(n):
    actualFreqs[misc.sampleCDF(cdf)[0]] += 1
actualCDF = np.cumsum(actualFreqs)
actualCDF /= actualCDF[-1]
print("sampled cdf: ", cdf)
print("empirical cdf: ", actualCDF)

#   Visual inspection of "exclusive"/ reasonability of output
for n in range(2, 5):
    print("n = ", n, ":", sep="")
    for i in range(3):
        print(misc.sampleCDF(cdf, n))
