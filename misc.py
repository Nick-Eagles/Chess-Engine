import numpy as np

#   A naive implementation of finding the indices of the largest N values in
#   vec. Recursively computes the argmax (not intended for large vectors/ large
#   N).
def topN(vec, N):
    assert N > 0, N
    realN = min(N, vec.shape[0])
    
    #   Handle a trivial case
    if realN == 1:
        return [int(np.argmax(vec))]

    vecCopy = vec.copy()
    minVal = min(vec)

    inds = []
    for i in range(realN):
        temp = int(np.argmax(vecCopy))
        inds.append(temp)
        vecCopy[temp] = minVal - 1

    assert len(inds) == realN, inds
    return inds

def match(needle, stack, is_float=False, tol=0.00001):
    found = False
    i = -1
    while not found and i < len(stack):
        i += 1
        if is_float:
            found = abs(needle - stack[i]) < tol
        else:
            found = needle == stack[i]

    return i
