from main import fillBuffer

p = {'bufferSize': 10,
     'memoryDec': .3,
     'bufferFrac': .4,
     'traverseCount': 20,
     'fracValidation': 0.1,
     'updatePeriod': 5}

tBuffer = []
vBuffer = []
bS = p['bufferSize']
for i in range(p['traverseCount']):
    if i > 0 and i % p['updatePeriod'] == 0:
        print('------------------------------')
        print('Syncing slowNet to fastNet...')
        print('------------------------------')

        tBuffer = list()
        vBuffer = list()
        
    temp = [[i, 'a'], [i, 'a'], [i, 'a']]
    tData = temp

    permute = list(range(len(tData)))
    cutoff = int(len(tData) / (1 + p['fracValidation']))
    tSet = [tData[permute[i]] for i in range(cutoff)]
    vSet = [tData[permute[i+cutoff]] for i in range(len(tData)-cutoff)]

    tDiff = bS - len(tBuffer)
    tBuffer, tData = fillBuffer(tBuffer, tSet, p)
    vBuffer += vSet
    vData = vBuffer.copy()

    print("i =", i)
    print("tBuffer:", tBuffer)
    print("tData:", tData)
    print("------------------------------")

