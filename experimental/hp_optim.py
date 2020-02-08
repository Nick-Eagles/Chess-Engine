import numpy as np

import Network
import input_handling

def optim(baseNet):
    #########################################################
    #   Set up parameters and examples to use
    #########################################################
    
    p = {'useConfig': True}
    #   Literally assign variables as specified by the config
    assignments = input_handling.readConfig(0)
    for i in assignments:
        var = i[:i.index(" ")]
        val = i[i.index(" ")+1:]
        if "." in val:
            p[var] = float(val)
        else:
            p[var] = int(val)
    assignments = input_handling.readConfig(2)
    for i in assignments:
        var = i[:i.index(" ")]
        val = i[i.index(" ")+1:]
        if "." in val:
            p[var] = float(val)
        else:
            p[var] = int(val)

    messDef = "Enter the name of a data file to use: "
    cond = 'True'
    messOnErr = "kodjasjd"
    filename = 'data/' + input_handling.getUserInput(messDef, messOnErr, 'str', cond)

    data = read_csv(filename)
    permute = list(range(len(data)))
    cutoff = int(len(data) / (1 + p['fracValidation']))
    tExamps = [data[permute[i]] for i in range(cutoff)]
    vExamps = [data[permute[i+cutoff]] for i in range(len(data)-cutoff)]

    temp = baseNet.copy()
    testNet = Network.Network(temp[0], temp[1], temp[2], temp[3])



def read_csv(filename):
    flatData = []
    #   Open and read file
    try:
        gameFile = open(filename, 'r')
        with gameFile:
            reader = csv.reader(gameFile, delimiter=',', quotechar=',', quoting=csv.QUOTE_MINIMAL)

            for line in reader:
                flatData.append(line)
        print("Read the file without errors.")
    except:
        print("Problem opening or reading file.")
    finally:
        gameFile.close()

    examples = []
    for row in range(len(flatData)):
        intRow = [int(float(i)) for i in flatData[row]]
        r = intRow.pop()
        inArr = np.array(intRow).reshape(-1,1)
        examples.append((inArr, r))

    return examples
