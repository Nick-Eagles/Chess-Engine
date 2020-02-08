# ---------------------------------------------------------------------
#
#   Manages user input in a smarter way, preventing the program from
#   crashing for dumb reasons. Note that input is assumed to be benign
#   (eval() is used in a way that could do harm outside the program if
#   you explicitly intended to)
#
# ---------------------------------------------------------------------

#   messDef: main/ default string to ask the user for input
#   messOnErr: message to warn user about invalid input not related
#       to the input's type
#   inType: string containing expected variable type of user input
#   cond: a string containing a condition to be evaluated to determine
#       if input is valid: Valid examples are "var == 4" or "var[-4] != '.csv'"
#   explan: an optional string explaining what the input will signify to
#       to the program
#   auxVars: a tuple containing an arbitrary number of variables to
#       include in "cond". Suppose you want a condition to involve the user
#       input and other variables y and z. Then auxVars = (y,z) and
#       you might have cond = 'var > 0 and var+auxVars[0]+auxVars[1] < 10'
#
#   returns the validated user input
def getUserInput(messDef, messOnErr, inType, cond, explan='', auxVars=()):
    valid = False
    finalIn = ''
    while not valid:
        if explan != '':
            print("Enter '-0' for an explanation.")
        rawIn = input(messDef)
        if rawIn == '-0':
            print(explan)
        elif len(rawIn) == 0:
            print("Please check your input.")
        else:
            if inType != "str":
                #   Try to coerce input (a string) to desired type
                try:
                    coercedIn = eval(inType + "(" + rawIn + ")")
                    var = coercedIn
                    if eval(cond):
                        finalIn = coercedIn
                        valid = True
                    else:
                        print(messOnErr)
                except:
                    print("Please check your input.")
                    pass
            else:
                var = rawIn
                if eval(cond):
                    finalIn = rawIn
                    valid = True
                else:
                    print(messOnErr)
    return finalIn

#   Read a config file where sections are delimited with "#", and each line
#   of content is a string of the form "[variableName] [value]". Fed to eval(), so
#   assumes the config file is legitimate and properly specified.
def readConfig(mode=0, fName="config.txt"):
    #   Read in entire file as list of strings, and record lines that
    #   begin with "#"; these are section markers
    lineList = []
    sectDivs = []
    f = open(fName,"r")
    i = 0
    for line in f:
        if line[0] == "#":
            sectDivs.append(i)
        lineList.append(line)
        i += 1
    sectDivs.append(i)  # the file end
    f.close()

    #   Each line contains the variable name and its value; store the lines
    #   from the correct section (and general section) in a dictionary and return
    p = {}
    if mode == 0:
        linesToRead = lineList[1:sectDivs[1]]
    else:
        linesToRead = lineList[1:sectDivs[1]] + lineList[sectDivs[mode]+1:sectDivs[mode+1]]
    for i in linesToRead:
        var = i[:i.index(" ")]
        val = i[i.index(" ")+1:]
        if '"' in val:
            p[var] = val[1:-2]
        elif "." in val:
            p[var] = float(val)
        else:
            p[var] = int(val)

    return p

#   A function storing the lengthy user-input queries for different sections of the program.
#   Returns p, the dictionary containing the relevant variables in the correct mode, or
#   index of the section of the config file delimited by "#"
def getP(mode):
    if mode == 0:
        #   bufferSize
        messDef = "Size of training example buffer (cache of old training examples)? "
        cond = 'var >= 0'
        messOnErr = "Not a valid choice."
        p = {'bufferSize': getUserInput(messDef, messOnErr, 'int', cond)}

        #   bufferFrac
        messDef = "Use what decimal fraction of the full buffer when training? "
        cond = 'var > 0 and var <= 1'
        messOnErr = "Not a valid choice."
        p['bufferFrac'] = getUserInput(messDef, messOnErr, 'float', cond)

        #   memoryDec
        messDef = "Rate of replacement of old training examples? "
        explan = "This is a real number in (0, 1] specifying the maximum decimal fraction of examples\n" \
                 + "in the training buffer to replace after each tree traversal. The training buffer\n" \
                 + "is exponentially distributed with respect to age of the examples, and this number\n" \
                 + "is the (negative) coefficient of that distribution, provided traversal is extensive enough."
        cond = 'var > 0 and var <= 1'
        messOnErr = "Not a valid choice."
        p['memoryDec'] = getUserInput(messDef, messOnErr, 'float', cond, explan)
        
        #   weightDec
        messDef = "Value for L2 regularization/ weight decay: "
        cond = 'var >= 0'
        messOnErr = "I doubt you want weight growth"
        p['weightDec'] = getUserInput(messDef, messOnErr, 'float', cond)
        
        #   nu
        messDef = "Learning rate for the network while training: "
        cond = 'var > 0'
        messOnErr = "Not a valid learning rate"
        p['nu'] = getUserInput(messDef, messOnErr, 'float', cond)

        #   batchSize
        messDef = "Enter the number of games per batch: "
        cond = 'var > 0'
        messOnErr = "Not a valid number"
        p['batchSize'] = getUserInput(messDef, messOnErr, 'int', cond)

        #   epochs
        messDef = "Enter the number of epochs (after each traversal) to train on: "
        cond = 'var > 0'
        messOnErr = "Not a valid input"
        p['epochs'] = getUserInput(messDef, messOnErr, 'int', cond)

        #   updatePeriod
        messDef = "Update period for the network? "
        explan = "To stabilize the reward expectation for positions in games during training, a 2nd network\n" \
                 + "trains on the data that the first network leads the engine to explore. The update period\n" \
                 + "specifies how many traversals occur before the first network is overwritten by the 2nd."
        cond = 'var > 0'
        messOnErr = "Not a valid input."
        p['updatePeriod'] = getUserInput(messDef, messOnErr, 'int', cond, explan)

        #   fracValidation
        messDef = "Enter the decimal fraction of examples to use as validation data: "
        cond = 'var >= 0 and var < 1'
        messOnErr = "Not a valid value."
        p['fracValidation'] = getUserInput(messDef, messOnErr, 'float', cond)
        
    elif mode == 1:    
        messDef = "Value for alpha (Enter -0 for a description of the parameter alpha)? "
        explan = "Alpha is a hyperparameter that specifies the decay factor, in reward\n" \
                    + "that the engine receives, for each move that the reward is delayed.\n" \
                    + "Choose a value for alpha in (0,1] (1 is no decay)"
        cond = 'var > 0 and var <= 1'
        messOnErr = "Not a valid value for alpha"
        p['alpha'] = getUserInput(messDef, messOnErr, 'float', cond, explan)

        messDef = "Value for epsilon (Enter -0 for a description of the parameter epsilon)? "
        explan = "Epsilon is a noise variable, pushing the engine to decide upon moves it evaluates\n" \
                    + "as suboptimal (to explore more novel positions). A value of 0 is no noise, while\n" \
                    + "A value of 1 produces fully random move choices; valid values are in [0,1]."
        cond = 'var >= 0 and var <= 1'
        messOnErr = "Not a valid value for epsilon"
        p['epsilon'] = getUserInput(messDef, messOnErr, 'float', cond, explan)

        messDef = "Breadth of the tree? "
        cond = 'var > 0'
        messOnErr = "Not a valid value for breadth"
        p['breadth'] = getUserInput(messDef, messOnErr, 'int', cond)

        messDef = "Depth limit for nodes to consider as valid training examples? "
        cond = 'var > 0 and var < 50'
        messOnErr = "Not a valid value for depth."
        p['tDepth'] = getUserInput(messDef, messOnErr, 'int', cond)

        messDef = "Reward depth to consider? "
        cond = 'var >= 0'
        messOnErr = "Not a valid value."
        p['rDepth'] = getUserInput(messDef, messOnErr, 'int', cond)

        messDef = "Min distance from any node in tree to game start or end? "
        cond = 'var >= 0'
        messOnErr = "Not a valid value."
        p['gBuffer'] = getUserInput(messDef, messOnErr, 'int', cond)

    return p
        
