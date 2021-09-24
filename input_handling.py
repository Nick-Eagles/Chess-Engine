import policy

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
#   of content is a string of the form "[variableName] [value]". Assumes the
#   config file is legitimate and properly specified.
def readConfig(f_name='config.txt'):
    #   Read the entire file into memory
    with open(f_name, 'r') as f:
        line_list = f.readlines()

    #   Filter out empty lines and lines to be treated as comments
    line_list = [x for x in line_list if ("#" not in x) and (x != "\n")]

    #   Add each key-value pair to a dictionary
    p = {}
    for x in line_list:
        key, val = x.split(' ')
        if '"' in val:
            p[key] = val[1:-2]
        elif "." in val:
            p[key] = float(val)
        else:
            p[key] = int(val)

    validateP(p)
    p['evalFun'] = getattr(policy, p['evalFun'])
    
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
        p['updatePeriod'] = getUserInput(
            messDef, messOnErr, 'int', cond, explan
        )

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

def validateP(p):
    #   General settings
    assert p['baseBreadth'] > 0, p['baseBreadth']
    assert p['epsGreedy'] >= 0 and p['epsGreedy'] <= 1, p['epsGreedy']
    assert p['mateReward'] > 0, "Mate reward must be positive to be " + \
           "consistent with rewards for captures"
    
    assert p['gamma'] > 0, p['gamma']
    if p['gamma'] > 1 and p['mode'] >= 1:
        print("Warning: you are asking for reward to increase with distance " +\
              "(gamma = " + str(p['gamma']) + ")!")

    #   Tree traversal parameters
    if 'gamma_exec' in p:
        assert p['gamma_exec'] > 0, p['gamma_exec']
        if p['gamma_exec'] > 1 and p['mode'] >= 1:
            print("Warning: you are asking for reward to increase with " + \
                  "distance (gamma_exec = " + str(p['gamma_exec']) + ")!")

        assert p['breadth'] >= 1, p['breadth']
        assert p['depth'] >= 1, p['depth']
        assert p['epsSearch'] >= 0 and p['epsSearch'] <= 1, p['epsSearch']

        assert p['minCertainty'] < 1, p['minCertainty']
        if p['minCertainty'] < 0 and p['mode'] >= 1:
            print("Warning: 'minCertainty' is negative- is this intended?")
            
        assert p['policyFun'] == "sampleMovesEG" \
               or p['policyFun'] == "sampleMovesSoft", p['policyFun']
        
    #   Network/ training
    if 'memDecay' in p:
        assert p['memDecay'] >= 0 and p['memDecay'] < 1, p['memDecay']
        if p['memDecay'] >= 0.125 and p['mode'] >= 1:
            print("Warning: setting 'memDecay' to 0.125 or above causes " + \
                  "some positions to be overrepresented in training data.")

        assert p['weightDec'] >= 0, p['weightDec']
        assert p['nu'] > 0, p['nu']
        
        assert p['mom'] >= 0 and p['mom'] <= 1, p['mom']
        if p['mom'] == 1 and p['mode'] >= 1:
            print("Warning: using undamped momentum in SGD.")

        assert p['batchSize'] > 1, p['batchSize']
        assert p['epochs'] >= 1, p['epochs']
        assert p['fracValidation'] >= 0 and p['fracValidation'] < 1, \
               p['fracValidation']
        assert p['fracFromFile'] >= 0 and p['fracFromFile'] < 1, \
               p['fracFromFile']

        assert p['popPersist'] >= 0 and p['popPersist'] < 1, p['popPersist']
        assert p['policyWeight'] >= 0 and p['policyWeight'] <= 1, \
               p['policyWeight']
        if p['policyWeight'] == 0 and p['mode'] >= 1:
            print("Warning: policy will not be optimized if using a policy-" + \
                  "value network (policyWeight = 0).")
        elif p['policyWeight'] == 1 and p['mode'] >= 1:
            print("Warning: value will not be optimized if using a policy-" + \
                  "value network (policyWeight = 1).")

    #   Q learning
    if 'maxSteps' in p:
        assert p['maxSteps'] > p['rDepthMin'], "'maxSteps' must exceed " + \
               "'rDepthMin' in order to produce training examples"
        assert p['rDepthMin'] >= 1, p['rDepthMin']
        assert p['persist'] >= 0 and p['persist'] < 1, p['persist']
