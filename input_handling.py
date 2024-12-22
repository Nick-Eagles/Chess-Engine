from pyhere import here

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
def readConfig(f_name = here('config.txt')):
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

def validateP(p):
    #   General settings
    assert p['baseBreadth'] > 0, p['baseBreadth']
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
