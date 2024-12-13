from pyhere import here

import policy

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
