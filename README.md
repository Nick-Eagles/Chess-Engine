# Self-contained Chess Engine: Deep Q-learning Agent #

## Configuration ##

In general, this program was developed with the idea that aspects of the data generation or training process should be customizable in a single configuration file. For any decision or variable in the program which must hold a particular value, and where its optimal value is ambiguous or changes variably with time, a variable exists in the config file.
The config `config.txt` is broken into sections beginning with a comment description (starting with '#') and ending in either an EOF or a new section.

### Full description of config variables ###

#### General variables ####

`mode`: an integer controlling how verbose the program's output is, and to some extent how thorough the requested tasks are performed.
    0: (Quiet) Print only the most significant output. Least verbose and highest performance
    1: (Normal) Print significant output and some additional info
    2: (Verbose/ Debug) Print all information about the program's execution, including debug info, and perform any tasks helpful for debugging.
`baseBreadth`: positive integer dependent on the training mode. For deep Q-learning, this is the number of data generation tasks, run asynchronously in parallel on the available CPUs. Each task involves the agent playing itself in chess for a total of "maxSteps" moves. For tree-search data generation, this is the number of root nodes from which the search begins (also performed asynchronously in parallel).
`epsGreedy`: float in [0,1]. For deep Q-learning mode with "policy" set to "sampleMovesEG", this is the choice of epsilon as the agent performs moves under an epsilon-greedy policy. This also applies for tree-search mode within the tree search itself, but not in deciding the root nodes.
`mateReward`: positive float (or integer). See "definition of 'reward'" for advice on choosing a particular value. This is the reward received by an agent for performing a checkmate. A value of 3-5 is probably a reasonable choice given the rewards of various captures- for reference, capturing a pawn as the first capture of the game has a reward of just above 0.02.

#### Tree traversal ####

`alpha`: float in [0,1]. The expected probability that the agent will select the optimal move at an arbitrary position (excluding by chance). This ultimately determines how much weight future rewards have (see 'mathematical choices' section).
`epsilon`: float in [0,1]. This is used in intializing games (from which root nodes are chosen). The initialized games are constructed by the agent playing itself with an epsilon-greedy strategy, which this parameter specifies.
`breadth`: positive integer. The number of branches for any non-leaf node in the tree search. In this case, the number of moves searched from any (non-leaf) position in the tree search.
`tDepth`: positive integer. The number of 'layers' in the tree search to be used as training data. Specifying a value of 2 would mean to include the root nodes and the nodes they branch to. The deeper nodes function simply to provide more accurate information about the expected reward from the training positions.
`rDepth`: positive integer. The number of 'layers' in the tree search (beyond the deepest nodes included as training data) to search/explore. An rDepth of 2 would imply all training positions contain information about the reward at least 2 halfmoves following.
`gBuffer`: non-negative integer. The number of halfmoves at the beginning and end of a game to exclude as potential root nodes. This will likely be removed as an option.
`clarity`: non-negative float. In the tree search, each node collects the rewards received from all of its branches, and summarizes these in a single scalar reward value. This value is the expected value, where the "probabilities" of selecting the moves associated with each explored branch is the softmax function of their observed rewards. This parameter is the softmax coefficient (inverse "temperature"). A value of 0 considers all branches equally probable; large values approach considering only the most rewarding branch.
`curiosity`: non-negative float. When policy "sampleMovesSoft" is used, the probability of each legal move being selected is the softmax of the reward for that move plus the expected reward from the resulting position. `curiosity` is the inverse "temperature" for that softmax function. Thus a value of 0 makes all moves equally probable; large values approach a purely greedy policy.
`policy`: a string determining the function the agent uses to select a particular move from the ones legally available. Currently two choices are available:
    - "sampleMovesSoft": take the softmax of the expected cumulative rewards for each move, with exponential coefficient `curiosity`
    - "sampleMovesEG": with moves ranked by their expected cumulative rewards, select one move by an epsilon-greedy policy. Epsilon is `epsGreedy` in the "General variabes" section.

#### Network/ training-related ####

`memDecay`: float in [0,1]. The decimal fraction of examples to randomly remove from the data buffers after each set of training epochs. The idea is that the age of data in number of episodes should be exponentially distributed (with coefficient `memDecay`). This differs from the traditional fixed buffer of "the last N training examples" used for episodic memory.
`weightDec`: non-negative float. The L2 penalty coefficient used for optimization.
`nu`: positive float. The learning rate for optimization.
`mom`: float in [0,1]. The momentum coefficient (SGD with momentum is the optimization algorithm used by this program).
`batchSize`: positive integer. The number of training examples per batch (note that gradients are actually computed separately on chunks created by splitting each batch- enabling parallelization. The size of these chunks is `batchSize`/the number of CPUs you have available).
`epochs`: positive integer. The number of epochs of training to perform every `updatePeriod` episodes.
`updatePeriod`: positive integer. The number of episodes (which always involve data collection) to perform before training the network.