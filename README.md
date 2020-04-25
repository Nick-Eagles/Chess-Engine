# Self-contained Chess Engine: Deep Q-learning Agent #

## Configuration ##

In general, this program was developed with the idea that aspects of the data generation or training process should be customizable in a single configuration file. For any decision or variable in the program which must hold a particular value, and where its optimal value is ambiguous or changes variably with time, a variable exists in the config file.
The config `config.txt` is broken into sections beginning with a comment description (starting with '#') and ending in either an *EOF* or a new section.

### Full description of config variables ###

#### General variables ####

`mode`: an integer controlling how verbose the program's output is, and to some extent how thorough the requested tasks are performed.
    0: (Quiet) Print only the most significant output. Least verbose and highest performance
    1: (Normal) Print significant output and some additional info
    2: (Verbose/ Debug) Print all information about the program's execution, including debug info, and perform any tasks helpful for debugging.
`baseBreadth`: positive integer dependent on the training mode. For deep Q-learning, this is the number of data generation tasks, run asynchronously in parallel on the available CPUs. Each task involves the agent playing itself in chess for a total of "maxSteps" moves. For tree-search data generation, this is the number of root nodes from which the search begins (also performed asynchronously in parallel).
`epsGreedy`: float in [0,1]. For deep Q-learning mode with "policy" set to "sampleMovesEG", this is the choice of epsilon as the agent performs moves under an epsilon-greedy policy. This also applies for tree-search mode within the tree search itself, but not in deciding the root nodes.
`mateReward`: positive float (or integer). See "definition of 'reward'" for advice on choosing a particular value. This is the reward received by an agent for performing a checkmate. A value of 3-5 is probably a reasonable choice given the rewards of various captures- for reference, capturing a pawn as the first capture of the game has a reward of just above 0.02.
`gamma`: float in (0,1]. As per convention, the decay in reward received per action (half-move).

#### Tree traversal ####

`gamma_exec`: float in [0, 1]. The decay in reward anticipated for all subsequent actions, relative to the first action from a state. This parameter exists to allow the network to learn a different reward-decay-rate than it acts with (see 'mathematical choices' section).
`epsilon`: float in [0,1]. This is used in intializing games (from which root nodes are chosen). The initialized games are constructed by the agent playing itself with an epsilon-greedy strategy, which this parameter specifies.
`breadth`: positive integer. The number of branches for any non-leaf node in the tree search. In this case, the number of moves searched from any (non-leaf) position in the tree search.
`depth`: integer >= 1. The number of subsequent half-moves considered for a tree search performed from a given state (board position). A depth of at least 2 is recommended due to overhead in parallelizing the search, which occurs at all depths.
`curiosity`: non-negative float. When policy "sampleMovesSoft" is used, the probability of each legal move being selected is the softmax of the reward for that move plus the expected reward from the resulting position. `curiosity` is the inverse "temperature" for that softmax function. Thus a value of 0 makes all moves equally probable; large values approach a purely greedy policy.
`epsSearch`: float in [0,1]. The probability that all moves from a given node are selected randomly, when performing the tree search (note that the decision to either randomly or informatively choose is made per potential move). This does not impact which move sequence is ultimately chosen from the completed tree search; the parameter `epsGreedy` exists for this purpose.
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
`fracValidation`: float in [0,1). The decimal fraction of generated data to use as held-out test examples. Note that old test examples are recycled for use as training data (but of course not vice versa).
`fracFromFile`: float in [0,1). The decimal fraction of all data which comes from a *.csv* file of checkmate positions (each episode). A fairly small value is recommended, especially because no check is performed to ensure a given checkmate position will not be placed in the training/validation buffers more than once.
`popPersist`: float in [0,1). The scalar to the previous population statistics (mean and variance) when computing the next values from the previous and most recent values. Statistics are updated using an exponential moving average (see 'mathematical choices' section).
`costLimit`: integer in [`batchSize`, \infty). The number of examples used when estimating average loss across the entire buffer (for computational speed).
`batchNormScale`: float scalar to the partial derivates of loss with respect to gamma and beta (after multiplying by learning rate as usual). An experimental parameter originally motiviated by the idea that optimizing normally for beta and gamma could cause 'overfitting'/optimization difficulties.

#### Q-learning related ####

`maxSteps`: integer > `rDepthMin`. The number of actions to perform from the initial game state on a single thread of data generation (see 'mathematical choices' section).
`rDepthMin`: non-negative integer. The minimum number of actions a game must contain to include at least one state from the game as training data (see 'mathematical choices' section).
`persist`: float in [0,1). Similarly to `popPersist`, the scalar to the previous network 'certainty' as part of an exponentially moving average. Larger values thus produce a more stable and slowly moving estimate of the network certainty, a metric in [-1,1] intended to measure how closely the network predicts future reward representative of the optimal policy (see 'mathematical choices' section).
