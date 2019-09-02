Overall approach/ structure to learning:
	General high level outline:
		-the main method employed by this program is an online approach to deep reinforcement learning. The idea is that the data
	 	generation method relies on the NN at its current state: we traverse a subtree of potential moves, and the moves selected
	 	are a fraction decided by a linear combination of noise and the NN's evaluation of the expected reward of a given option.
	 	Thus the quality of the training data will be poor for a long time, and indeed the gradient expected to result from a
	 	batch of examples is only necessarily contextually appropriate for the NN's current state.
	Data generation:
		-two types of data are produced online. The first is designed to enrich the network's general tactical skill, and searches
	 	a subtree starting at some random point in a generated game, with probability skewed towards board positions allowing
	 	more potential captures next move. This is still generally fairly sparse in reward, motivating the second type of data
	 	generation. This is a similar subtree search, located specifically at the end of a generated game and constrained to at
	 	least contain the original win/loss (not draw). Thus average reward should usually be much higher.
	Learning algorithm/ procedure:
		-Learning occurs in sets of episodes, and 2 networks are involved. Each episode involves generating data via the first
		 method mentioned above (sparse reward), then training on it via SGD for some number of epochs. More specifically, the first
		 network, "slowNet", generates the data each episode, and "fastNet" is exclusively updated by SGD. A set of episodes
		 builds up a buffer of training examples, which has some max size, gets a certain fraction replaced when full, and samples
		 some fraction as a subset for each episode of training. A set of episodes ends with slowNet receiving parameters from fastNet.
		 This separation of networks follows the common technique in online reinforcement learning designed to stabilize the space of
		 training examples to produce a meaningful and clear gradient. A set of episodes begins with adding data produced by method 2
		 to the buffers. These examples are held in the buffer for the remainder of the set. Buffers are emptied at the end of each set.
How "reward" is implemented
	-the NN's job is to assign each board position a scalar output in (0, 1) (it has sigmoid output). The data generation method produces
	 estimates for the expected value of the rewards accumulated by subsequent moves in the game (with no explicit "horizon distance").
	 We define the "reward" of a move from position A to B as the 0.1 * ln(mB / mA), where mB is the ratio of material points for white to those
	 for black in position B (and similarly for A). An exception is for moves ending the game, with rewards for loss, draw, and win as
	 -1, 0, and 1, respectively, regardless of material. Note that positive reward always refers to good expectation for white, regardless of
	 if black is moving or not. We consider the expected value of reward for the next potential move to be the sum over m of
		softmax(f(Bm), f(B)) * R(A, Bm) * breadth / len(B)
	 where f is the NN inference for a potential position Bm resulting from move m, B is the set of potential positions, R is the reward function
	 of initial position A and ending position Bm, and breadth is number of moves explored compared with the total, len(B). Here the softmax
	 behaves like "the probability of playing move m", since the NN should expect an opponent to play the strongest moves it can compute. The
	 reward is dampened by the fraction of moves actually explored, mirroring the compounding uncertainty that the expected value estimates are
	 reliable, with increasing depth of a partial tree traversal.
	-Thus the NN is trained to output the expected value of all future rewards from the input position
	