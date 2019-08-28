# Deep-Notakto
A deep-reinforcement learning approach to solving mis&egrave;re
combinatorial games.
This project won the 12th Grade category at the 2018 Hoosier Science and
Engineering Fair and competed in the 2018 Intel International Science and Engineering Fair.
This program was developed in conjunction with the Indiana University
Computer Vision Lab.

Keywords: reinforcement learning, combinatorial games, mis&egrave;re
play, AlphaZero

### Abstract
In 2017, Silver et al. designed a reinforcement learning algorithm that
learned tabula rasa without any human-generated data. They applied this
algorithm to the games of Go, Chess, and Shogi with positive results
suggesting that the algorithm is general enough to learn many other
games as well. Here, we investigate how well this technique performs on
the misère version of a combinatorial game, misère tic-tac-toe, in which
the goal is not to complete a row, column, or diagonal. We found that a
successful Deep-Q neural network could be trained solely from self-play
for the case of a three-by-three board.  However, with board sizes of
four-by-four and larger, the models failed to converge to a winning
strategy. Although the computational power of the original tests could
not be matched and the hyperparameters could not be as finely tuned,
this result suggests that misère games may be fundamentally different
than their regular counterparts, possibly requiring different algorithms
and approaches.

### Introduction
A central goal of machine learning has been to train models tabula rasa,
i.e. from a blank slate with no human guidance. Progress towards this
goal has accelerated with DeepMind’s AlphaGo Zero’s win over the
previous version of AlphaGo which defeated the Go world champion in 2016
and their models’ wins over the strongest computer opponents in Chess
and Shogi. The training algorithm proposed requires no human-crafted
features or human-generated game data and is general enough to be
applied to games other than those tested. Many combinatorial games
(discrete zero-sum perfect information games) can be “solved"
mathematically. However, the misère versions of these games — in which
the winner by the regular game’s rules loses — are much more difficult
to analyze and thus possibly harder for computers to play well. Better
understandings of these games could lead to advances in combinatorics,
optimization, complexity theory, and even gene folding.

### Mis&egrave;re Tic-Tac-Toe
Misère tic-tac-toe (“Notakto”) is like tic-tac-toe played on an n x n
board, except that both players place the same symbol, and the player
that completes a row, column, or diagonal loses. For certain n, the game
is “weakly solved” -- it is known which player can win no matter the
actions of the opponent. For a 3x3 board, player one can always force a
win, while for 4x4, player two can always win. By computer serach, we
know which player can win (called the game theoretic value) for certain
n, but not the winning strategy: player one wins for 5x5, player one
wins for 6x6, and player two wins all boards with side lengths divisible
by four.

### Background
To measure the complexity of a game, we inspect the size of its game
tree: the set of all possible game sequences for a given game. Often
visualized as a tree where the nodes are game states and the edges are
the possible moves, each time a game is played, players trace out one
branch of the game tree. For small games (games with few options and
outcomes), the game tree is computable explicitly. For example,
tic-tac-toe has a complexity upper bound of 3^9=19,683 possible board
states (ignoring symmetries and impossible game states) which, assuming
18 bits per board, would only require about 44 kilobytes to store.
However, for larger games like chess and Go where there are sometimes
hundreds of options for every move, the game tree is intractable.
For comparison, a similar upper bound computation for Go calculates
3^361=1.7x10^172 possible board states which, assuming 722 bits per
board, would necessitate over 10162 terabytes, more than squared the
number of atoms in the universe.

One goal of combinatorial game theory is to determine the game-theoretic
value of a game. This value indicates whether the current player is the
winner or the loser of the game with optimal play. In perfect
information zero-sum games, the game-theoretic value is +1 if the
current player wins, 0 if neither player wins, or -1 if the opposite
player wins. For example, the game-theoretic value of a 3x3 Notakto game
starting on a blank board is +1 for the first player because that player
has a strategy that can guarantee a win and -1 for the second player
because, under optimal play, that player always loses. Games for which
we know the game-theoretic value are called “solved.” A game is
“ultra-weakly solved” if only the value of the initial starting position
is known and “weakly solved” if the strategy for the first player to
obtain that value is also known. A game  is “strongly solved” if the
value and the strategy to obtain that value is known for every legal
game state.

A Markov Decision Process (MDP) is defined by a set of possible states,
a set of possible actions from a given state, a state transition
function that transfers an agent from a given state to another state
after performing an action (generally, this may be stochastic but is
deterministic in our environment), and a reward function that indirectly
defines the desired actions. For a given non-terminal state, an agent
chooses an action to take from a distribution called a policy, transfers
the environment to a new state, and receives a reward. A game is over
when the environment reaches a terminal state where it emits a final
winner or outcome called the value of the terminal state. The value of a
non-terminal state is the expected value of all states following it. The
goal of an agent is to design its policy such that it maximizes the
discounted expected reward. Equivalently, the agent can find the true
value function and choose actions according it.

### Deep-Q Model
To model the decisions of a player (the “policy”) and the
likelihood that the player will win (the ”value”), we use a deep neural
network, a system of addition and multiplication nodes (”neurons”) that
abstractly model the functioning of a human brain. Like Silver et al,
we use a neural network to produce a policy and value for a given state.
The value of the game at the given state, represented by a scalar in
[-1, 1], is  calculated by applying the hyperbolic tangent function to
the first output neuron. The policy, represented by a discrete
probability distribution, is calculated by applying the softmax
normalization function to the remaining outputs. The network consists of
randomly-initialized fully-connected layers each followed by the Relu
activation function. Training data for the model is stored as a set of
tuples {s, target,z} where target is a desired policy generated by the
training algorithm and z= +/- 1 is the eventual winner from the
perspective of the current player. Old data points are discarded with a
memory replay queue as new ones are generated. During training, random
batches of the memory replay are passed to the network which is updated
with gradient descent minimizing the loss function defined by Silver et
al.

### Hypothesis
With limited computational resources, we will not be able to train a
perfect model for board sizes five and greater but we will be able to
train perfect models such that player one can win 100% of  games on 3x3
boards and player two can win 100% of games on 4x4 boards.

### Training Algorithm (Silver et al's AlphaZero algorithm)
Designed by Silver et al, the algorithm generates target policies for a
given move by playing games against itself guided by a modified
Monte-Carlo Tree Search. Each state is the root node of a tree with
child nodes representing the state after a given action and where each
node contains a set of statistics {N, W, P}. N is the visit count
for a given node, W is the total value of all runs that traversed the
node, and P is the probability of selecting an action according to the
model. The algorithm proceeds by:

__(1) Selection__ Starting from a given
root node, choose an action by a hueristic depending on N, W, and P
until a node is reached whose possible actions have not all been
evaluated.

__(2) Expand and Evaluate__ Randomly choose an un-explored action, play
the action, and evaluate the value and action probabilities with the
model. Initialize a new child with {N = 1, W = value, P = probability}.

__(3) Backpropogate__ Pass the value up the tree. For each node
traversed, update the node’s statistics by {N'=N+1, W'= W+ new value,
P'=P}. (4) Repeat Repeat the process for a set number of simulations.

__(5) Calculate Policy__ Starting with a blank board, run steps (1) to
(4). Compute the target policy and choose an action from this policy.
Play this action and begin a new tree with root node. Repreat this
process until a terminal state. Save the data from this game. Train the
neural network over these data points and run further self-play games
with the newly trained network.

### Analysis
The number of neurons and layers seems not to significantly impact score
or required number of training iterations, which is surprising because
higher capacity models should have greater capacity to model data. Some
models trained within seconds, suggesting that the algorithm would be
able to easily scale to larger boards, but this was not the case.
Although a 4x4 board has only 128 times as many possible positions as
3x3, our best model only won 46% of games even after training 30,000
times longer, and the trend did not suggest that it would win more games
given more time (Figure 2). All other 4x4 models were unable even to
achieve a 40% win rate. The training performed by Silver et al with
massive computing resources involved over 4.9 million generated games
each using 16,000 simulations per game5; our difficulty with the 4x4
board may be due to insufficient training period with too weak computing
power.

### Future Work
Performing tree searches to determine the game theoretic values for
games with large search spaces, like large misère tic-tac-toe games or
Go, would take thousands of years with current technology. The models in
this work could reduce this time by orders of magnitude by “tree
pruning” in which the search can prioritize certain branches of the tree
from the model’s suggestions. We hope to use this method to find the
guaranteed winners and possibly the winning strategies of large misère
tic-tac-toe games. Definitively solving these games could lead to
understanding in other areas in which combinatorial games naturally
occur as complexity theory and optimization. In terms of Silver et al’s
algorithm’s effectiveness against misère games, nothing can be
definitively concluded without conducting experiments with much greater
computational power. Also, it would be worth testing different neural
network architectures such as convolutional neural networks, residual
networks, and capsule networks to check the effectiveness of the
algorthim versus the effectiveness of the model being trained.

### References
Allis, L. V. (1994). Searching for solutions in games and artificial
intelligence. Rijksuniversiteit Limburg.

arXiv:1712.01815 [cs.AI]

Heule, M. J., & Rothkrantz, L. J. (2007). Solving games: Dependence of
applicable solving procedures. Science of Computer Programming, 67(1),
105-124.

Littman, M. L. (1994). Markov games as a framework for multi-agent
reinforcement learning. In Machine Learning Proceedings 1994
(pp. 157-163).

Silver, D., Huang, A., Maddison, C. J., Guez, A., Sifre, L., Van Den
Driessche, G., ... & Dieleman, S. (2016). Mastering the game of Go with
deep neural networks and tree search. Nature, 529(7587), 484-489.

Silver, D., Schrittwieser, J., Simonyan, K., Antonoglou, I., Huang, A.,
Guez, A., ... & Chen, Y. (2017). Mastering the game of Go without human
knowledge. Nature, 550(7676), 354.
