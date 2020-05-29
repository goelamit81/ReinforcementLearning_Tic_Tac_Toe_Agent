# Reinforcement_Learning_Tic_Tac_Toe_Agent


You need to build an RL agent that learns to play Numerical Tic-Tac-Toe with odd numbers (the agent will always make the first move). You need to train your agent using Q-Learning. The environment is playing randomly with the agent, i.e. its strategy is to put an even number randomly in an empty cell. If your agent wins the game, it gets 10 points, if the environment wins, the agent loses 10 points. And if the game ends in a draw, it gets 0. Also, you want the agent to win in as few moves as possible, so for each move, it gets a -1 point.

You need to accomplish following:

1. Create an MDP for Numerical Tic-Tac-Toe game. The basic framework for this is:

- Initialise the state

- Define the action space for each state. (Be careful in defining actions. The actions are not the same for each state)

- Define the winning states: the sum of three numbers in a row, column or diagonal is 15.

- Define the terminal states (win,tie,loss)

- Build the reward structure as below:

 - (a) +10 if the agent wins (makes 15 points first)

 - (b) -10 if the environment wins

 - (c) 0 if the game ends in a draw (no one is able to make 15 and the board is filled up)

 - (d) -1 for each move agent takes

- Define a step function which takes in an input of the agent’s action and state; and outputs the next state and reward. (Make sure you incorporate environment’s move in the next state).

2. Build an agent that learns the game by Q-Learning. You can choose the hyperparameters (epsilon (decay rate), learning-rate, discount factor) of your choice. For that, you can train the model iteratively to obtain a good combination of hyperparameters. You won’t be evaluated on your choice of the hyperparameters. You need to submit only the final model. 

- While updating the Q-values, if the next state is a terminal state, then the Q-values from that state are 0. (No action is possible from that state)

3. Q-values convergence- check whether Q-values learnt by the agent have converged or not. Sample 4 state-action pairs and plot it with the number of episodes to understand the convergence.
