from gym import spaces
import numpy as np
import random
from itertools import groupby
from itertools import product

#######################################################################################################################################

class TicTacToe():

    ###################################################################################################################################
    def __init__(self):
        """initialise the board"""
        
        # initialise state as an array
        self.state = [np.nan for _ in range(9)]  # initialises the board position, can initialise to an array or matrix
        # all possible numbers
        self.all_possible_numbers = [i for i in range(1, len(self.state) + 1)] # , can initialise to an array or matrix

        self.reset()

    ###################################################################################################################################
    def is_winning(self, curr_state):
        """Takes state as an input and returns whether any row, column or diagonal has winning sum
        Example: Input state- [1, 2, 3, 4, nan, nan, nan, nan, nan]
        Output = False"""
 
        # 2 diagonal positions of a 3 X 3 board
        diagonal_pos = [[0, 4, 8], [2, 4, 6]]

        # 3 horizontal positions of a 3 X 3 board
        horizontal_pos = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]

        # 3 vertical positions of a 3 X 3 board
        vertical_pos = [[0, 3, 6], [1, 4, 7], [2, 5, 8]]

        # Take sum across each group of positions
        diagonal_sum = [np.nansum(np.array(curr_state)[i], dtype=int) for i in diagonal_pos]
        horizontal_sum = [np.nansum(np.array(curr_state)[i], dtype=int) for i in horizontal_pos]
        vertical_sum = [np.nansum(np.array(curr_state)[i], dtype=int) for i in vertical_pos]


        #  Game is won if sum across any direction (diagonal, horizontal, vertical) is equal to 15
        diagonal_filtered = list(filter(lambda x: x == 15, diagonal_sum))
        horizontal_filtered = list(filter(lambda x: x == 15, horizontal_sum))
        vertical_filtered = list(filter(lambda x: x == 15, vertical_sum))
        
        #  Check if game is won
        if len(diagonal_filtered) != 0 or len(horizontal_filtered) != 0 or len(vertical_filtered) != 0:
            return True
        else:
            return False

    ###################################################################################################################################
    def is_terminal(self, curr_state):
        # Terminal state could be winning state or when the board is filled up

        if self.is_winning(curr_state) == True:
            return True, 'Win'

        elif len(self.allowed_positions(curr_state)) ==0:
            return True, 'Tie'

        else:
            return False, 'Resume'

    ###################################################################################################################################
    def allowed_positions(self, curr_state):
        """Takes state as an input and returns all indexes that are blank"""
        return [i for i, val in enumerate(curr_state) if np.isnan(val)]

    ###################################################################################################################################
    def allowed_values(self, curr_state):
        """Takes the current state as input and returns all possible (unused) values that can be placed on the board"""

        used_values = [val for val in curr_state if not np.isnan(val)]
        agent_values = [val for val in self.all_possible_numbers if val not in used_values and val % 2 !=0]
        env_values = [val for val in self.all_possible_numbers if val not in used_values and val % 2 ==0]

        return (agent_values, env_values)

    ###################################################################################################################################
    def action_space(self, curr_state):
        """Takes the current state as input and returns all possible actions, i.e, all combinations of allowed positions and allowed values"""

        agent_actions = product(self.allowed_positions(curr_state), self.allowed_values(curr_state)[0])
        env_actions = product(self.allowed_positions(curr_state), self.allowed_values(curr_state)[1])
        return (agent_actions, env_actions)

    ###################################################################################################################################
    def state_transition(self, curr_state, curr_action):
        """Takes current state and action and returns the board position just after agent's move.
        Example: Input state- [1, 2, 3, 4, nan, nan, nan, nan, nan], action- [7, 9] or [position, value]
        Output = [1, 2, 3, 4, nan, nan, nan, 9, nan]
        """

        curr_state[curr_action[0]] = curr_action[1]
        return curr_state

    ###################################################################################################################################
    def step(self, curr_state, curr_action):
        """Takes current state and action and returns the next state, reward and whether the state is terminal. Hint: First, check the board position after
        agent's move, whether the game is won/loss/tied. Then incorporate environment's move and again check the board status.
        Example: Input state- [1, 2, 3, 4, nan, nan, nan, nan, nan], action- [7, 9] or [position, value]
        Output = ([1, 2, 3, 4, nan, nan, nan, 9, nan], -1, False)"""

        # In Output, will be returning another field namely result which will mention if game is won by Agent or Environment or it is a Tie or continue playing

        # Get intermediate state - agent's move
        state_post_agent_action = self.state_transition(curr_state, curr_action)

        # Check if state is terminal post agent's move
        terminal_state, game_status = self.is_terminal(state_post_agent_action)

        # Decide reward if it is a terminal state after agent's move
        if terminal_state == True:

            if game_status == 'Win':
                reward=10
                result = "Agent" # Need to track who wins so can understand if agent is learning better
            else:
                reward=0
                result = "Tie" # Need to track if it is a Tie

            return state_post_agent_action, reward, terminal_state, result
        else:
            # Allow environment's move - find out action space for environment and pick one random action
            env_action = random.choice([i for i in self.action_space(state_post_agent_action)[1]])

            # Transition state due to environment's move
            state_post_env_action = self.state_transition(state_post_agent_action, env_action)

            # Check if state is terminal post environment's move
            terminal_state, game_status = self.is_terminal(state_post_env_action)

            # Decide reward if it is a terminal state after environemnt's move
            if terminal_state == True:
                
                if game_status == 'Win':
                    reward=-10
                    result = "Environment" # Need to track who wins so can understand if agent is learning better
                else:
                    reward=0
                    result = "Tie" # Need to track if it is a Tie
            else:
                reward=-1
                result = "Resume" # If not a terminal state then continue playing

            return state_post_env_action, reward, terminal_state, result

    ###################################################################################################################################
    def reset(self):
        return self.state

#######################################################################################################################################