# Yuqi Shao 199507014208
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import time
from IPython import display

# Implemented methods
methods = ['DynProg', 'ValIter'];

# Some colours
LIGHT_RED    = '#FFC4CC';
LIGHT_GREEN  = '#95FD99';
BLACK        = '#000000';
WHITE        = '#FFFFFF';
LIGHT_PURPLE = '#E8D0FF';
LIGHT_ORANGE = '#FAE0C3';

# if the minotaur has 35% probability moves towards person
random_minotaur_flag = False

class Maze:

    # Actions
    STAY       = 0
    MOVE_LEFT  = 1
    MOVE_RIGHT = 2
    MOVE_UP    = 3
    MOVE_DOWN  = 4

    # Give names to actions
    actions_names = {
        STAY: "stay",
        MOVE_LEFT: "move left",
        MOVE_RIGHT: "move right",
        MOVE_UP: "move up",
        MOVE_DOWN: "move down"
    }

    # Reward values
    REWARD_STEP = 0
    REWARD_EXIT = 100
    REWARD_WALL = -1
    REWARD_DEATH = -100
    REWARD_MOVING_AT_EXIT = -1


    def __init__(self, maze, weights=None, random_rewards=None):
        """ Constructor of the environment Maze.
        """
        self.maze                     = maze;
        self.actions                  = self.__actions();
        self.states, self.map         = self.__states();
        self.n_actions                = len(self.actions);
        self.n_states                 = len(self.states);
        self.transition_probabilities = self.__transitions();
        self.rewards                  = self.__rewards(weights=weights,
                                                random_rewards=random_rewards);

    def __actions(self):
        actions = dict();
        actions[self.STAY] = (0, 0);
        actions[self.MOVE_LEFT] = (0, -1);
        actions[self.MOVE_RIGHT] = (0, 1);
        actions[self.MOVE_UP] = (-1, 0);
        actions[self.MOVE_DOWN] = (1, 0);
        return actions;

    def __states(self):
        states = dict();
        map = dict();
        end = False;
        s = 0;
        for i in range(self.maze.shape[0]): #person
            for j in range(self.maze.shape[1]): #person
                for g in range(self.maze.shape[0]): #minotaur
                    for h in range(self.maze.shape[1]): #minotaur
                        if self.maze[i, j] != 1:
                            states[s] = (i, j, g, h);
                            map[(i, j, g, h)] = s;
                            s += 1;

        return states, map

    def __get_next_human_position(self, state, action):
        """ Makes a step in the maze, given a current position and an action.
            If the action STAY or an inadmissible action is used, the agent stays in place.

            :return tuple next_cell: Position (row,column) of the human.
        """
        # Compute the future position given current (state, action)
        row = self.states[state][0] + self.actions[action][0];
        col = self.states[state][1] + self.actions[action][1];
        # Is the future position an impossible one ?
        hitting_maze_walls =  (row == -1) or (row == self.maze.shape[0]) or \
                              (col == -1) or (col == self.maze.shape[1]) or \
                              (self.maze[row,col] == 1);
        # Based on the impossiblity check return the next state.
        if hitting_maze_walls:
            return (self.states[state][0], self.states[state][1]);
        else:
            return (row, col);

    def __get_transitions_minotaur(self, s):
        transition_probabilities_minotaur = {}
        (row, col) = (self.states[s][2], self.states[s][3])
        if (row == 0) and (col == 0):
            transition_probabilities_minotaur[(row + 1, col)] = 1/2
            transition_probabilities_minotaur[(row, col + 1)] = 1/2
        elif (row == 0) and (col == self.maze.shape[1] - 1):
            transition_probabilities_minotaur[(row, col - 1)] = 1/2
            transition_probabilities_minotaur[(row + 1, col)] = 1/2
        elif (row == self.maze.shape[0] - 1) and (col == 0):
            transition_probabilities_minotaur[(row, col + 1)] = 1/2
            transition_probabilities_minotaur[(row - 1, col)] = 1/2
        elif (row == self.maze.shape[0] - 1) and (col == self.maze.shape[1] - 1):
            transition_probabilities_minotaur[(row, col - 1)] = 1/2
            transition_probabilities_minotaur[(row - 1, col)] = 1/2
        elif (row == 0):
            transition_probabilities_minotaur[(row, col - 1)] = 1/3
            transition_probabilities_minotaur[(row + 1, col)] = 1/3
            transition_probabilities_minotaur[(row, col + 1)] = 1/3
        elif (col == 0):
            transition_probabilities_minotaur[(row - 1, col)] = 1/3
            transition_probabilities_minotaur[(row + 1, col)] = 1/3
            transition_probabilities_minotaur[(row, col + 1)] = 1/3
        elif (row == self.maze.shape[0] - 1):
            transition_probabilities_minotaur[(row, col - 1)] = 1/3
            transition_probabilities_minotaur[(row - 1, col)] = 1/3
            transition_probabilities_minotaur[(row, col + 1)] = 1/3
        elif (col == self.maze.shape[1] - 1):
            transition_probabilities_minotaur[(row, col - 1)] = 1/3
            transition_probabilities_minotaur[(row - 1, col)] = 1/3
            transition_probabilities_minotaur[(row + 1, col)] = 1/3
        else:
            transition_probabilities_minotaur[(row, col - 1)] = 1/4
            transition_probabilities_minotaur[(row - 1, col)] = 1/4
            transition_probabilities_minotaur[(row + 1, col)] = 1/4
            transition_probabilities_minotaur[(row, col + 1)] = 1/4

        if random_minotaur_flag:
            for key in transition_probabilities_minotaur:
                transition_probabilities_minotaur[key] = transition_probabilities_minotaur[key] * 0.65
            # get human position
            (prow, pcol) = (self.states[s][0], self.states[s][1])
            (drow, dcol) = (row-prow, col-pcol)
            next_possible_positions = []
            if drow > 0:
                if dcol > 0:
                    next_possible_positions.append((row - 1, col))
                    next_possible_positions.append((row, col - 1))
                elif dcol < 0:
                    next_possible_positions.append((row - 1, col))
                    next_possible_positions.append((row, col + 1))
                else: #=0
                    next_possible_positions.append((row - 1, col))
            elif drow < 0:
                if dcol > 0:
                    next_possible_positions.append((row + 1, col))
                    next_possible_positions.append((row, col - 1))
                elif dcol < 0:
                    next_possible_positions.append((row + 1, col))
                    next_possible_positions.append((row, col + 1))
                else: #=0
                    next_possible_positions.append((row + 1, col))
            else: #=0
                if dcol > 0:
                    next_possible_positions.append((row, col - 1))
                elif dcol < 0:
                    next_possible_positions.append((row, col + 1))
                #else: #dead, no nothing

            p = 1/(len(next_possible_positions))
            for n in next_possible_positions:
                transition_probabilities_minotaur[n] += 0.35*p

        return transition_probabilities_minotaur

    def __simulate_next_minotaur_position(self, s):
        transition_prob_minotaur = self.__get_transitions_minotaur(s)
        n = len(transition_prob_minotaur)
        m = random.randint(0, n-1)
        keys_list = list(transition_prob_minotaur)
        next_position = keys_list[m]
        return next_position

    def __transitions(self):
        """ Computes the transition probabilities for every state action pair.
            :return numpy.tensor transition probabilities: tensor of transition
            probabilities of dimension S*S*A
        """
        # Initialize the transition probailities tensor (S,S,A)
        dimensions = (self.n_states,self.n_states,self.n_actions);
        transition_probabilities = np.zeros(dimensions);

        # Compute the transition probabilities. Note that the transitions
        # are stochastic.
        for s in range(self.n_states):
            for a in range(self.n_actions):
                next_human_position = self.__get_next_human_position(s,a)
                transition_prob_minotaur = self.__get_transitions_minotaur(s)

                for m in transition_prob_minotaur:
                    transition_probabilities[self.map[(next_human_position[0],next_human_position[1],
                                              m[0],m[1])], s, a] = transition_prob_minotaur[m];

        return transition_probabilities;

    def __rewards(self, weights=None, random_rewards=None):

        rewards = np.zeros((self.n_states, self.n_actions));

        # If the rewards are not described by a weight matrix
        for s in range(self.n_states):
            for a in range(self.n_actions):
                next_position = self.__get_next_human_position(s,a);
                # Reward for death
                if ((self.states[s][0], self.states[s][1]) == (self.states[s][2], self.states[s][3])) and \
                        (self.maze[(self.states[s][0], self.states[s][1])] != 2):
                    rewards[s, a] = self.REWARD_DEATH;
                # Rewrd for hitting a wall
                elif (self.states[s][0],self.states[s][1]) == next_position and a != self.STAY:
                    rewards[s,a] = self.REWARD_WALL;
                # Reward for reaching the exit
                elif self.maze[next_position] == 2:
                    rewards[s,a] = self.REWARD_EXIT;
                # Reward for moving when already at the exit
                elif self.maze[(self.states[s][0],self.states[s][1])] == 2 and a != self.STAY:
                    rewards[s,a] = self.REWARD_MOVING_AT_EXIT;
                # Reward for taking a step to an empty cell that is not the exit
                else:
                    rewards[s,a] = self.REWARD_STEP;

        return rewards;

    def simulate(self, start, policy, method):
        if method not in methods:
            error = 'ERROR: the argument method must be in {}'.format(methods);
            raise NameError(error);

        path = list();
        if method == 'DynProg':
            # Deduce the horizon from the policy shape
            horizon = policy.shape[1];
            # Initialize current state and time
            t = 0;
            s = self.map[start];
            # Add the starting position in the maze to the path
            path.append(start);
            while t < horizon - 1:
                # Move to next state given the policy and the current state
                (pr, pc) = self.__get_next_human_position(s, policy[s, t]);
                (mr, mc) = self.__simulate_next_minotaur_position(s)
                # Add the position in the maze corresponding to the next state
                # to the path
                path.append((pr, pc, mr, mc))
                # Update time and state for next iteration
                t += 1;
                s = self.map[(pr, pc, mr, mc)];
        if method == 'ValIter':
            # Initialize current state, next state and time
            t = 1;
            s = self.map[start];
            # Add the starting position in the maze to the path
            path.append(start);
            # Move to next state given the policy and the current state
            (pr, pc) = self.__get_next_human_position(s, policy[s]);
            (mr, mc) = self.__simulate_next_minotaur_position(s)
            next_s = self.map[(pr, pc, mr, mc)]
            # Add the position in the maze corresponding to the next state
            # to the path
            path.append((pr, pc, mr, mc));
            # Loop while state is not the goal state
            while self.maze[(self.states[next_s][0],self.states[next_s][1])] != 2:
                # Update state
                s = next_s;
                # Move to next state given the policy and the current state
                (pr, pc) = self.__get_next_human_position(s, policy[s]);
                (mr, mc) = self.__simulate_next_minotaur_position(s)
                next_s = self.map[(pr, pc, mr, mc)]
                # Add the position in the maze corresponding to the next state
                # to the path
                path.append((pr, pc, mr, mc))
                # Update time and state for next iteration
                t += 1;
        return path

    def compute_probability_exiting(self, start, policy, method, iterations, T):
        n_death = 0
        n_exit = 0
        for n in range(iterations):
            path = self.simulate(start, policy, method)
            if method == 'DynProg':
                for i in range(len(path)):
                    # count death before entering exit
                    if self.maze[path[i][0], path[i][1]] == 2:
                        n_exit += 1
                        break
                    if (path[i][0], path[i][1]) == (path[i][2], path[i][3]):
                        print(path)
                        print(i)
                        n_death += 1
                        break
            elif method == 'ValIter':
                if len(path)> T[n] + 1:
                    continue
                dead = 0
                for i in range(len(path)):
                    if (path[i][0], path[i][1]) == (path[i][2], path[i][3]):
                        print(path)
                        print(i)
                        dead = 1
                        break
                if dead == 0:
                    n_exit += 1
        prob = n_exit/iterations
        return prob


def dynamic_programming(env, horizon):
    """ Solves the shortest path problem using dynamic programming
        :input Maze env           : The maze environment in which we seek to
                                    find the shortest path.
        :input int horizon        : The time T up to which we solve the problem.
        :return numpy.array V     : Optimal values for every state at every
                                    time, dimension S*T
        :return numpy.array policy: Optimal time-varying policy at every state,
                                    dimension S*T
    """

    # The dynamic prgramming requires the knowledge of :
    # - Transition probabilities
    # - Rewards
    # - State space
    # - Action space
    # - The finite horizon
    p         = env.transition_probabilities;
    r         = env.rewards;
    n_states  = env.n_states;
    n_actions = env.n_actions;
    T         = horizon;

    # The variables involved in the dynamic programming backwards recursions
    V      = np.zeros((n_states, T+1));
    policy = np.zeros((n_states, T+1));
    Q      = np.zeros((n_states, n_actions));


    # Initialization
    Q            = np.copy(r);
    V[:, T]      = np.max(Q,1);
    policy[:, T] = np.argmax(Q,1);

    # The dynamic programming bakwards recursion
    for t in range(T-1,-1,-1):
        # Update the value function acccording to the bellman equation
        for s in range(n_states):
            for a in range(n_actions):
                # Update of the temporary Q values
                Q[s,a] = r[s,a] + np.dot(p[:,s,a],V[:,t+1])
        # Update by taking the maximum Q value w.r.t the action a
        V[:,t] = np.max(Q,1);
        # The optimal action is the one that maximizes the Q function
        policy[:,t] = np.argmax(Q,1);
    return V, policy;

def value_iteration(env, gamma, epsilon):
    """ Solves the shortest path problem using value iteration
        :input Maze env           : The maze environment in which we seek to
                                    find the shortest path.
        :input float gamma        : The discount factor.
        :input float epsilon      : accuracy of the value iteration procedure.
        :return numpy.array V     : Optimal values for every state at every
                                    time, dimension S*T
        :return numpy.array policy: Optimal time-varying policy at every state,
                                    dimension S*T
    """
    # The value itearation algorithm requires the knowledge of :
    # - Transition probabilities
    # - Rewards
    # - State space
    # - Action space
    # - The finite horizon
    p         = env.transition_probabilities;
    r         = env.rewards;
    n_states  = env.n_states;
    n_actions = env.n_actions;

    # Required variables and temporary ones for the VI to run
    V   = np.zeros(n_states);
    Q   = np.zeros((n_states, n_actions));
    BV  = np.zeros(n_states);
    # Iteration counter
    n   = 0;
    # Tolerance error
    #tol = (1 - gamma)* epsilon/gamma;

    # Initialization of the VI
    for s in range(n_states):
        for a in range(n_actions):
            Q[s, a] = r[s, a] + gamma*np.dot(p[:,s,a],V);
    BV = np.max(Q, 1);

    # Iterate until convergence
    while np.linalg.norm(V - BV) >= epsilon and n < 200:
        # Increment by one the numbers of iteration
        n += 1;
        # Update the value function
        V = np.copy(BV);
        # Compute the new BV
        for s in range(n_states):
            for a in range(n_actions):
                Q[s, a] = r[s, a] + gamma*np.dot(p[:,s,a],V);
        BV = np.max(Q, 1);
        # Show error
        #print(np.linalg.norm(V - BV))

    # Compute policy
    policy = np.argmax(Q,1);
    # Return the obtained policy
    return V, policy;



def animate_solution(maze, path):
    # Map a color to each cell in the maze
    col_map = {0: WHITE, 1: BLACK, 2: LIGHT_GREEN};

    # Size of the maze
    rows,cols = maze.shape;


    # Remove the axis ticks and add title title
    ax = plt.gca();
    ax.set_title('Policy simulation');
    ax.set_xticks([]);
    ax.set_yticks([]);

    # Give a color to each cell
    colored_maze = [[col_map[maze[j,i]] for i in range(cols)] for j in range(rows)];

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols,rows))

    # Create a table to color
    grid = plt.table(cellText=None,
                     cellColours=colored_maze,
                     cellLoc='center',
                     loc=(0,0),
                     edges='closed');

    # Modify the hight and width of the cells in the table
    tc = grid.properties()['children']
    for cell in tc:
        cell.set_height(1.0/rows);
        cell.set_width(1.0/cols);


    # Update the color at each frame
    for i in range(len(path)):
        if i > 0:
            if (path[i-1][0], path[i-1][1]) != (path[i][0], path[i][1]):
                grid.get_celld()[(path[i-1][0], path[i-1][1])].set_facecolor(col_map[maze[path[i-1][0],path[i-1][1]]])
                grid.get_celld()[(path[i-1][0], path[i-1][1])].get_text().set_text('')
            if (path[i-1][2], path[i-1][3]) != (path[i][2], path[i][3]):
                grid.get_celld()[(path[i-1][2], path[i-1][3])].set_facecolor(col_map[maze[path[i-1][2],path[i-1][3]]])
                grid.get_celld()[(path[i-1][2], path[i-1][3])].get_text().set_text('')
            if maze[path[i][0], path[i][1]] == 2:
                grid.get_celld()[(path[i][0], path[i][1])].set_facecolor(LIGHT_GREEN)
                grid.get_celld()[(path[i][0], path[i][1])].get_text().set_text('Player reached exit')
                print('reached exit')
                break
            elif (path[i][0], path[i][1]) == (path[i][2], path[i][3]):
                grid.get_celld()[(path[i][0], path[i][1])].set_facecolor(LIGHT_GREEN)
                grid.get_celld()[(path[i][0], path[i][1])].get_text().set_text('Player dead')
                print('dead')
                break
        grid.get_celld()[(path[i][0], path[i][1])].set_facecolor(LIGHT_ORANGE)
        grid.get_celld()[(path[i][0], path[i][1])].get_text().set_text('Player')
        grid.get_celld()[(path[i][2], path[i][3])].set_facecolor(LIGHT_RED)
        grid.get_celld()[(path[i][2], path[i][3])].get_text().set_text('Minotaur')
        display.display(fig)
        display.clear_output(wait=True)
        time.sleep(1)

