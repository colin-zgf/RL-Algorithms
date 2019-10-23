import numpy as np
from collections import defaultdict
import sys
from gym.envs.toy_text import discrete
from io import StringIO
from itertools import product, count
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import Rectangle, Arrow, Circle
import matplotlib.patches as patches
import cv2
import ast
import random

r=1 # road
b=2 # barrier
g=3 # goal

# 32 cases
grid_array = np.array([[[r,r,r,r,r,g],[r,r,b,b,r,r],[b,r,b,b,r,r],[r,r,r,r,b,b],
               [r,r,b,r,r,r],[r,r,r,r,r,r]],\
              [[b,r,r,r,r,g],[r,b,r,r,b,r],[r,r,b,r,r,b],[r,r,r,b,r,r],
               [r,r,b,r,r,r],[r,r,r,r,b,r]],\
              [[r,r,r,b,r,g],[r,r,b,r,r,b],[b,r,r,r,b,r],[r,r,r,b,r,r],
               [r,r,b,r,r,r],[r,r,r,r,b,r]],\
              [[r,r,r,r,r,g],[r,r,r,r,b,r],[b,r,r,b,r,b],[r,r,b,r,r,r],
               [r,b,r,r,b,r],[r,r,r,b,r,r]],\
              [[r,r,r,r,r,g],[r,b,b,b,r,b],[r,r,r,r,b,r],[b,r,b,r,r,r],
               [r,r,r,r,r,r],[r,b,r,r,r,r]],\
              [[r,r,b,r,r,g],[r,b,r,r,r,b],[r,r,r,r,b,r],[r,b,r,b,r,r],
               [r,r,b,r,r,r],[b,r,r,r,r,r]],\
              [[r,b,r,r,r,g],[r,r,r,b,r,r],[r,b,b,b,b,b],[r,r,r,r,r,r],
               [r,r,b,r,r,r],[r,r,r,r,r,r]],\
              [[r,r,b,r,r,g],[r,r,b,r,r,b],[r,r,r,r,b,r],[b,r,b,r,r,r],
               [r,r,r,r,b,r],[r,r,r,b,r,r]],\
              [[r,r,r,b,r,g],[b,r,r,b,r,r],[r,r,r,r,r,b],[r,b,r,b,r,r],
               [r,r,r,r,r,b],[r,r,r,r,b,r]],\
              [[r,r,r,b,r,g],[r,r,b,r,r,b],[r,b,r,r,r,b],[r,r,b,b,r,r],
               [b,r,r,r,r,r],[r,r,r,r,r,r]],\
              [[r,b,r,r,r,g],[r,r,b,r,r,b],[r,r,r,b,r,r],[r,b,r,r,b,r],
               [r,r,r,b,r,r],[r,b,r,r,r,r]],\
              [[r,r,r,r,r,g],[b,b,b,b,r,b],[r,r,r,b,r,r],[r,r,b,r,r,r],
               [r,r,r,r,b,r],[r,r,r,r,r,r]],\
              [[r,r,r,r,r,r],[r,r,b,b,b,b],[r,b,r,r,r,r],[r,g,b,r,b,r],
               [r,b,r,r,r,b],[r,r,r,b,r,r]],\
              [[r,r,r,r,r,b],[r,b,r,b,r,r],[r,r,b,r,b,r],[r,g,r,r,b,r],
               [r,r,b,b,b,r],[b,r,r,r,r,r]],\
              [[b,r,r,b,r,r],[r,b,r,r,b,r],[r,r,b,r,r,r],[r,g,r,b,b,r],
               [r,b,r,r,b,r],[r,r,b,r,r,r]],\
              [[r,r,r,r,r,r],[r,b,r,r,r,r],[r,r,b,b,b,b],[b,g,r,r,r,r],
               [r,r,b,r,b,r],[r,b,r,r,r,b]],\
              [[r,b,r,r,r,r],[r,r,r,r,b,r],[b,r,b,b,r,r],[r,g,r,b,r,r],
               [r,b,b,r,r,b],[b,r,r,r,r,r]],\
              [[r,b,b,r,r,r],[r,r,r,r,b,r],[r,b,b,r,r,r],[r,g,r,b,r,r],
               [r,r,b,r,b,b],[r,r,r,r,r,b]],\
              [[r,r,b,r,r,r],[r,r,r,r,b,r],[b,b,r,b,b,r],[r,g,b,r,r,r],
               [r,b,r,r,b,r],[r,r,r,r,b,r]],\
              [[r,r,b,r,r,r],[b,r,r,r,r,r],[r,b,r,b,b,r],[b,g,r,b,r,r],
               [r,b,r,r,b,r],[r,r,r,b,r,r]],\
              [[b,r,r,r,r,r],[r,r,b,r,r,r],[b,r,r,b,r,r],[r,g,b,r,r,r],
               [b,r,r,b,r,b],[r,r,b,r,b,r]],\
              [[r,r,b,b,b,b],[b,r,r,r,r,r],[r,r,b,b,r,b],[r,g,b,r,r,r],
               [r,b,r,r,r,r],[r,r,r,r,r,r]],\
              [[r,b,r,r,r,r],[r,b,r,b,b,r],[r,r,r,r,r,r],[b,g,r,b,r,b],
               [r,b,r,b,r,r],[r,r,r,r,b,r]],\
              [[b,r,b,r,r,b],[r,r,r,r,b,r],[r,b,r,b,r,r],[r,g,b,r,r,b],
               [r,b,r,r,r,r],[r,r,r,b,r,r]],\
              [[b,b,b,b,b,g],[b,r,r,r,b,r],[r,r,r,r,b,r],[r,r,r,b,r,r],
               [r,b,b,r,r,r],[r,r,r,r,r,r]],\
              [[r,r,r,b,r,g],[r,b,b,b,b,r],[r,r,r,b,r,r],[r,b,r,r,b,r],
               [r,r,b,r,b,r],[r,b,r,r,r,r]],\
              [[b,r,r,r,r,g],[r,b,r,r,b,r],[r,r,b,r,b,r],[b,r,b,r,r,b],
               [r,b,r,b,r,b],[r,r,r,r,r,r]],\
              [[r,b,r,r,r,g],[r,r,r,b,b,b],[r,r,r,r,r,r],[b,b,b,b,b,r],
               [r,r,r,b,r,r],[r,b,r,r,r,r]],\
              [[r,r,r,b,r,g],[r,b,r,r,r,b],[r,r,b,b,b,b],[r,r,b,r,r,r],
               [r,r,b,r,b,r],[r,r,r,r,b,r]],\
              [[r,r,b,r,b,g],[r,r,r,r,r,r],[r,b,r,b,b,r],[r,b,r,r,r,b],
               [b,r,r,b,b,b],[r,r,r,r,r,r]],\
              [[r,r,r,b,r,g],[r,r,r,b,r,b],[r,b,r,b,r,r],[r,b,r,r,r,r],
               [r,b,b,b,b,b],[r,r,r,r,r,r]],\
              [[r,r,b,r,r,g],[b,r,b,r,b,b],[r,r,r,r,r,r],[r,b,b,b,b,r],
               [r,r,r,r,b,r],[r,r,b,r,r,r]],\
              [[r,r,r,r,r,g],[b,r,b,b,r,b],[r,r,r,b,r,r],[r,b,r,b,b,r],
               [r,r,r,r,r,b],[r,r,b,r,b,r]],\
              [[r,r,b,r,r,g],[r,b,r,r,b,r],[r,r,r,b,r,b],[r,b,r,r,r,r],
               [b,b,b,b,b,r],[r,r,r,r,r,r]],\
              [[r,r,b,r,b,g],[b,r,r,r,r,r],[r,r,b,b,b,b],[r,r,r,r,r,r],
               [r,b,b,b,b,r],[r,r,r,r,r,r]],\
              [[b,r,b,b,r,g],[r,r,b,b,r,b],[b,r,r,r,r,r],[r,r,r,r,b,r],
               [r,b,r,b,r,r],[r,r,r,r,b,r]]])

def visualize_states(ax=None, states=None, tile_color=None, plot_size=None,
                     panels=None, **kwargs):
    """
    Supported kwargs:
        -tile_color: a dictionary from tiles (states) to colors
        -plot_size is an integer specifying how many tiles wide
            and high the plot is, with the grid itself in the middle
    """
    if tile_color is None:
        tile_color = {}
    
    if ax == None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    
    if panels is None:
        panels = []
    
    # plot squares
    for s in states:
        if s == (-1, -1):
            continue
        square = Rectangle(xy=s, width=1, height=1, color=tile_color.get(s, 'white'),
                           ec='k', lw=2)
        ax.add_patch(square)
    
    ax.axis('off')
    if plot_size is None and len(panels) == 0:
        ax.set_xlim(-0.1, 1 + max([s[0] for s in states]) + 0.1)
        ax.set_ylim(-0.1, 1 + max([s[1] for s in states]) + 0.1)
        ax.axis('scaled')

def visualize_trajectory(axis, traj,
                         jitter_mean=0,
                         jitter_var=.1,
                         plot_actions=False,
                         endpoint_jitter=False,
                         color='black',
                         **kwargs):

    traj = [(t[0], t[1]) for t in traj]  # traj only depends on state actions
#    print('length', len(traj),traj)

    if len(traj) == 2:
        p0 = tuple(np.array(traj[0][0]) + .5)
        p2 = tuple(np.array(traj[1][0]) + .5)
        p1 = np.array([(p0[0] + p2[0]) / 2, (p0[1] + p2[1]) / 2]) \
             + np.random.normal(0, jitter_var, 2)
        if endpoint_jitter:
            p0 = tuple(
                np.array(p0) + np.random.normal(jitter_mean, jitter_var, 2))
            p1 = tuple(
                np.array(p1) + np.random.normal(jitter_mean, jitter_var, 2))
        segments = [[p0, p1, p2], ]
    elif (len(traj) == 3) and (traj[0][0] == traj[2][0]):
        p0 = tuple(np.array(traj[0][0]) + .5)
        p2 = tuple(np.array(traj[1][0]) + .5)
        if abs(p0[0] - p2[0]) > 0:  # horizontal
            jitter = np.array(
                [0, np.random.normal(jitter_mean, jitter_var * 2)])
            p2 = p2 - np.array([.25, 0])
        else:  # vertical
            jitter = np.array(
                [np.random.normal(jitter_mean, jitter_var * 2), 0])
            p2 = p2 - np.array([0, .25])
        p1 = p2 + jitter
        p3 = p2 - jitter
        segments = [[p0, p1, p2], [p2, p3, p0]]
    else:
        state_coords = []
        for s, a in traj:
            jitter = np.random.normal(jitter_mean, jitter_var, 2)
            coord = np.array(s) + .5 + jitter
            state_coords.append(tuple(coord))
        if not endpoint_jitter:
            state_coords[0] = tuple(np.array(traj[0][0]) + .5)
            state_coords[-1] = tuple(np.array(traj[-1][0]) + .5)
        join_point = state_coords[0]
#        print('state_coords:',state_coords)
        segments = []
        for i, s in enumerate(state_coords[:-1]):
#            print(s)
            ns = state_coords[i + 1]

            segment = []
            segment.append(join_point)
            segment.append(s)
            if i < len(traj) - 2:
                join_point = tuple(np.mean([s, ns], axis=0))
                segment.append(join_point)
            else:
                segment.append(ns)
            segments.append(segment)
#    print(segments)

    for segment, step in zip(segments, traj[:-1]):
        action = step[1]

        codes = [Path.MOVETO, Path.CURVE3, Path.CURVE3]
        path = Path(segment, codes)
        patch = patches.PathPatch(path, facecolor='none', capstyle='butt',
                                  linewidth=2,edgecolor=color, **kwargs)
        axis.add_patch(patch)
        if plot_actions:
            dx = 0
            dy = 0
            if action == 2:
                dx = 1
            elif action == 3:
                dy = -1
            elif action == 1:
                dy = 1
            elif action == 0 :
                dx = -1
            action_arrow = patches.Arrow(segment[1][0], segment[1][1],
                                         dx * .4,
                                         dy * .4,
                                         width=.4,
                                         color='blue')
            axis.add_patch(action_arrow)
    return axis

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3
punish = -100.0
wall = -10.0

class SituationsEnv(discrete.DiscreteEnv):
    
    """
    This is the grid world environment from Sutton's Reinforcement leanring book
    You are an agent on an MxN grid and your goal is to reach the terminal
    state based on different situations
    
    for example, a 6x6 grid looks as follows:
        
    O O O O O G
    O O X X O O
    X O X X O O
    O O O O X X
    O O X O O O
    O O O O O O
    
    X is the block and G is the goal.
    
    You can take actions in each direction (UP=0, RIGHT=1, DOWN=2, LEFT=3).
    Actions going off the edge you in your current state.
    You receive a reward of -1 at 'O' state, and -30 at the 'X' state 
    until you reach a goal state.
    """
    
    metadata = {'render.modes': ['human', 'ansi']}
    
    def __init__(self, grid, shape=[6,6]):
        if not isinstance(shape, (list, tuple)) or not len(shape) == 2:
            raise ValueError('shape argument must be a list/tuple of length 2')
        
        self.shape = shape
        
        self.nS = np.prod(shape)
        self.nA = 4 # four actions
        
        MAX_Y = shape[0]
        MAX_X = shape[1]
        
        P = {}
        self.grid = grid
        it = np.nditer(self.grid, flags=['multi_index'])
        
        while not it.finished:
#            s = it.iterindex
            x,y = it.multi_index
            
            P[(x,y)] = {a : [] for a in range(self.nA)}
            
            is_done = lambda x,y : self.grid[x][y] == 3
            is_barrier = lambda x,y : self.grid[x][y] == 2
            is_wall = lambda x,y: x == y
            if is_done(x,y):
                reward = 0.0
            elif is_barrier(x,y):
                reward = punish
            else:
                reward = -1.0
#            print(x,y, self.grid[x][y])
            
            # We're stuck in a terminal state
            # P[s][a] is a list of transition tuples (prob, next_state, reward, done)
            if is_done(x,y):
                P[(x,y)][UP] = [(1.0, (x,y), reward, True)]
                P[(x,y)][RIGHT] = [(1.0, (x,y), reward, True)]
                P[(x,y)][DOWN] = [(1.0, (x,y), reward, True)]
                P[(x,y)][LEFT] = [(1.0, (x,y), reward, True)]
            # We have not reached the goal
            else:
                ns_up = (x, y) if x == 0 else (x - 1, y)
                if is_done(ns_up[0], ns_up[1]):
                    upreward = 0.0
                elif is_barrier(ns_up[0], ns_up[1]):
                    upreward = punish
                elif is_wall(ns_up, (x,y)):
                    upreward = wall
                else:
                    upreward = -1.0
                    
                ns_right = (x, y) if y == (MAX_Y - 1) else (x, y + 1)
                if is_done(ns_right[0], ns_right[1]):
                    rightreward = 0.0
                elif is_barrier(ns_right[0], ns_right[1]):
                    rightreward = punish
                elif is_wall(ns_right, (x,y)):
                    rightreward = wall
                else:
                    rightreward = -1.0
                    
                ns_down = (x, y) if x == (MAX_X - 1) else (x + 1, y)
                if is_done(ns_down[0], ns_down[1]):
                    downreward = 0.0
                elif is_barrier(ns_down[0], ns_down[1]):
                    downreward = punish
                elif is_wall(ns_down, (x,y)):
                    downreward = wall
                else:
                    downreward = -1.0
                    
                ns_left = (x, y) if y == 0 else (x, y - 1)
                if is_done(ns_left[0], ns_left[1]):
                    leftreward = 0.0
                elif is_barrier(ns_left[0], ns_left[1]):
                    leftreward = punish
                elif is_wall(ns_left, (x,y)):
                    leftreward = wall
                else:
                    leftreward = -1.0                   
                P[(x, y)][UP] = [(1.0, ns_up, upreward, is_done(ns_up[0],ns_up[1]))]
                P[(x, y)][RIGHT] = [(1.0, ns_right, rightreward, is_done(ns_right[0], ns_right[1]))]
                P[(x, y)][DOWN] = [(1.0, ns_down, downreward, is_done(ns_down[0], ns_down[1]))]
                P[(x, y)][LEFT] = [(1.0, ns_left, leftreward, is_done(ns_left[0], ns_left[1]))]
            
            it.iternext()
        
        # Initial state distribution is uniform
        isd = np.ones(self.nS)/self.nS
        
        # We expose the model of the environment for educational purposes
        # This should not be used in any model-free learning algorithm
        self.P = P
        
        super(SituationsEnv, self).__init__(self.nS, self.nA, self.P, isd)


def make_epsilon_greedy_policy(Q, epsilon, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function and epsilon.
    
    Args:
        Q: A dictionary that maps from state -> action-values.
            Each value is a numpy array of length nA (see below)
        epsilon: The probability to select a random action . float between 0 and 1.
        nA: Number of actions in the environment.
    
    Returns:
        A function that takes the observation as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.
    
    """
    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA
        best_action = np.argmax(Q[observation])
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn

"""
Find an optimal determinstic policy
"""
def QLearning(environment, episodes, discount_factor=0.95, init_state=(0, 0), 
              alpha=0.5, epsilon=0.6):
    """
    Q-Learning algorithm: Off-policy TD control. Finds the optimal greedy policy
    while following an epsilon-greedy policy
    
    Args:
        environment: chosen environment.
        episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        max_stepss:
        alpha: TD learning rate.
        epsilon: Chance the sample a random action. Float betwen 0 and 1.
    
    Returns:
        A tuple (Q, episode_lengths).
        Q is the optimal action-value function, a dictionary mapping state -> action values.
        stats is an EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """
    # The final action-value function.
    # A nested dictionary that maps state -> (action -> action-value).
    Q = defaultdict(lambda: np.zeros(environment.nA))
    op = {}
    
    # Keeps track of useful statistics
#    stats = plotting.EpisodeStats(
#        episode_lengths=np.zeros(num_episodes),
#        episode_rewards=np.zeros(num_episodes))
    
    # The policy we're following
    policy = make_epsilon_greedy_policy(Q, epsilon, environment.nA)
#    print(policy(init_state))
    for i_episode in range(episodes):
        # Print out which episode we're on, useful for debugging.
        if (i_episode + 1) % 100 == 0:
#            print("\rEpisode {}/{}.".format(i_episode + 1, episodes), end="")
            sys.stdout.flush()
        
        state = init_state
        
        for t in count():
            # Take a step
            action_probs = policy(state)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            transition = environment.P[state][action]
            prob, next_state, reward, done = transition[0]
            
            # TD Updata
            best_next_action = np.argmax(Q[next_state])
            td_target = reward + discount_factor * Q[next_state][best_next_action]
            td_delta = td_target - Q[state][action]
            Q[state][action] += alpha * td_delta
            
            if done:
#                print(i_episode, 'state is:', next_state)
                break
            state = next_state
    
    # Get the best action for each state
    for s,a_v in Q.items():
        op[s] = np.argmax(a_v)
    
    return Q, op
    
ranNum = random.randint(0,31)
random_grid = grid_array[ranNum]
env = SituationsEnv(grid=random_grid)

w = len(random_grid[0])
h = len(random_grid)
f_colors = {r: 'white', b: 'grey', g: 'red'}
state_features = {(x,y): random_grid[x][y] for x, y in product(range(w), range(w))}
t_colors = {s: f_colors[f] for s, f in state_features.items()}
tiles = list(product(range(h), range(w)))
fig = plt.figure(figsize=(8, 6))
ax1 = fig.add_subplot(111)
visualize_states(ax=ax1, states=tiles, tile_color=t_colors)
plt.savefig('output1.jpg')
img1 = cv2.imread('output1.jpg')
cv2.imshow('img1',img1)

m, op = QLearning(env,2000,alpha=0.2,epsilon=0.6)

"""Data output through windows"""
print('\nHello, my name is feifei, and I would like to help you find a way home')
while True:
    inputState = input('Please choose the initial state you want: ')
    if inputState == 'q':
        cv2.destroyAllWindows()
        print('Thank you and have a wonderful day')
        break
    else:
        s = ast.literal_eval(inputState)
        if len(s) != 2:
            print('Please input a tuple with length being equal to 2')
            continue
        else:
            if random_grid[s[0]][s[1]] == 2:
                print('Wow, you set the starting location on wall, unbelievable')
            fig2 = plt.figure(figsize=(8, 6))
            ax = fig2.add_subplot(111)
            visualize_states(ax=ax, states=tiles, tile_color=t_colors)
            traj = []
            for _ in range(500):
                a = op[s]
                temp = env.P[s][a]
                ns = temp[0][1] # gw.transition(s, a)
                traj.append((s, a, ns))
                is_terminal = lambda x,y: random_grid[x][y] == 3
                if is_terminal(ns[0],ns[1]):
                    traj.append((ns, a, ns))
                    break
                s = ns
            output = visualize_trajectory(ax, traj, plot_actions=True)
            plt.savefig('output2.jpg')
            img2 = cv2.imread('output2.jpg')
            cv2.imshow('img2',img2)
#            newinput = input('Please give me some action: ')
#            if newinput == 't':
            cv2.waitKey(0)
            cv2.destroyWindow('img2')
