# Authors:
# Yuqi Zheng 990122-1243 yuqizh@kth.se
# Jingxuan Mao 001214-9068 jmao@kth.se

import numpy as np
import my_maze as mz
import matplotlib.pyplot as plt
from tqdm import trange

# Problem 1: The Maze and the Random Minotaur
 
#Initiate maze environment for player(P)
maze = np.array([
    [0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0, 1, 1, 1],
    [0, 0, 1, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 0],
    [0, 0, 0, 0, 1, 2, 0, 0]
])

# mz.draw_maze(maze)

# Dynamic Programming
def prob_exit(n_episode, STAY):
    env = mz.Maze(maze, stay=STAY)
    y_prob = []

    Tvalue = trange(15, 30, desc='Tvalue: ', leave=True)
    for T in Tvalue:
        # print(f'Current T:{T}')
        # Initialization
        n_success = 0
        for i in range(n_episode):
            # Solve the MDP problem with dynamic programming
            _, policy = mz.dynamic_programming(env,T)
            # Simulate the shortest path starting from position A with dynamic programming
            method = 'DynProg'
            start  = ((0,0),(6,5))  # ((Player_pose),(Minotaur_pose))
            path = env.simulate(start, policy, method) # Generate the path for the player and the minotaur

            # Count number of successful trails
            for k in range(len(path)):
                if path[k][0] == path[k][1]:
                    break
                if path[k][0] == (6, 5):
                    n_success += 1
                    break
            if i%30 == 29:
                print('Finished 30 episodes')
        y_prob.append(n_success/n_episode)

        Tvalue.set_description("T value {} - Success rate: {:.2f}".format(T+1, n_success/n_episode))

    return y_prob


n_episode = 1000
y_prob_nostay = prob_exit(n_episode, STAY=False)
y_prob_stay = prob_exit(n_episode, STAY=True)

x_horizon = np.arange(1, 31)
plt.scatter(x_horizon, y_prob_nostay)
plt.title('Probability of exiting the maze when minotaur cannot stay')
plt.xlabel('T')
plt.ylabel('Probability')
plt.show()

plt.scatter(x_horizon, y_prob_stay)
plt.title('Probability of exiting the maze when minotaur can stay')
plt.xlabel('T')
plt.ylabel('Probability')
plt.show()




# Value Iteration
STAY = False
env = mz.Maze(maze, stay=STAY)

# Calculate the time parameters for infinite value iteration
gamma = 1 - 1/30    # Discount factor
epsilon = 1e-5     # Accuracy threshold

# Estimate the exiting probability of the policy by simulating 10000 games
n_sim = 10000
n_exit = 0
method = 'ValIter'
start  = ((0,0),(6,5))

T_lim = np.random.geometric(1/30, n_sim)
for t in T_lim:
    path = env.simulate(start, policy, method) # Generate the path for the player and the minotaur
    for k in range(t):
        if path[k][0] == (6, 5):
            n_exit += 1
            break

prob = n_exit / n_sim
print(f'The probability of exiting the maze when minotaur cannot stay is {prob}')


STAY = True
env = mz.Maze(maze, stay=STAY)

# Calculate the time parameters for infinite value iteration
gamma = 1 - 1/30    # Discount factor
epsilon = 1e-5     # Accuracy threshold

# Estimate the exiting probability of the policy by simulating 10000 games
n_sim = 10000
n_exit = 0
method = 'ValIter'
start  = ((0,0),(6,5))

T_lim = np.random.geometric(1/30, n_sim)
for t in T_lim:
    path = env.simulate(start, policy, method) # Generate the path for the player and the minotaur
    for k in range(t):
        if path[k][0] == (6, 5):
            n_exit += 1
            break

prob = n_exit / n_sim
print(f'The probability of exiting the maze when minotaur can stay is {prob}')




