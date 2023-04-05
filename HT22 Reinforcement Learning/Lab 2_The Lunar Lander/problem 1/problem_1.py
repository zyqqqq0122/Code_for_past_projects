

# EL2805 Lab2 The Lunar Lander -- Problem 1


import numpy as np
import gym
import random
from tqdm import trange
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import math
from matplotlib.font_manager import json_load
from mpl_toolkits import mplot3d

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T



#=================================  Hyper-Parameters ================================#

# Max size of memory buffer
BUFF_SIZE = 100000

# Discount factor
GAMMA = 0.99

# Epsilon greedy parameters (eps is changing in different episodes)
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200

# Training parameters
BATCH_SIZE = 64
LR = 1e-3
TARGET_UPDATE = 5

# Set random seed
SEED = 0

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



#======================================== Q-Network ==================================#

class DQN(nn.Module):
    '''
    Deep Q-Network
    Inputs:
    state_dim -- dimension of each state
    n_actions -- total number of actions
    seed -- the random seed
    '''
    def __init__(self, dim_state, n_actions, seed):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(dim_state, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, n_actions)  
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x



#==================================== Replace Memory ================================#

# Define the Experience tuple to save states, action, reward and done status
Experience = namedtuple('Experience',
                        ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayMemory(object):
    '''
    Memory buffer
    Inputs: capacity -- the size of memory buffer
    Functions: 
    push -- save a memory to the buffer
    sample -- sample a batch of memory from the buffer
    len -- return the lenght of the experience
    '''
    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        self.memory.append(Experience(*args))

    def sample(self, batch_size):
        indices = np.random.choice(len(self.memory), batch_size, replace=False)
        state, action, reward, next_state, done = zip(*[self.memory[idx] for idx in indices])
        state = torch.tensor(np.array(state)).float().to(device)    # [batch_size, 8]
        action = torch.tensor(np.array(action)).long().unsqueeze(1).to(device)    # [batch_size, 1]
        reward = torch.tensor(np.array(reward)).float().unsqueeze(1).to(device)   # [batch_size, 1]
        next_state = torch.tensor(np.array(next_state)).float().to(device)    # [batch_size, 8]
        done = torch.tensor(np.array(done).astype(np.uint8)).float().unsqueeze(1).to(device)    # [batch_size, 1]
        return (state, action, reward, next_state, done)
    
    def __len__(self):
        return len(self.memory)



#====================================== DQN Agent ===============================#

class DQN_Agent(object):
    ''' 
    DQN Agent, fetches Q value from DQN and takes action to gather next state and reward fromt the Environment
    Inputs:
    dim_state -- dimension of each state
    n_actions -- total number of actions
    seed -- the random seed
    Functions:
    take_step -- main function, push experience into buffer and train the policy DQN when memory can fill a batch
                 by calling function train_batch
    select_action -- select an action with epsilon greedy
    train_batch -- update the policy DQN in every step after enough experience is accumulated
                   update the target DQN only every #TARGET_UPDATE steps
    '''
    global BUFF_SIZE, GAMMA, EPS_START, EPS_END, EPS_DECAY, BATCH_SIZE, LR, TARGET_UPDATE
    
    def __init__(self, dim_state: int, n_actions: int, seed):
        self.dim_state = dim_state
        self.n_actions = n_actions
        self.seed = random.seed(seed)
        # Initialize the main/target neural network
        self.policy_net = DQN(dim_state, n_actions, seed).to(device)
        self.target_net = DQN(dim_state, n_actions, seed).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        # Initialize the memory buffer with capacity of BUFF_SIZE
        self.memory = ReplayMemory(BUFF_SIZE)
        # Define the optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
        # Initialize the taken steps for the updating of target DQN
        self.steps_done = 0

    def take_step(self, state, action, reward, next_state, done):
        # Save current experience to the memory
        self.memory.push(state, action, reward, next_state, done)
        # Update value of steps_done
        self.steps_done += 1
        # Sample a random batch and train the policy DQN when there's enough experience in the buffer
        loss = 0
        if len(self.memory) > BATCH_SIZE:
            experience = self.memory.sample(BATCH_SIZE)
            loss = self.train_batch(experience)
        return loss

    def select_action(self, state, TEST=False):
        if TEST == False:
            # Epsilon value changes in each episode when learning
            eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * self.steps_done / EPS_DECAY)
        else:
            # In testing, always choose the best action, no randomness
            eps_threshold = 0
        # Take epsilon greedy action
        if random.random() > eps_threshold:
            self.policy_net.eval()
            with torch.no_grad():
                # Pick action with the largest expected reward
                state = torch.tensor(state).unsqueeze(0).to(device)   
            return np.argmax(self.policy_net(state).cpu().data.numpy())
        else:
            return random.choice(np.arange(self.n_actions))

    def train_batch(self, experience):
        # Set policy DQN to train mode
        self.policy_net.train()
        # Clean accumulated gradients
        self.optimizer.zero_grad()
        # Converts batch array of Experience to Experience of batch aray
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = experience
        # Compute expected Q values
        next_state_Qvalues = self.target_net(next_state_batch).max(1)[0].detach().unsqueeze(1)    # [batch_size, 1]
        Qvalues_expected = reward_batch + GAMMA*(1 - done_batch)*next_state_Qvalues    # [batch_size, 1]
        # Compute output Q values from the policy network
        Qvalues_output = self.policy_net(state_batch).gather(1, action_batch)   # [batch_size, 4] -> [batch_size, 1]
        # Compute MSE loss
        loss = F.mse_loss(Qvalues_output, Qvalues_expected)
        loss.backward()
        # Clip the gradient
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1)
        self.optimizer.step()
        # Update target network
        if self.steps_done % TARGET_UPDATE == 0:
             self.target_net.load_state_dict(self.policy_net.state_dict())
        return loss.detach().cpu()


class RandomAgent():
    ''' Agent taking actions uniformly at random, child of the class Agent'''
    def __init__(self, n_actions: int):
        self.n_actions = n_actions

    def forward(self, state: np.ndarray) -> int:
        ''' Compute an action uniformly at random across n_actions possible
            choices

            Returns:
                action (int): the random action
        '''
        self.last_action = np.random.randint(0, self.n_actions)
        return self.last_action



#================================== Model Training =============================#

def train_DQN(env, dim_state, n_actions, N_episodes, max_steps, save_path, savepath_data):
    global SEED

    # We will use these variables to compute the average episodic reward and the average number of steps per episode
    episode_reward_list = []       # this list contains the total reward per episode
    episode_number_of_steps = []   # this list contains the number of steps per episode
    loss = []

    # Random agent initialization
    agent = DQN_Agent(dim_state, n_actions, SEED)
    
    # trange is an alternative to range in python. It shows a nice progression bar that you can update with useful information
    EPISODES = trange(N_episodes, desc='Episode: ', leave=True)
    
    # Training process
    for i in EPISODES:
        # Reset enviroment data and initialize variables
        done = False
        state = env.reset()
        total_episode_reward = 0.
        loss_episode = 0.
        for step in range(max_steps):
            # Agent takes a random action
            action = agent.select_action(state)
            # Get next state and reward
            next_state, reward, done, _ = env.step(action)
            # Agent takes a step
            loss_episode += agent.take_step(state, action, reward, next_state, done)
            # Update episode reward
            total_episode_reward += reward
            # Update state for next iteration
            state = next_state
            # Stop exploring in this episode if done=True
            if done == True:
                break
        # Append episode reward and total number of steps
        episode_reward_list.append(total_episode_reward)
        episode_number_of_steps.append(step+1)
        loss.append(loss_episode)
        # Stop training and save the trained DQN if the score is >=200
        if np.mean(episode_reward_list[-100:]) >= 200.:
            print('\nEnvironment solved! Average Score: {:.2f}'.format(np.mean(episode_reward_list[-100:])))
            torch.save(agent.policy_net, save_path)
            data = {'reward': episode_reward_list, 'step': episode_number_of_steps}
            torch.save(data, savepath_data)
            break
            
        # Updates the tqdm update bar with fresh information
        EPISODES.set_description("Episode {} - Reward/Steps: {:.1f}/{}".format(i, total_episode_reward, step+1))
    
    return episode_reward_list, episode_number_of_steps, loss


# Import and initialize the discrete Lunar Laner Environment
env = gym.make('LunarLander-v2')
env.seed(seed = SEED)

# Parameters
# State dimensionality
dim_state = len(env.observation_space.high)  
# Number of available actions
n_actions = env.action_space.n     
# Max number of training episodes          
N_episodes = 1000
# Max number of steps in each episode
max_steps = 1000
# Path for saving success trained model
save_path = "neural-network-1.pth"
savepath_data = "1-data.pt"

# Training
episode_reward_list, episode_number_of_steps, loss = train_DQN(env, dim_state, n_actions, N_episodes, max_steps, save_path, savepath_data)

# Close environment
env.close()



#=================================== Plot results ============================#

def running_average(x, N):
    ''' Function used to compute the running average
        of the last N elements of a vector x
    '''
    if len(x) >= N:
        y = np.copy(x)
        y[N-1:] = np.convolve(x, np.ones((N, )) / N, mode='valid')
    else:
        y = np.zeros_like(x)
    return y


n_ep_running_average = 50

# Load saved training data
data = torch.load(savepath_data)
episode_reward_list = data['reward']
episode_number_of_steps = data['step']

# Plot Rewards and steps
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 9))
ax[0].plot(episode_reward_list, label='Episode reward')
ax[0].plot(running_average(episode_reward_list, n_ep_running_average), label='Avg. episode reward')
ax[0].set_xlabel('Episodes')
ax[0].set_ylabel('Total reward')
ax[0].set_title('Total Reward vs Episodes')
ax[0].legend()
ax[0].grid(alpha=0.3)

ax[1].plot(episode_number_of_steps, label='Steps per episode')
ax[1].plot(running_average(episode_number_of_steps, n_ep_running_average), label='Avg. number of steps per episode')
ax[1].set_xlabel('Episodes')
ax[1].set_ylabel('Total number of steps')
ax[1].set_title('Total number of steps vs Episodes')
ax[1].legend()
ax[1].grid(alpha=0.3)
plt.show()



# Load model
try:
    model = torch.load("neural-network-1-actor.pth")
    model.eval()
except:
    print('File neural-network-1.pth not found!')
    exit(-1)


Xvalue = torch.linspace(0, 1.5, steps=100)
Yvalue = torch.linspace(-math.pi, math.pi, steps=100)
Qvalues = np.zeros((100, 100))
Avalues = np.zeros((100, 100))
X = np.zeros((100, 100))
Y = np.zeros((100, 100))

with torch.no_grad():
    i = 0
    for i in range(100):
        for j in range(100):
            state = torch.zeros(1, 8)
            state[:,1] = Xvalue[i]
            state[:, 4] = Yvalue[j]
            Qvalues[i, j] = np.max(model(state).detach().numpy(), axis=1)
            Avalues[i, j] = np.argmax(model(state).detach().numpy())
            X[i, j] = Xvalue[i]
            Y[i, j] = Yvalue[j]

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(Y, X, Qvalues, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
ax.set_title('Q values of restricted state region')


fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(Y, X, Avalues, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
ax.set_title('Action values of restricted state region')