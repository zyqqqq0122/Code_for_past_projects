# EL2805 Lab2 The Lunar Lander -- Problem 2

import numpy as np
import gym
import random
from tqdm import trange
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import math
import copy
from matplotlib.font_manager import json_load
from mpl_toolkits import mplot3d

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T




#==================================== Hyper Parameters ===========================#

# Max size of memory buffer
BUFF_SIZE = 30000

# Discount factor
GAMMA = 0.99

# Target networks update constant τ
TAU = 1e-3

# Policy update frequency d
D = 2

# Training parameters
BATCH_SIZE = 64
# Learning rate for actor network
LR_ACT = 5e-5
# Learning rate for critic network
LR_CRT = 5e-4

# Set random seed
SEED = 0

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



#================================= Actor and Critic Networks ===========================#

class Critic(nn.Module):
    '''
    Critic Network
    Inputs:
    input_size = dim_state + dim_action
    output_size = dim_action
    seed -- the random seed
    '''
    def __init__(self, input_size):
        super(Critic, self).__init__()
        self.linear1 = nn.Linear(input_size, 400)
        self.linear2 = nn.Linear(400, 200)
        self.linear3 = nn.Linear(200, 1)

    def forward(self, state, action):
        '''
        Inputs:
        state and actions are torch tensors
        '''
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)

        return x


class Actor(nn.Module):
    '''
    Actor Network
    Inputs:
    dim_state -- dimension of each state
    dim_action -- dimension of each action
    seed -- the random seed
    '''
    def __init__(self, dim_state, dim_action):
        super(Actor, self).__init__()
        self.linear1 = nn.Linear(dim_state, 400)
        self.linear2 = nn.Linear(400, 200)
        self.linear3 = nn.Linear(200, dim_action)
        
    def forward(self, state):
        '''
        Inputs:
        state is torch tensor
        '''
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        # tanh activation to constraint the output action to be between [−1, 1]
        x = torch.tanh(self.linear3(x))

        return x




#================================== Replace Memory =================================#

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
        action = torch.tensor(np.array(action)).float().to(device)    # [batch_size, 2]
        reward = torch.tensor(np.array(reward)).float().unsqueeze(1).to(device)   # [batch_size, 1]
        next_state = torch.tensor(np.array(next_state)).float().to(device)    # [batch_size, 8]
        done = torch.tensor(np.array(done).astype(np.uint8)).float().unsqueeze(1).to(device)    # [batch_size, 1]
        return (state, action, reward, next_state, done)
    
    def __len__(self):
        return len(self.memory)




#================================ Ornstein-Uhlenbeck Agent ============================#

class OU_Agent(object):
    '''
    Agent for Ornstein-Uhlenbeck process, adding coloured noise to the action
    '''
    def __init__(self, action_space, mu=0., theta=0.15, sigma=0.2):
        ''' Initialize parameters and noise process '''
        self.mu = mu * np.ones(action_space.shape[0])
        self.dim = action_space.shape[0]
        self.theta = theta
        self.sigma = sigma
        self.action_high = action_space.high
        self.action_low = action_space.low
        self.reset()

    def reset(self):
        ''' Reset the internal state (= noise) to mean (mu) '''
        self.state = copy.copy(self.mu)

    def sample(self):
        ''' Update internal state and return it as a noise sample '''
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.dim)
        self.state = x + dx
        return self.state

    def add_noise(self, action):
        ''' Add noise to the action and clip the result within the possible range of action '''
        noise = self.sample()
        return np.clip(action + noise, self.action_low, self.action_high)



#================================= DDPG Agent ===============================#

class DDPG_Agent(object):
    ''' 
    DDPG Agent
    Inputs:
    n_states -- total number of each state
    n_actions -- total number of actions
    seed -- the random seed
    Functions:
    
    '''
    global BUFF_SIZE, GAMMA, TAU, D, BATCH_SIZE, LR_ACT, LR_CRT
    
    def __init__(self, dim_state: int, dim_action: int, seed):
        self.dim_state = dim_state
        self.dim_action = dim_action
        self.seed = random.seed(seed)

        # Initialize the actor/actor_target neural network
        self.actor = Actor(dim_state, dim_action).to(device)
        self.actor_tgt = Actor(dim_state, dim_action).to(device)
        self.actor_tgt.load_state_dict(self.actor.state_dict())
        self.actor_tgt.eval()

        # Initialize the critic/critic_target neural network
        self.critic = Critic(dim_state+dim_action).to(device)
        self.critic_tgt = Critic(dim_state+dim_action).to(device)
        self.critic_tgt.load_state_dict(self.critic.state_dict())
        self.critic_tgt.eval()

        # Initialize the memory buffer with capacity of BUFF_SIZE
        self.memory = ReplayMemory(BUFF_SIZE)

        # Define the optimizer for Actor and Critic Network
        self.opt_act = optim.Adam(self.actor.parameters(), lr=LR_ACT)
        self.opt_crt = optim.Adam(self.critic.parameters(), lr=LR_CRT)

        # Initialize the taken steps for the updating of target networks
        self.steps_done = 0


    def take_step(self, state, action, reward, next_state, done):
        # Save current experience to the memory
        self.memory.push(state, action, reward, next_state, done)
        # Update value of steps_done
        self.steps_done += 1
        # Sample a random batch and train the policy DQN when there's enough experience in the buffer
        loss_crt = 0
        loss_act = 0
        if len(self.memory) > BATCH_SIZE:   
            experience = self.memory.sample(BATCH_SIZE)
            # loss_crt, loss_act = self.train_batch(experience)
            loss_crt = self.train_batch(experience)
        # return loss_crt, loss_act
        return loss_crt

    def get_action(self, state):
        ''' The returned 'action' is a numpy scalar '''
        self.actor.eval()
        with torch.no_grad():
            # Take the output of actor network as action
            state = torch.tensor(state).unsqueeze(0).to(device)
            action = self.actor(state).cpu().data.numpy().squeeze(0)   
        return action

    def train_batch(self, experience):
        # Set actor and critic networks to train mode
        self.actor.train()
        self.critic.train()

        # Clean accumulated gradients
        self.opt_crt.zero_grad()

        # Converts batch array of Experience to Experience of batch aray
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = experience
        
        # Compute expected Q values
        next_action_batch = self.actor_tgt(next_state_batch)
        next_state_Qvalues = self.critic_tgt(next_state_batch, next_action_batch).detach()    # [batch_size, 1]
        Qvalues_expected = reward_batch + GAMMA*(1 - done_batch)*next_state_Qvalues    # [batch_size, 1]
        
        # Compute output Q values from the taget and policy network
        Qvalues_output = self.critic(state_batch, action_batch)     # [batch_size, 2]
        # Q_j = self.critic(state_batch, self.actor(state_batch)).detach()

        # Compute critic loss
        crt_loss = F.mse_loss(Qvalues_output, Qvalues_expected)
        # act_loss = -F.l1_loss(torch.zeros_like(Q_j, requires_grad=True), Q_j)
        

        # Critic loss backward
        crt_loss.backward()
        # act_loss = 0
        
        # Clip the gradients and Update critic network
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1)
        self.opt_crt.step()
        
        # Update actor network and Soft-update target networks
        if self.steps_done%D == 0:
            # Clean accumulated gradients
            self.opt_act.zero_grad()
            # Actor loss backward
            self.critic.eval()
            act_loss = -self.critic(state_batch, self.actor(state_batch)).mean()
            self.critic.train()
            act_loss.backward()
            # Clip the gradients and Update actor network
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1)
            self.opt_act.step()
            self.soft_update(self.actor, self.actor_tgt, TAU)   
            self.soft_update(self.critic, self.critic_tgt, TAU)
        
        return crt_loss.detach().cpu()#, act_loss.detach().cpu()


    def soft_update(self, policy_net, target_net, tau):
        ''' 
        Performs a soft copy of the network's parameters to the target network's parameter
        Args:
        policy_net (nn.Module): neural network from which we want to copy the parameters
        target_net (nn.Module): network that is being updated
        tau (float): time constant that defines the update speed in (0,1)
        Returns:
        target_network (nn.Module): the target network
        '''
        tgt_state = target_net.state_dict()
        for k, v in policy_net.state_dict().items():
            tgt_state[k] = (1 - tau)  * tgt_state[k]  + tau * v
        target_net.load_state_dict(tgt_state)
        return target_net


class RandomAgent():
    ''' Agent taking actions uniformly at random, child of the class Agent'''
    def __init__(self, n_actions: int):
        self.n_actions = n_actions

    def forward(self, state: np.ndarray) -> np.ndarray:
        ''' Compute a random action in [-1, 1]

            Returns:
                action (np.ndarray): array of float values containing the
                    action. The dimensionality is equal to self.n_actions from
                    the parent class Agent.
        '''
        return np.clip(-1 + 2 * np.random.rand(self.n_actions), -1, 1)



#==================================== Model Training ===============================#

def train_DDPG(env, dim_state, dim_action, N_episodes, max_steps, savepath_act, savepath_crt, savepath_data):
    global SEED

    # We will use these variables to compute the average episodic reward and the average number of steps per episode
    episode_reward_list = []       # this list contains the total reward per episode
    episode_number_of_steps = []   # this list contains the number of steps per episode
    loss_crt = []
    loss_act = []

    # Agent initialization
    agent = DDPG_Agent(dim_state, dim_action, SEED)
    noise = OU_Agent(env.action_space)

    # trange is an alternative to range in python. It shows a nice progression bar that you can update with useful information
    EPISODES = trange(N_episodes, desc='Episode: ', leave=True)
    
    # Training process
    for i in EPISODES:
        # Reset enviroment data and initialize variables
        done = False
        state = env.reset()
        total_episode_reward = 0.
        loss_episode_crt = 0.
        loss_episode_act = 0.
        for step in range(max_steps):
            # Agent takes a random action
            action = agent.get_action(state)
            noised_action = noise.add_noise(action)
            # Get next state and reward
            # print(action)
            # print(noised_action)
            # print(noised_action.shape)
            next_state, reward, done, _ = env.step(noised_action)
            # Agent takes a step
            # loss_step_crt, loss_step_act = agent.take_step(state, noised_action, reward, next_state, done)
            loss_step_crt = agent.take_step(state, noised_action, reward, next_state, done)
            loss_episode_crt += loss_step_crt
            # loss_episode_act += loss_step_act
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
        loss_crt.append(loss_episode_crt)
        loss_act.append(loss_episode_act)
        # Stop training and save the trained actor and critic network if the average score in 50 episodes is >=200
        if np.mean(episode_reward_list[-50:]) >= 200.:
            print('\nEnvironment solved! Average Score: {:.2f}'.format(np.mean(episode_reward_list[-50:])))
            torch.save(agent.actor, savepath_act)
            torch.save(agent.critic, savepath_crt)
            data = {'reward': episode_reward_list, 'step': episode_number_of_steps}
            torch.save(data, savepath_data)
            break
        
        # Updates the tqdm update bar with fresh information
        EPISODES.set_description("Episode {} - Reward/Steps: {:.1f}/{}".format(i, total_episode_reward, step+1))


    return episode_reward_list, episode_number_of_steps, loss_crt, loss_act


# Import and initialize the continuous Lunar Laner Environment
env = gym.make('LunarLanderContinuous-v2')
env.seed(seed = SEED)

# Parameters
# State dimensionality
# dim_state = len(env.observation_space.high)  
dim_state = env.observation_space.shape[0]
# Number of available actions
dim_action = env.action_space.shape[0]     
# Max number of training episodes          
N_episodes = 400
# Max number of steps in each episode
max_steps = 1000
# Path for saving success trained model
savepath_act = "neural-network-2-actor.pth"
savepath_crt = "neural-network-2-critic.pth"
savepath_data = "2-data.pt"

# Training
episode_reward_list, episode_number_of_steps, loss_crt, loss_act = train_DDPG(env, dim_state, dim_action,
                                                                  N_episodes, max_steps, 
                                                                  savepath_act, savepath_crt, savepath_data)

# Close environment
env.close()




#================================== Plot results =============================#

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
    actor = torch.load("neural-network-2-actor.pth")
    critic = torch.load("neural-network-2-critic.pth")
    actor.eval()
    critic.eval()
except:
    print('File neural-network-2.pth not found!')
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
            action = actor(state)
            Qvalues[i, j] = critic(state, action).detach().numpy()
            Avalues[i, j] = action.detach().numpy()[:, 1]
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
ax.set_title('Engine values of restricted state region')

