# EL2805 Lab2 The Lunar Lander -- Problem 3

import numpy as np
import gym
import random
from tqdm import trange
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import math
from scipy.stats import multivariate_normal
from matplotlib.font_manager import json_load
from mpl_toolkits import mplot3d


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch.distributions.multivariate_normal import MultivariateNormal



#================================ Hyper Parameters ==========================#

# Discount factor
GAMMA = 0.99

# Epsilon value for clipping function
EPS = 0.2

# Training parameters
EPOCH = 10
# Learning rate for actor network
LR_ACT = 1e-5
# Learning rate for critic network
LR_CRT = 1e-3

# Set random seed
SEED = 0

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




#===================================== Actor and Critic Networks ===============================#

class Critic(nn.Module):
    '''
    Critic Network
    Inputs:
    dim_state -- dimension of each state
    '''
    def __init__(self, dim_state):
        super(Critic, self).__init__()
        self.linear1 = nn.Linear(dim_state, 400)
        self.linear2 = nn.Linear(400, 200)
        self.linear3 = nn.Linear(200, 1)

    def forward(self, state):
        '''
        Inputs:
        state is torch tensor
        '''
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)

        return x


class Actor(nn.Module):
    '''
    Actor Network
    Inputs:
    dim_state -- dimension of each state
    dim_action -- dimension of each action
    '''
    def __init__(self, dim_state, dim_action):
        super(Actor, self).__init__()
        self.linear1 = nn.Linear(dim_state, 400)
        self.mu_head = nn.Sequential(nn.Linear(400, 200),
                                    nn.ReLU(),
                                    nn.Linear(200, dim_action),
                                    nn.Tanh())
        self.sigma_head = nn.Sequential(nn.Linear(400, 200),
                                    nn.ReLU(),
                                    nn.Linear(200, dim_action),
                                    nn.Sigmoid())

    def forward(self, state):
        '''
        Inputs:
        state is torch tensor
        '''
        x = F.relu(self.linear1(state))
        mu = self.mu_head(x)
        sigma = self.sigma_head(x)
        return mu, sigma




#========================================= Replace Memory =====================================#

# Define the Experience tuple to save states, action, reward, old probs and done status
Experience = namedtuple('Experience',
                        ('state', 'action', 'reward', 'prob', 'done'))

class ReplayMemory(object):
    '''
    Memory buffer
    Functions: 
    push -- save a memory to the buffer
    sample -- pop all memory in the buffer
    clear -- clear all memory in the buffer
    len -- return the lenght of the experience
    '''
    def __init__(self):
        self.memory = deque()

    def push(self, *args):
        self.memory.append(Experience(*args))

    def sample(self):
        state, action, reward, prob, done = zip(*(self.memory))
        state = torch.tensor(np.array(state)).float().to(device)    # [batch_size, 8]
        action = torch.tensor(np.array(action)).float().to(device)    # [batch_size, 2]
        reward = torch.tensor(np.array(reward)).float().unsqueeze(1).to(device)   # [batch_size, 1]
        prob = torch.tensor(np.array(prob)).float().unsqueeze(1).to(device)    # [batch_size, 1]
        done = torch.tensor(np.array(done).astype(np.uint8)).float().unsqueeze(1).to(device)    # [batch_size, 1]
        return (state, action, reward, prob, done)
    
    def clear(self):
        self.memory = deque()
    
    def __len__(self):
        return len(self.memory)




#========================================= PPO Agent =====================================#

class PPO_Agent(object):
    ''' 
    PPO Agent
    Inputs:
    n_states -- total number of each state
    n_actions -- total number of actions
    seed -- the random seed
    Functions:
    
    '''
    global GAMMA, EPS, EPOCH, LR_ACT, LR_CRT, device
    
    def __init__(self, dim_state: int, dim_action: int, seed):
        self.dim_state = dim_state
        self.dim_action = dim_action
        self.seed = random.seed(seed)

        # Initialize the actor and critic neural network
        self.actor = Actor(dim_state, dim_action).to(device)
        self.critic = Critic(dim_state).to(device)

        # Initialize the memory buffer with capacity of BUFF_SIZE
        self.memory = ReplayMemory()

        # Define the optimizer for Actor and Critic Network
        self.opt_act = optim.Adam(self.actor.parameters(), lr=LR_ACT)
        self.opt_crt = optim.Adam(self.critic.parameters(), lr=LR_CRT)

        # Initialize the taken steps for the updating of target networks
        self.steps_done = 0


    def take_step(self, state, action, reward, prob, done):
        # Save current experience to the memory
        self.memory.push(state, action, reward, prob, done)
        # Update value of steps_done
        self.steps_done += 1
        

    def get_action(self, state):
        ''' The returned 'action' is a numpy scalar '''
        self.actor.eval()
        with torch.no_grad():
            # Take the output of actor network as action
            state = torch.tensor(state).unsqueeze(0).to(device)
            mu, sigma = self.actor(state)
            # Generate a random action from the distribution
            # print(torch.diag(sigma.squeeze(0)))
            dist = MultivariateNormal(mu.squeeze(0), torch.diag(sigma.squeeze(0)))
            action = dist.sample() 
            prob = dist.log_prob(action).item()
            action = action.cpu().numpy()
        return action, prob


    def get_target(self, step):
        # Converts batch array of Experience to Experience of batch aray
        experience = self.memory.sample()
        state_batch, action_batch, reward_batch, old_prob_batch, done_batch = experience

        # Compute G values
        batch_size = self.memory.__len__()           
        Gvalues = []
        for i in range(batch_size):
            y_sum = 0 
            for k in range(i, step):
                y_sum += GAMMA**(k-i) * reward_batch[k] 
            Gvalues.append(y_sum)
        Gvalues = torch.tensor(Gvalues, requires_grad=False, dtype=torch.float32).unsqueeze(dim=-1).to(device)

        # Compute advantage estimation Psi
        self.critic.eval()
        with torch.no_grad():
            Psi = Gvalues - self.critic(state_batch).detach()   # [batch_size, 1]
        Psi = Psi.to(device)
        return Gvalues, Psi
        

    def train_epochs(self, Gvalues, Psi):
        # Converts batch array of Experience to Experience of batch aray
        experience = self.memory.sample()
        state_batch, action_batch, reward_batch, old_prob_batch, done_batch = experience
        # print(state_batch.shape)
        old_prob_batch = old_prob_batch.squeeze(1)

        # Train the networks for epoches
        crt_loss_epochs = 0.
        act_loss_epochs = 0.
        for epoch in range(EPOCH):
            # Set actor and critic networks to train mode
            self.actor.train()
            self.critic.train()
            
            # Compute critic loss
            batch_size = self.memory.__len__()
            # print(batch_size)
            Vvalues= self.critic(state_batch)     # [batch_size, 1]
            crt_loss = F.mse_loss(Vvalues, Gvalues)           
            
            # Compute actor loss
            mus, sigmas = self.actor(state_batch)
            act_loss = torch.zeros(1).to(device)
            for i in range(batch_size):
                actorDist = MultivariateNormal(mus[i], torch.diag(sigmas[i].squeeze(0))) # Get actor distribution for mu_i & var_i
                action_prob = actorDist.log_prob(action_batch[i]) # Get prob. of the action for new actor distribution
                r_theta = torch.exp(action_prob - old_prob_batch[i]).to(device) # pi_theta / pi_theta_old
                c_epsilon = torch.clamp(r_theta, 1-EPS, 1+EPS).to(device) # clipping function
                act_loss += torch.min(r_theta*Psi[i], c_epsilon*Psi[i])
            act_loss = (-1 / torch.tensor(batch_size)) * act_loss

            # Update critic network
            self.opt_crt.zero_grad()
            crt_loss.backward()
            self.opt_crt.step()
            
            # Update actor network
            self.opt_act.zero_grad()
            act_loss.backward()
            self.opt_act.step()          

            # Average critic and actor loss in all epochs
            crt_loss_epochs += crt_loss.detach().cpu()/EPOCH
            act_loss_epochs += act_loss.detach().cpu()/EPOCH
        
        # Clear memory buffer
        self.memory.clear()

        return crt_loss_epochs, act_loss_epochs


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




#===================================== Model Training ===================================#

def train_PPO(env, dim_state, dim_action, N_episodes, max_steps, savepath_act, savepath_crt, savepath_data):
    global SEED

    # We will use these variables to compute the average episodic reward and the average number of steps per episode
    episode_reward_list = []       # this list contains the total reward per episode
    episode_number_of_steps = []   # this list contains the number of steps per episode
    loss_crt = []
    loss_act = []

    # Agent initialization
    agent = PPO_Agent(dim_state, dim_action, SEED)
    # agent.actor.load_state_dict(torch.load("/content/drive/MyDrive/Colab_Notebooks/EL2805/3limit_actor.pth"))
    # agent.critic.load_state_dict(torch.load("/content/drive/MyDrive/Colab_Notebooks/EL2805/3limit_critic.pth"))

    # trange is an alternative to range in python. It shows a nice progression bar that you can update with useful information
    EPISODES = trange(N_episodes, desc='Episode: ', leave=True)
    # EPISODES = range(N_episodes)

    # Training process
    for i in EPISODES:
        # Reset enviroment data and initialize variables
        done = False
        state = env.reset()
        t = 0
        total_episode_reward = 0.
        while not done:
            # Stop exploring in this episode if done=True
            # Agent takes a random action
            # print(state.shape)
            action, old_prob = agent.get_action(state)
            # print(action)
            # print(old_prob.shape)
            # Get next state and reward
            next_state, reward, done, _ = env.step(action)
            # Agent takes a step
            agent.take_step(state, action, reward, old_prob, done)
            # Update episode reward
            total_episode_reward += reward
            # Update state for next iteration
            state = next_state
            t += 1
            
        # Get target values
        Gvalues, Psi = agent.get_target(t)
        
        # Train the agent for a few epochs
        loss_epoch_crt, loss_epoch_act = agent.train_epochs(Gvalues, Psi)
        
        # Append episode reward and total number of steps
        episode_reward_list.append(total_episode_reward)
        episode_number_of_steps.append(t)
        loss_crt.append(loss_epoch_crt)
        loss_act.append(loss_epoch_act)

        # Save model and rewards every 100 episodes
        if i%100==99:
            torch.save(agent.actor, "3actor.pth")
            torch.save(agent.critic, "3critic.pth")
            data = {'reward': episode_reward_list, 'step': episode_number_of_steps}
            torch.save(data, "3data.pt")

        # Stop training and save the trained actor and critic network if the average score in 50 episodes is >=200
        if np.mean(episode_reward_list[-50:]) >= 200.:
            print('\nEnvironment solved! Average Score: {:.2f}'.format(np.mean(episode_reward_list[-50:])))
            torch.save(agent.actor, savepath_act)
            torch.save(agent.critic, savepath_crt)
            data = {'reward': episode_reward_list, 'step': episode_number_of_steps}
            torch.save(data, savepath_data)
            break
        # Updates the tqdm update bar with fresh information
        EPISODES.set_description("Episode {} - Reward/Steps: {:.1f}/{}".format(i, total_episode_reward, t))
    
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
N_episodes = 1600
# Max number of steps in each episode
max_steps = 1000
# Path for saving success trained model
savepath_act = "neural-network-3-actor.pth"
savepath_crt = "neural-network-3-critic.pth"
savepath_data = "3data.pt"

# Training
episode_reward_list, episode_number_of_steps, loss_crt, loss_act = train_PPO(env, dim_state, dim_action,
                                                                  N_episodes, max_steps, 
                                                                  savepath_act, savepath_crt, savepath_data)

# Close environment
env.close()




#======================================= Plot results ==================================#

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
    actor = torch.load("/content/drive/MyDrive/neural-network-3-actor.pth")
    critic = torch.load("/content/drive/MyDrive/neural-network-3-critic.pth")
    actor.eval()
    critic.eval()
except:
    print('File neural-network-1.pth not found!')
    exit(-1)


Xvalue = torch.linspace(0, 1.5, steps=100)
Yvalue = torch.linspace(-math.pi, math.pi, steps=100)
Vvalues = np.zeros((100, 100))
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
            Vvalues[i, j] = critic(state).detach().numpy()
            mu, _ = actor(state)
            Avalues[i, j] = mu.detach().numpy()[:,1]
            X[i, j] = Xvalue[i]
            Y[i, j] = Yvalue[j]

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(Y, X, Vvalues, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
ax.set_title('Q values of restricted state region')


fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(Y, X, Avalues, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
ax.set_title('Engine values of restricted state region')

