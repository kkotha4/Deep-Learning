import random
import numpy as np
from collections import deque
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from memory import ReplayMemory
from model import *
from utils import *
from config import *
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    def __init__(self, action_size):
        self.load_model = False

        self.action_size = action_size

        # These are hyper parameters for the DQN
        self.discount_factor = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.explore_step = 1000000
        self.epsilon_decay = (self.epsilon - self.epsilon_min) / self.explore_step
        self.train_start = 100000
        self.update_target = 1000

        # Generate the memory
        self.memory = ReplayMemory()

        # Create the policy net and the target net
        self.policy_net = DQN(action_size)
        self.policy_net.to(device)
        self.target_net = DQN(action_size)
        self.target_net.to(device)

        self.optimizer = optim.Adam(params=self.policy_net.parameters(), lr=learning_rate)

        # initialize target net
        self.update_target_net()

        if self.load_model:
            self.policy_net = torch.load('save_model/breakout_dqn')

    # after some time interval update the target net to be same with policy net
    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    """Get action using policy net using epsilon-greedy policy"""
    def get_action(self, state):
        #print(type(state))
        if np.random.rand() <= self.epsilon:
            '''### CODE ####'''
            '''# Choose a random action'''
            a= torch.tensor([[random.randrange(self.action_size)]], device=device, dtype=torch.long)
            return a
           
        else:
            ### CODE ####
            # 
            with torch.no_grad():
                  state=torch.FloatTensor(state).unsqueeze(0).cuda()  
                  a=self.policy_net(state).max(1)[1].view(1, 1)
            return a      
           

    # pick samples randomly from replay memory (with batch_size)
    def train_policy_net(self, frame):
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay

        mini_batch = self.memory.sample_mini_batch(frame)
        mini_batch = np.array(mini_batch).transpose()

        history = np.stack(mini_batch[0], axis=0)
        states = torch.FloatTensor(np.float32(history[:, :4, :, :]) / 255.).cuda()
        actions = torch.tensor(list(mini_batch[1])).cuda()
        rewards = torch.tensor(list(mini_batch[2])).cuda()
        next_states = torch.tensor(np.float32(history[:, 1:, :, :]) / 255.).cuda()
        dones = mini_batch[3] # checks if the game is over
        
        # Compute Q(s_t, a) - Q of the current state
        ### CODE ####
        state_action_values = self.policy_net(states).gather(1, actions.view(batch_size,-1))
        
        # Compute Q function of next state
        ### CODE ####
        
        next_state_values = torch.zeros(batch_size,device=device).cuda()
        non_final_mask=torch.tensor(tuple(map(lambda s: s is not None,
                                          next_states)), device=device, dtype=torch.uint8)
        non_final_next_states = torch.cat([s for s in next_states
                                                if s is not None]).view(states.size()).cuda()
        #print(non_final_next_states)
        # Find maximum Q-value of action at next state from target net
        ### CODE ####
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        # Compute the Huber Loss
        ### CODE ####
        expected_state_action_values = (next_state_values * self.discount_factor) + rewards
        loss = F.smooth_l1_loss(state_action_values.view(32), expected_state_action_values)


        # Optimize the model 
        ### CODE ####
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()