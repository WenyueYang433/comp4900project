# COMP 4900 - Snake Game Project: 
# PPO snakes implementation:
# by: Mingrui (Rayment) Liang

import os
import numpy as np
import torch as trc
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

class PpoSnakeMemory():
    def __init__(self, batch_size):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.batch_size = batch_size
        
    def create_batches(self):
        # n_states = len(self.states)
        # reshaped_states = np.array(self.states).reshape((n_states, -1)) # Reshape the states to have a consistent shape
        # return reshaped_states, np.array(self.actions), np.array(self.probs), np.array(self.vals), np.array(self.rewards), np.array(self.dones)
        
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype = np.int64)
        np.random.shuffle(indices)
        batches = [indices [x : x + self.batch_size] for x in batch_start]
        return np.array(self.states),\
        np.array(self.actions),\
        np.array(self.probs),\
        np.array(self.vals),\
        np.array(self.rewards),\
        np.array(self.dones),\
        batches
    
    def save_memory(self, state, action, probs, vals, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)
        
    def clear_memory(self): 
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []
        
class ActorNetwork(nn.Module): 
    def __init__(self, n_actions, input_size, alpha, 
                 fc1_dims = 256, fc2_dims = 256, 
                 checkPt_dir = 'tmp/snake_ppo'):
        super(ActorNetwork, self).__init__()
        
        self.checkPt_file = os.path.join(checkPt_dir, 'actor_torch_snake_ppo')
        self.actor = nn.Sequential(
            nn.Linear(input_size, fc1_dims), 
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims), 
            nn.ReLU(),
            nn.Linear(fc2_dims, n_actions), 
            nn.ReLU(),
            nn.Softmax(dim = -1)
        )
        
        self.optimizer = optim.Adam(self.parameters(), lr = alpha)
        self.device = trc.device('cuda:0' if trc.cuda.is_available() else 'cpu') # Device handler: check if CUDA is supported in the current version of Pytorch
        self.to(self.device)
    
    # determine ratio: 
    def forward(self, state):
        distribution = self.actor(state)
        distribution = Categorical(distribution)
        return distribution
    
    def save_checkPt(self):
        trc.save(self.state_dict(), self.checkPt_file)
        
    def load_checkPt(self): 
        self.load_state_dict(trc.load(self.checkPt_file))
        
class CriticNetwork(nn.Module):
    def __init__(self, input_size, alpha,
                 fc1_dims = 256, fc2_dims = 256,
                 checkPt_dir = 'tmp/snake_ppo'):
        super(CriticNetwork, self).__init__()        
        
        self.checkPt_file = os.path.join(checkPt_dir, 'critic_torch_snake_ppo')
        self.fc1 = nn.Linear(input_size, fc1_dims)
        self.critic = nn.Sequential(
            nn.Linear(input_size, fc1_dims), 
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims), 
            nn.ReLU(),
            nn.Linear(fc2_dims, 1), 
        )
        
        self.optimizer = optim.Adam(self.parameters(), lr = alpha)
        self.device = trc.device('cuda:0' if trc.cuda.is_available() else 'cpu')
        self.to(self.device)
        
    def forward(self, state):
        value = self.critic(state)
        return value
    
    def save_checkPt(self):
        trc.save(self.state_dict(), self.checkPt_file)
        
    def load_checkPt(self): 
        self.load_state_dict(trc.load(self.checkPt_file))
    
class PpoNet(nn.Module): 
    def __init__(self, input_size, hidden_size, output_size,
                 n_actions, input_dims, gamma = 0.99, alpha = 0.0003, 
                 gae_lambda = 0.95, policy_clip = 0.2, batch_size = 64, 
                 N = 2048, n_epochs = 10): # policy clip can be changed to 0.1
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda
        
        self.actor = ActorNetwork(n_actions, input_dims, alpha)
        self.critic = CriticNetwork(input_dims, alpha)
        self.memory = PpoSnakeMemory(batch_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x
        
    def save_snakes(self):
        print('Saving snakes...')
        self.actor.save_checkPt()
        self.critic.save_checkPt()
        
    def load_snakes(self):
        print('Loading snakes...')
        self.actor.load_checkPt()
        self.critic.load_checkPt()
        
    def save(self, file_name='model.pth'):
        self.save_snakes()
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        trc.save(self.state_dict(), file_name)    

    def select_action(self, observation):
        state = trc.tensor([observation], dtype = trc.float).to(self.actor.device)
        distribution = self.actor(state)
        value = self.critic(state)
        action = distribution.sample()
        
        probs = trc.squeeze(distribution.log_prob(action)).item()     # item() returns an integer
        value = trc.squeeze(value).item()
        action = trc.squeeze(action).item()
        return action, probs, value
    
    def remember(self, state, action, probs, vals, reward, done):
        self.memory.save_memory(state, action, probs, vals, reward, done)
    
        
class PpoTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        
    def train_step(self, model):
        for _ in range(model.n_epochs):
            state_arr, action_arr, old_probs_arr, vals_arr,\
            reward_arr, done_arr, batches \
                = model.memory.create_batches()
                
            values = vals_arr
            advantage = np.zeros(len(reward_arr), dtype = np.float32)
            
            for t in range(len(reward_arr) - 1):
                # initialize discount factor and action t value: 
                discount_factor = 1
                action_t = 0
                for n in range (t, len(reward_arr) - 1): 
                    # by RL convention: 
                    action_t += discount_factor * \
                        (reward_arr[n] + model.gamma * values[n + 1] * (1 - int(done_arr[n])) - values[n])
                    # determine discount factor: 
                    discount_factor *= model.gamma * model.gae_lambda
                # determine current advantage value
                advantage[t] = action_t
            # return advantages
            advantage = trc.tensor(advantage).to(model.actor.device)
            
            values = trc.tensor(values).to(model.actor.device)
            
            for b in batches: 
                states = trc.tensor(state_arr[b], dtype = trc.float).to(model.actor.device)
                old_probs = trc.tensor(old_probs_arr[b]).to(model.actor.device)
                actions = trc.tensor(action_arr[b]).to(model.actor.device)
                
                distribution = model.actor(states)
                critic_value = model.critic(states)
                critic_value = trc.squeeze(critic_value)
                
                new_probs = distribution.log_prob(actions)
                prob_ratio = new_probs.exp()/old_probs.exp()
                
                weighted_probs = advantage[b] * prob_ratio
                weighted_clipped_probs = trc.clamp(prob_ratio, 1 - model.policy_clip)
                
                actor_loss = (-1) * trc.min(weighted_probs, weighted_clipped_probs).mean()
                
                results = advantage[b] + values[b]
                
                critic_loss = (results - critic_value)
                critic_loss = critic_loss.mean()
                
                total_loss = actor_loss + critic_loss * 0.5
                model.actor.optimizer.zero_grad()
                model.critic.optimizer.zero_grad()
                
                # backpropagating total loss:
                total_loss.backward()
                
                model.actor.optimizer.step()
                model.critic.optimizer.step()
            
            # model.memory.clear_memory()
    
class Memory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []

    def add_memory(self, state, action, prob, val, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(prob)
        self.vals.append(val)
        self.rewards.append(reward)
        self.dones.append(done)

    def create_batches(self):
        print(f"States shape: {np.array(self.states).shape}")  # Print shape of states array
        print(f"States values: {np.array(self.states)}")  # Print values of states array
        return np.array(self.states), \
               np.array(self.actions), \
               np.array(self.probs), \
               np.array(self.vals), \
               np.array(self.rewards), \
               np.array(self.dones)
