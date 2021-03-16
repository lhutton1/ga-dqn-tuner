"""
DQN model used by the reinforcement learning tuner.
"""

import random

import numpy as np

import torch
from torch import nn
import torch.nn.functional as F


class ReplayMemory(object):
    """
    Experience replay memory allows the DQN network to retain previous experiences.
    """
    def __init__(self, capacity, device):
        """
        Initialise empty experience replay buffer.
        """
        self.capacity = capacity
        self.device = device
        self.buffer = []
        self.idx = 0

    def store(self, value):
        """
        Store a value in replay memory.
        """
        if len(self.buffer) < self.capacity:
            self.buffer.append(value)
        else:
            self.buffer[self.idx] = value
        self.idx = (self.idx+1) % self.capacity  # note in original about circular memory

    def sample(self, batch_size, device):
        """
        Sample random values from replay memory.
        """
        transitions = np.array(random.sample(self.buffer, batch_size))
        states = torch.tensor(transitions[:, 0].tolist(), dtype=torch.float32).to(device)
        actions = torch.tensor(transitions[:, 1].tolist(), dtype=torch.long).to(device)
        next_states = torch.tensor(transitions[:, 2].tolist(), dtype=torch.float32).to(device)
        rewards = torch.tensor(transitions[:, 3].tolist(), dtype=torch.float32).to(device)
        ep_done_group = torch.tensor(transitions[:, 4].tolist()).to(device)
        return states, actions, next_states, rewards, ep_done_group

    def __len__(self):
        """Get length of the buffer.
        """
        return len(self.buffer)


# class DQNModel(nn.Module):
#     """A simple DQN model."""
#     def __init__(self, input_size, output_size, hidden_size=256):
#         """
#         Create a simple DQN model with 3 linear layers.
#         """
#         super(DQNModel, self).__init__()
#         #self.linear1 = nn.Linear(input_size, hidden_size)
#         #self.linear2 = nn.Linear(hidden_size, hidden_size)
#         self.linear2 = nn.LSTM(input_size, hidden_size, num_layers=2)
#         self.linear3 = nn.Linear(hidden_size, output_size)
#         self.optimizer = torch.optim.Adam(self.parameters())
#
#     def forward(self, inp):
#         """
#         Compute the forward pass.
#         """
#         #layer1 = F.relu(self.linear1(inp))
#         #layer1 = layer1[None, :, :]
#         inp = inp[None, :, :]
#         layer2, _ = self.linear2(inp)
#         layer2 = torch.squeeze(layer2)
#         out = self.linear3(layer2)
#         return out
#
#     def save_model(self, filename):
#         torch.save(self.state_dict(), filename)
#
#     def load_model(self, filename):
#         self.load_state_dict(torch.load(filename))


class DQNModel(nn.Module):
    """A simple DQN model."""
    def __init__(self, input_size, output_size, hidden_size=256):
        """
        Create a simple DQN model with 3 linear layers.
        """
        super(DQNModel, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)
        self.optimizer = torch.optim.Adam(self.parameters())

    def forward(self, inp):
        """
        Compute the forward pass.
        """
        layer1 = F.relu(self.linear1(inp))
        layer2 = F.relu(self.linear2(layer1))
        out = self.linear3(layer2)
        return out

    def save_model(self, filename):
        torch.save(self.state_dict(), filename)

    def load_model(self, filename):
        self.load_state_dict(torch.load(filename))


class DQNAgent(object):
    def __init__(self,
                 name,
                 device,
                 state_size,
                 action_size,
                 discount=0.99,
                 eps_max=1.0,
                 eps_min=0.01,
                 eps_decay=0.995,
                 memory_capacity=5000,
                 debug=False):
        """
        Create a DQN agent as described in the following paper:
        V. Mnih, K. Kavukcuoglu, D. Silver, A. Graves, I. Antonoglou, D. Wierstra, andM. Riedmiller.
        Playing atari with deep reinforcement learning.arXiv preprintarXiv:1312.5602, 2013.

        This work is inspired by: https://github.com/saashanair/rl-series/ who use DQN to learn how
        to play simple games using a framework called Gym.

        The agent uses an experience replay memory with a policy network to update a target network.
        """
        self.name = name
        self.device = device
        self.debug = debug

        self.epsilon = eps_max
        self.epsilon_min = eps_min
        self.epsilon_decay = eps_decay

        self.discount = discount
        self.state_size = state_size
        self.action_size = action_size
        self.policy_net = DQNModel(self.state_size, self.action_size).to(self.device)
        self.target_net = DQNModel(self.state_size, self.action_size).to(self.device)
        self.target_net.eval()

        self.memory = ReplayMemory(capacity=memory_capacity, device=self.device)

    def update_target_net(self):
        """
        Weights from the current policy network are copied to the target network.
        """
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def update_epsilon(self):
        """
        Reduce the epsilon value for annealing with epsilon-greedy exploration strategy.
        """
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def select_action(self, state):
        """
        Select an action based on epsilon-greedy exploration.
        """
        if random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.tensor([state], dtype=torch.float32).to(self.device)
        with torch.no_grad():
            action = self.policy_net.forward(state)
        return torch.argmax(action).item()

    def train(self, batch_size):
        """
        Learn by updating weights in the required direction.

        DQN Training algorithm:
        step 1 - Fetch the initial configuration state and its reward.
        step 2 - Pass state and reward to DQN and obtain action on that state.
        step 3 - Select the largest Q-value and store it in replay memory.
        step 4 - Use e-greedy exploration to select an action - this could be random or the maximum Q-value.
        step 5 - Apply the selected action and observe the next state and reward.
        step 6 - Input reward, state, observed state, and action taken to Q-network and get Q-value per action.
        step 7 - Again, take the maximum and store it, but apply a discount factor.
        step 8 - Determine loss from values found in steps 3 and 7, and perform optimization step.
        step 9 - Increment state using observation from step 5.
        step 10 - Repeat steps 2-9 for a determined number of episodes.
        """
        if len(self.memory) < batch_size:
            # don't learn if memory not fully initialised
            return
        states, actions, next_states, rewards, ep_dones = self.memory.sample(batch_size, self.device)
        actions_reshape = actions.view(-1, 1)
        q_pred = self.policy_net.forward(states).gather(1, actions_reshape)
        q_target = self.target_net.forward(next_states).max(dim=1).values
        q_target[ep_dones] = 0.0
        y_j = rewards + (self.discount * q_target)
        y_j = y_j.view(-1, 1)
        self.policy_net.optimizer.zero_grad()
        loss = F.mse_loss(y_j, q_pred).mean()
        loss.backward()
        self.policy_net.optimizer.step()
        return loss

    def save_models(self, policy_net_filename, target_net_filename):
        self.policy_net.save_model(policy_net_filename)
        self.target_net.save_model(target_net_filename)

    def load_models(self, policy_net_filename, target_net_filename):
        self.policy_net.load_model(policy_net_filename)
        self.target_net.load_model(target_net_filename)


