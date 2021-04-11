"""
DQN model used by GA-DQN.

NOTE: This work is inspired by: https://github.com/saashanair/rl-series/ who use DQN to learn how
      to play simple Atari games using a framework called Gym.
"""

import random

import numpy as np

import torch
from torch import nn
import torch.nn.functional as F


class ExperienceReplayMemory:
    """
    A simple experience replay buffer for storing DQN experience.
    Experience replay memory allows the DQN network to retain previous
    experiences.
    """
    def __init__(self, capacity):
        """
        Initialise empty experience replay buffer.
        """
        self.capacity = capacity
        self.memory_start = 0

        # Buffers
        self.state = []
        self.action = []
        self.reward = []
        self.next_state = []

    def store_experience(self, experience):
        """
        Store a value in experience replay memory.
        """
        state, action, reward, next_state = experience

        if len(self.state) < self.capacity:
            self.state.append(state)
            self.action.append(action)
            self.reward.append(reward)
            self.next_state.append(next_state)
        else:
            self.state[self.memory_start] = state
            self.action[self.memory_start] = action
            self.reward[self.memory_start] = reward
            self.next_state[self.memory_start] = next_state

        self.memory_start = (self.memory_start + 1) % self.capacity

    def sample(self, batch_size, device):
        """
        Sample of size batch size from experience replay memory.
        """
        sample_idx = np.random.randint(0, len(self.state) - 1, batch_size)
        samples = []

        for buffer in [self.state, self.action, self.reward, self.next_state]:
            tensor = torch.from_numpy(np.array(buffer)[sample_idx])
            # Don't convert action buffer to float
            if not isinstance(buffer[0], int):
                tensor = tensor.float()
            samples.append(tensor.to(device))

        states, actions, next_states, rewards = samples
        return states, actions, next_states, rewards

    def __len__(self):
        """
        Get length of the buffer.
        """
        return len(self.state)


class DQN(nn.Module):
    """
    A simple DQN model.

    Architecture:
        >> Input layer
        >> Hidden layer 1: linear + ReLu
        >> Hidden layer 2: linear + ReLu
        >> Output layer: linear
    """
    def __init__(self, input_size, output_size, hidden_size=(256, 128), learning_rate=1e-3):
        """
        Create a simple DQN model with 3 linear layers.
        """
        super(DQN, self).__init__()

        # DQN model
        self.linear1 = nn.Linear(input_size, hidden_size[0])
        self.linear2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.linear3 = nn.Linear(hidden_size[1], output_size)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, inp):
        """
        Compute the forward pass.
        """
        layer1 = F.relu(self.linear1(inp))
        layer2 = F.relu(self.linear2(layer1))
        return self.linear3(layer2)

    def save_model(self, filename):
        """
        Save a trained model.
        """
        torch.save(self.state_dict(), filename)

    def load_model(self, filename):
        """
        Load a trained model.
        """
        self.load_state_dict(torch.load(filename))


class DQNAgent:
    def __init__(self,
                 name,
                 device,
                 state_size,
                 action_size,
                 discount=0.99,
                 eps=(1.0, 0.01, 0.99),
                 memory_capacity=5000,
                 hidden_sizes=(256, 128),
                 learning_rate=1e-3,
                 debug=False):
        """
        Create a DQN agent that learns actions to take based on previous experience.
        The agent uses an experience replay memory with a policy network to update a target network,
        this can be considered the vanilla variant of DQN.
        """
        self.name = name
        self.device = device
        self.debug = debug

        self.eps_max, self.eps_min, self.eps_decay = eps

        self.discount = discount
        self.state_size = state_size
        self.action_size = action_size

        self.policy = DQN(self.state_size,
                          self.action_size,
                          hidden_sizes,
                          learning_rate).to(self.device)
        self.target = DQN(self.state_size,
                          self.action_size,
                          hidden_sizes,
                          learning_rate).to(self.device)

        # no learning on the target network.
        self.target.eval()

        self.memory = ExperienceReplayMemory(capacity=memory_capacity)

    def increment_target(self):
        """
        Weights from the current policy network are copied to the target network.
        """
        state_dict = self.policy.state_dict()
        self.target.load_state_dict(state_dict)

    def reduce_epsilon(self):
        """
        Reduce the epsilon value for annealing with epsilon-greedy exploration strategy.
        """
        new_eps = self.eps_max * self.eps_decay
        self.eps_max = max(self.eps_min, new_eps)
        return self.eps_max

    def get_action(self, state):
        """
        Select an action based on epsilon-greedy exploration. Returns
        both the action index selected and the maximum Q-value.
        """
        if random.random() <= self.eps_max:
            idx = random.randrange(self.action_size)
            return idx

        if not torch.is_tensor(state):
            state = torch.tensor([state], dtype=torch.float32).to(self.device)

        with torch.no_grad():
            action = self.policy.forward(state)

        max_idx = torch.argmax(action).item()
        return max_idx

    def train(self, batch_size):
        """
        Train the DQN by updating the weights using gradient descent.

        DQN Training algorithm:
            >> Add state, action, reward, next state to experience replay memory
            >> Uniformly sample from memory according to batch size.
            >> Perform forward pass on policy with states and forward pass on target with next states.
            >> Determine loss from values found in steps 3 and 7, and perform optimization step.
            >> Increment state using observation from step 5.
        """
        if len(self.memory) < batch_size:
            return

        states, actions, next_states, rewards = self.memory.sample(
            batch_size, self.device)

        actions = actions.view(-1, 1)

        q_predictions = self.policy.forward(states).gather(1, actions)
        q_target = self.target.forward(next_states).max(dim=1).values
        q_target = rewards + (self.discount * q_target)

        q_target = q_target.view(-1, 1)

        self.policy.optimizer.zero_grad()
        loss = F.mse_loss(q_target, q_predictions).mean()
        loss.backward()
        self.policy.optimizer.step()

        # return loss for debugging
        return loss

    def save_models(self, policy_net_filename, target_net_filename):
        """
        Save both trained policy and target models.
        """
        self.policy.save_model(policy_net_filename)
        self.target.save_model(target_net_filename)

    def load_models(self, policy_net_filename, target_net_filename):
        """
        Load both trained policy and target models.
        """
        self.policy.load_model(policy_net_filename)
        self.target.load_model(target_net_filename)
