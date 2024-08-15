''' code for public entity RL policy '''
# if the public entity knows the state of the network and agents' original trust levels
# (and knows how agents update their trust)
# it can compute an ideal policy

import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import formation
import networkx as nx

N = 5 # number of agents and actions
rho = 1 # the resource constraint
#selfishness = .5 # this affects how much the public entity cares about providing poor services

# complete network formation process
for i in range(50):
    if i == 0:
        G = formation.new(N, alpha = .5, Tau_a = 2, Tau_b = 2)
        formation.christakis(G,i)
    else:
        formation.christakis(G,i)


# hyperparameters
state_size = N  # number of agents
action_size = N  # number of agents (vector that represents resources given to each agent)
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
learning_rate = 0.001
batch_size = 32
memory_size = 2000
episodes = 20

# Define a simple environment
class SimpleEnvironment:
    def __init__(self, N, rho):
        self.num_agents = N
        self.resources = rho

        self.state = list(nx.get_node_attributes(G, 'trust').values())
        #print(self.state)

    def reset(self):
        self.state = list(nx.get_node_attributes(G, 'trust').values())
        return self.state

    def step(self, action):
        reward = 0
        #action = np.clip(action, 0, self.resources)
        #total_action = np.sum(action)

        # normalize so sum is below total resources
        #if total_action > self.resources:
        #    action = (action / total_action) * self.resources

        #for i in range(self.num_agents):
            #if action[i] > 0:
                # if service is given, trust will change
                # TO DO: use metrics.util; but for now just make some function
                #trust_increase = np.sqrt(action[i]) # say that the agent trust increase is exactly the resources given
                #reward += (1-selfishness)*trust_increase   # Reward for increasing trust

                #self.state[i] = np.clip(self.state[i] + trust_increase, 0, 1)

            #else:
                #trust_decrease = .001
                #reward -= trust_decrease   # penalty for not providing any service
                #self.state[i] = np.clip(self.state[i] - trust_decrease, 0, 1)

            # reward/penalty for neighbors
            #neighbor_effect_array = []
            #for j in list(G.neighbors(list(G.nodes)[i])):
                #neighbor_effect_array.append(self.state[j-1])

            #if (len(neighbor_effect_array) != 0):
                #self.state[i] = (self.state[i] + np.mean(neighbor_effect_array))/2



        #if sum(action) > self.resources:
            #resource_penalty = 10000  # penalty for resource overuse
            #reward -= resource_penalty

        #else:
            #resource_reward = (self.resources - sum(action))/self.resources # small reward for resource under-use
            #reward += resource_reward

        #for i in range(self.num_agents):

        reward = np.prod(action)

        #if np.sum(action) > self.resources:
        #    reward = -1
        if np.sum(action) < self.resources:
            reward -= .000001
        #else:
        #    reward += .0001

        next_state = self.state
        return next_state, reward

# Define the DQN model
class DQNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return torch.relu(self.fc3(x))  # Ensure non-negative actions

# Define the DQN agent
class DQNAgent:
    def __init__(self, state_size, action_size, rho):
        self.state_size = state_size
        self.action_size = action_size
        self.resources = rho
        self.memory = deque(maxlen=memory_size)
        self.epsilon = epsilon
        self.model = DQNetwork(state_size, action_size)
        self.target_model = DQNetwork(state_size, action_size)
        self.update_target_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            action = np.random.rand(self.action_size)  # Random action
        else:
            state = torch.FloatTensor(state).unsqueeze(0)
            act_values = self.model(state)
            action = act_values.detach().numpy()[0]

        # Constrain actions to the available resources
        #action = np.clip(action, 0, None) # keep actions non-negative
        if np.sum(action) > self.resources:
            action = (action / np.sum(action)) * self.resources

        return action # Best action according to model

    def replay(self):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state in minibatch:
            target = reward
            if next_state is not None:
                next_state = torch.FloatTensor(next_state).unsqueeze(0)
                target = reward + gamma * torch.max(self.target_model(next_state)[0]).item()
            state = torch.FloatTensor(state).unsqueeze(0)
            target_f = self.model(state)
            target_f = target_f.clone()  # Clone the tensor to avoid in-place operations
            target_f[0][np.argmax(action)] = target
            self.optimizer.zero_grad()
            loss = self.criterion(target_f, self.model(state))
            loss.backward()
            self.optimizer.step()
        if self.epsilon > epsilon_min:
            self.epsilon *= epsilon_decay

# Training loop
def train_dqn(episodes):
    env = SimpleEnvironment(state_size, rho)
    agent = DQNAgent(state_size, action_size, rho)
    for e in range(episodes):
        state = env.reset()
        for time in range(200):
            action = agent.act(state)
            next_state, reward = env.step(action)
            agent.remember(state, action, reward, next_state)
            state = next_state
            if time == 199:
                print(f"Episode {e+1}/{episodes}, Reward: {reward}")
                break
            agent.replay()
        agent.update_target_model()
    # Save the trained model
    torch.save(agent.model.state_dict(), 'dqn_model.pth')
    return agent, env

# Start training
agent, env = train_dqn(episodes)

# Load the trained model and use the policy
agent.model.load_state_dict(torch.load('dqn_model.pth'))
agent.model.eval()  # Set the model to evaluation mode

# Use the trained policy
state = env.reset()
state = torch.FloatTensor(state).unsqueeze(0)
with torch.no_grad():  # No need to track gradients for inference
    action = agent.model(state)
    action = action.detach().numpy()[0]
    print(f"Selected action: {action}")
