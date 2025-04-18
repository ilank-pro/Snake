import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        # Dueling DQN architecture: shared feature layers
        self.feature = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        # Advantage stream
        self.adv_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )

    def forward(self, x):
        features = self.feature(x)
        values = self.value_stream(features)
        advantages = self.adv_stream(features)
        # Combine streams: Q(s,a) = V(s) + A(s,a) - mean(A(s,*))
        q_vals = values + (advantages - advantages.mean(dim=1, keepdim=True))
        return q_vals

## Add Prioritized Replay buffer implementation
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = []
        self.pos = 0

    def __len__(self):  # Allow len(buffer)
        return len(self.buffer)

    def push(self, transition):
        max_prio = max(self.priorities) if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
            self.priorities.append(max_prio)
        else:
            self.buffer[self.pos] = transition
            self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta):
        prios = np.array(self.priorities)
        probs = prios ** self.alpha
        probs /= probs.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[i] for i in indices]
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        return samples, indices, weights

    def update_priorities(self, indices, priorities):
        for idx, prio in zip(indices, priorities):
            self.priorities[idx] = prio

class RLAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        # Prioritized replay memory
        self.memory = PrioritizedReplayBuffer(capacity=100000, alpha=0.6)
        self.beta = 0.4  # initial importance-sampling weight exponent
        self.beta_increment = (1.0 - self.beta) / 100000
        
        self.gamma = 0.99  # Focus on long-term
        self.epsilon = 1.0
        self.epsilon_min = 0.05  # Higher minimum epsilon for ongoing exploration
        self.epsilon_decay = 0.999  # Slow decay for exploration stability
        self.learning_rate = 0.0005 # Moderate learning rate
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Main network and target network
        self.model = DQN(state_size, action_size).to(self.device)
        self.target_model = DQN(state_size, action_size).to(self.device)
        self.update_target_model()
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.9)
        self.criterion = nn.HuberLoss()
        
        # For double DQN
        self.target_update_counter = 0
        self.target_update_frequency = 10 # Moderate update frequency

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def get_state(self, snake, food_pos, other_snake_body, obstacles, grid_size):
        head = snake.body[0]
        
        # Normalized positions
        head_x = head[0] / grid_size
        head_y = head[1] / grid_size
        food_x = food_pos[0] / grid_size
        food_y = food_pos[1] / grid_size
        
        # Direction to food (normalized)
        dx = (food_x - head_x)
        dy = (food_y - head_y)
        
        # Distance to walls (normalized)
        wall_dist_left = head_x
        wall_dist_right = 1 - head_x
        wall_dist_up = head_y
        wall_dist_down = 1 - head_y
        
        # Danger detection with multiple ranges
        directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
        danger = []
        vision_ranges = [1, 2, 3]
        
        for dx_dir, dy_dir in directions:
            danger_level = 0
            for range_idx, vision_range in enumerate(vision_ranges):
                check_x = (head[0] + dx_dir * vision_range) % grid_size
                check_y = (head[1] + dy_dir * vision_range) % grid_size
                check_pos = (check_x, check_y)
                
                if (check_pos in snake.body[1:] or 
                    check_pos in other_snake_body or 
                    check_pos in obstacles):
                    danger_level = 1 - (range_idx / len(vision_ranges))
                    break
            danger.append(danger_level)
        
        # Current direction one-hot encoding
        dir_vec = [0, 0, 0, 0]
        dir_idx = directions.index(snake.direction)
        dir_vec[dir_idx] = 1
        
        # Snake lengths (normalized)
        self_length = len(snake.body) / grid_size
        other_length = len(other_snake_body) / grid_size
        
        # Combine all features (state_size should be 20)
        state = ([head_x, head_y, food_x, food_y, dx, dy] +  # 6 features
                danger +  # 4 features
                dir_vec +  # 4 features
                [wall_dist_left, wall_dist_right, wall_dist_up, wall_dist_down] +  # 4 features
                [self_length, other_length])  # 2 features
        
        return np.array(state, dtype=np.float32)

    def remember(self, state, action, reward, next_state, done):
        self.memory.push((state, action, reward, next_state, done))

    def act(self, state, training=True):
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.model(state_tensor)
            return q_values.argmax().item()

    def replay(self, batch_size):
        if len(self.memory.buffer) < batch_size:
            return
        
        # Sample from prioritized memory
        minibatch, indices, weights = self.memory.sample(batch_size, beta=self.beta)
        self.beta = min(1.0, self.beta + self.beta_increment)
        states, actions, rewards, next_states, dones = zip(*minibatch)
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)
        weights = np.array(weights)

        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        weights = torch.FloatTensor(weights).unsqueeze(1).to(self.device)

        # Double DQN
        current_q_values = self.model(states).gather(1, actions.unsqueeze(1))
        next_actions = self.model(next_states).argmax(1)
        next_q_values = self.target_model(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # Compute per-sample loss
        loss_elements = F.smooth_l1_loss(current_q_values.squeeze(), target_q_values, reduction='none')
        # Update priorities
        new_prios = loss_elements.detach().cpu().numpy() + 1e-6
        self.memory.update_priorities(indices, new_prios)
        # Weighted loss
        loss = (weights * loss_elements.unsqueeze(1)).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0) # Keep gradient clipping at 1.0
        self.optimizer.step()

        # Update target network
        self.target_update_counter += 1
        if self.target_update_counter >= self.target_update_frequency:
            self.update_target_model()
            self.target_update_counter = 0

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'target_model_state_dict': self.target_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device) # Added map_location
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.target_model.load_state_dict(checkpoint['target_model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.model.eval()
        self.target_model.eval() 