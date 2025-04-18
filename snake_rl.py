import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )

    def forward(self, x):
        return self.network(x)

class RLAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=100000)  # Increased memory size
        self.gamma = 0.99  # Increased discount factor
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.997  # Slower decay
        self.learning_rate = 0.0005  # Reduced learning rate
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Main network and target network
        self.model = DQN(state_size, action_size).to(self.device)
        self.target_model = DQN(state_size, action_size).to(self.device)
        self.update_target_model()
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        
        # For double DQN
        self.target_update_counter = 0
        self.target_update_frequency = 10

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def get_state(self, snake, food_pos, other_snake_body, obstacles, grid_size):
        head = snake.body[0]
        
        # Distance to food
        food_x_dist = (food_pos[0] - head[0]) / grid_size
        food_y_dist = (food_pos[1] - head[1]) / grid_size
        
        # Distance to walls
        wall_dist_left = head[0] / grid_size
        wall_dist_right = (grid_size - head[0]) / grid_size
        wall_dist_up = head[1] / grid_size
        wall_dist_down = (grid_size - head[1]) / grid_size
        
        # Danger in each direction (up, right, down, left)
        directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
        danger = []
        vision_range = 3  # Look ahead 3 cells
        
        for dx, dy in directions:
            danger_level = 0
            for i in range(1, vision_range + 1):
                check_x = (head[0] + dx * i) % grid_size
                check_y = (head[1] + dy * i) % grid_size
                check_pos = (check_x, check_y)
                
                if (check_pos in snake.body[1:] or 
                    check_pos in other_snake_body or 
                    check_pos in obstacles):
                    danger_level = 1 - (i - 1) / vision_range
                    break
            danger.append(danger_level)
        
        # Current direction one-hot encoding
        dir_vec = [0, 0, 0, 0]
        dir_idx = directions.index(snake.direction)
        dir_vec[dir_idx] = 1
        
        # Length of both snakes
        self_length = len(snake.body) / grid_size
        other_length = len(other_snake_body) / grid_size
        
        # Combine all features
        state = (danger + dir_vec + 
                [food_x_dist, food_y_dist] + 
                [wall_dist_left, wall_dist_right, wall_dist_up, wall_dist_down] +
                [self_length, other_length])
        
        return np.array(state)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, training=True):
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.model(state_tensor)
            return q_values.argmax().item()

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        
        minibatch = random.sample(self.memory, batch_size)
        states = torch.FloatTensor([t[0] for t in minibatch]).to(self.device)
        actions = torch.LongTensor([t[1] for t in minibatch]).to(self.device)
        rewards = torch.FloatTensor([t[2] for t in minibatch]).to(self.device)
        next_states = torch.FloatTensor([t[3] for t in minibatch]).to(self.device)
        dones = torch.FloatTensor([t[4] for t in minibatch]).to(self.device)

        # Double DQN
        current_q_values = self.model(states).gather(1, actions.unsqueeze(1))
        next_actions = self.model(next_states).argmax(1)
        next_q_values = self.target_model(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        loss = self.criterion(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
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
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.target_model.load_state_dict(checkpoint['target_model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.model.eval()
        self.target_model.eval() 