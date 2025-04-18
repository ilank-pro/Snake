import pygame
import random
import numpy as np
from typing import List, Tuple
import time
import argparse
from snake_rl import RLAgent

# Initialize Pygame
pygame.init()

# Constants
WINDOW_SIZE = 800
GRID_SIZE = 20
GRID_WIDTH = WINDOW_SIZE // GRID_SIZE
GRID_HEIGHT = WINDOW_SIZE // GRID_SIZE

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
GRAY = (50, 50, 50)

# Set up the display
screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
pygame.display.set_caption("Snake Battle")

class Snake:
    def __init__(self, x: int, y: int, color: Tuple[int, int, int]):
        self.body = [(x, y)]
        self.direction = random.choice([(1, 0), (-1, 0), (0, 1), (0, -1)])
        self.color = color
        self.alive = True
        self.score = 0
        self.cumulative_score = 0
        self.last_score_update = time.time()

    def move(self, food_pos: Tuple[int, int], other_snake_body: List[Tuple[int, int]], obstacles: List[Tuple[int, int]], rl_agent=None) -> bool:
        if not self.alive:
            return False

        # Update score based on time alive
        current_time = time.time()
        time_diff = int(current_time - self.last_score_update)
        if time_diff >= 1:
            self.score += time_diff
            self.cumulative_score += time_diff
            self.last_score_update = current_time

        head = self.body[0]
        
        if rl_agent:
            # Use RL agent for movement
            state = rl_agent.get_state(self, food_pos, other_snake_body, obstacles, GRID_WIDTH)
            action = rl_agent.act(state)
            directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # up, right, down, left
            self.direction = directions[action]
        else:
            # Simple AI: Choose direction based on food position and obstacles
            possible_directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
            best_direction = self.direction
            min_distance = float('inf')

            for dx, dy in possible_directions:
                new_x = (head[0] + dx) % GRID_WIDTH
                new_y = (head[1] + dy) % GRID_HEIGHT
                new_pos = (new_x, new_y)

                if new_pos in self.body[1:] or new_pos in other_snake_body or new_pos in obstacles:
                    continue

                distance = abs(new_x - food_pos[0]) + abs(new_y - food_pos[1])
                if distance < min_distance:
                    min_distance = distance
                    best_direction = (dx, dy)

            self.direction = best_direction

        new_head = ((head[0] + self.direction[0]) % GRID_WIDTH,
                   (head[1] + self.direction[1]) % GRID_HEIGHT)

        # Check for collisions
        if new_head in self.body[1:] or new_head in other_snake_body or new_head in obstacles:
            self.alive = False
            return False

        self.body.insert(0, new_head)

        # Check if food is eaten
        if new_head == food_pos:
            self.score += 10
            self.cumulative_score += 10
            return True

        self.body.pop()
        return False

def spawn_food(snake1_body: List[Tuple[int, int]], snake2_body: List[Tuple[int, int]], obstacles: List[Tuple[int, int]]) -> Tuple[int, int]:
    while True:
        food = (random.randint(0, GRID_WIDTH-1), random.randint(0, GRID_HEIGHT-1))
        if food not in snake1_body and food not in snake2_body and food not in obstacles:
            return food

def spawn_obstacle(snake1_body: List[Tuple[int, int]], snake2_body: List[Tuple[int, int]], obstacles: List[Tuple[int, int]]) -> Tuple[int, int]:
    while True:
        obstacle = (random.randint(0, GRID_WIDTH-1), random.randint(0, GRID_HEIGHT-1))
        if (obstacle not in snake1_body and 
            obstacle not in snake2_body and 
            obstacle not in obstacles):
            return obstacle

def draw_grid():
    for x in range(0, WINDOW_SIZE, GRID_SIZE):
        pygame.draw.line(screen, GRAY, (x, 0), (x, WINDOW_SIZE))
    for y in range(0, WINDOW_SIZE, GRID_SIZE):
        pygame.draw.line(screen, GRAY, (0, y), (WINDOW_SIZE, y))

def train_rl_agent(episodes=500):
    state_size = 16  # 4 danger + 4 direction + 2 food distance + 4 wall distances + 2 snake lengths
    action_size = 4  # up, right, down, left
    agent = RLAgent(state_size, action_size)
    batch_size = 64  # Increased batch size
    max_steps = 2000  # Maximum steps per episode
    
    # Training statistics
    scores = []
    epsilon_history = []
    
    for episode in range(episodes):
        snake1 = Snake(GRID_WIDTH//4, GRID_HEIGHT//2, BLUE)
        snake2 = Snake(3*GRID_WIDTH//4, GRID_HEIGHT//2, GREEN)
        obstacles = []
        food_pos = spawn_food(snake1.body, snake2.body, obstacles)
        last_obstacle_time = time.time()
        
        state = agent.get_state(snake1, food_pos, snake2.body, obstacles, GRID_WIDTH)
        total_reward = 0
        steps = 0
        last_distance = abs(snake1.body[0][0] - food_pos[0]) + abs(snake1.body[0][1] - food_pos[1])
        
        while snake1.alive and snake2.alive and steps < max_steps:
            steps += 1
            
            # Spawn new obstacle every 3 seconds
            current_time = time.time()
            if current_time - last_obstacle_time >= 3:
                new_obstacle = spawn_obstacle(snake1.body, snake2.body, obstacles)
                obstacles.append(new_obstacle)
                last_obstacle_time = current_time

            # Move snakes
            food_eaten1 = snake1.move(food_pos, snake2.body, obstacles, agent)
            food_eaten2 = snake2.move(food_pos, snake1.body, obstacles)

            # Calculate reward
            reward = 0
            
            # Distance-based reward
            current_distance = abs(snake1.body[0][0] - food_pos[0]) + abs(snake1.body[0][1] - food_pos[1])
            if current_distance < last_distance:
                reward += 0.1  # Reward for moving closer to food
            elif current_distance > last_distance:
                reward -= 0.1  # Small penalty for moving away from food
            last_distance = current_distance
            
            # Food rewards
            if food_eaten1:
                reward += 20  # Increased food reward
            elif food_eaten2:
                reward -= 5   # Penalty if opponent gets food
            
            # Survival rewards
            if not snake1.alive:
                reward -= 100  # Increased death penalty
            elif not snake2.alive:
                reward += 100  # Increased survival reward
            elif steps >= max_steps:
                reward += 50   # Reward for surviving full episode
            
            # Get new state
            next_state = agent.get_state(snake1, food_pos, snake2.body, obstacles, GRID_WIDTH)
            
            # Store experience
            done = not snake1.alive or steps >= max_steps
            agent.remember(state, agent.act(state), reward, next_state, done)
            
            state = next_state
            total_reward += reward

            # Spawn new food if eaten
            if food_eaten1 or food_eaten2:
                food_pos = spawn_food(snake1.body, snake2.body, obstacles)

            # Train on batch
            if len(agent.memory) >= batch_size:
                agent.replay(batch_size)

        # Store statistics
        scores.append(snake1.score)
        epsilon_history.append(agent.epsilon)
        
        # Print progress
        avg_score = np.mean(scores[-100:]) if len(scores) >= 100 else np.mean(scores)
        print(f"Episode: {episode+1}/{episodes}, Score: {snake1.score}, Avg Score: {avg_score:.2f}, "
              f"Steps: {steps}, Epsilon: {agent.epsilon:.3f}, Total Reward: {total_reward:.2f}")
    
    # Save the trained model
    agent.save("snake_model.pth")
    return agent

def main():
    parser = argparse.ArgumentParser(description='Snake Battle Game')
    parser.add_argument('--play', action='store_true', help='Play with two autonomous snakes')
    parser.add_argument('--train', action='store_true', help='Train the RL agent')
    parser.add_argument('--one', action='store_true', help='Use trained model for snake one')
    parser.add_argument('--two', action='store_true', help='Use trained model for snake two')
    args = parser.parse_args()

    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 36)

    # Load the RL agent if needed
    rl_agent = None
    if args.one or args.two:
        state_size = 16  # 4 danger + 4 direction + 2 food distance + 4 wall distances + 2 snake lengths
        action_size = 4
        rl_agent = RLAgent(state_size, action_size)
        try:
            rl_agent.load("snake_model.pth")
        except FileNotFoundError:
            print("Error: Could not find snake_model.pth. Please train the model first using --train")
            return
        except Exception as e:
            print(f"Error loading model: {e}")
            return

    if args.train:
        train_rl_agent()
        return

    while True:
        # Initialize game state
        snake1 = Snake(GRID_WIDTH//4, GRID_HEIGHT//2, BLUE)
        snake2 = Snake(3*GRID_WIDTH//4, GRID_HEIGHT//2, GREEN)
        obstacles = []
        food_pos = spawn_food(snake1.body, snake2.body, obstacles)
        game_active = True
        last_obstacle_time = time.time()

        while game_active:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            # Spawn new obstacle every 3 seconds
            current_time = time.time()
            if current_time - last_obstacle_time >= 3:
                new_obstacle = spawn_obstacle(snake1.body, snake2.body, obstacles)
                obstacles.append(new_obstacle)
                last_obstacle_time = current_time

            # Move snakes and check for food consumption
            food_eaten1 = snake1.move(food_pos, snake2.body, obstacles, rl_agent if args.one else None)
            food_eaten2 = snake2.move(food_pos, snake1.body, obstacles, rl_agent if args.two else None)

            if food_eaten1 or food_eaten2:
                food_pos = spawn_food(snake1.body, snake2.body, obstacles)

            # Check for winner and update scores
            if not snake1.alive and snake2.alive:
                snake2.score += 50
                snake2.cumulative_score += 50
                game_active = False
            elif not snake2.alive and snake1.alive:
                snake1.score += 50
                snake1.cumulative_score += 50
                game_active = False
            elif not snake1.alive and not snake2.alive:
                game_active = False

            # Draw everything
            screen.fill(BLACK)
            draw_grid()

            # Draw obstacles
            for obstacle in obstacles:
                pygame.draw.rect(screen, YELLOW,
                               (obstacle[0]*GRID_SIZE, obstacle[1]*GRID_SIZE,
                                GRID_SIZE, GRID_SIZE))

            # Draw food
            pygame.draw.rect(screen, RED,
                           (food_pos[0]*GRID_SIZE, food_pos[1]*GRID_SIZE,
                            GRID_SIZE, GRID_SIZE))

            # Draw snakes
            for snake in [snake1, snake2]:
                for segment in snake.body:
                    pygame.draw.rect(screen, snake.color,
                                   (segment[0]*GRID_SIZE, segment[1]*GRID_SIZE,
                                    GRID_SIZE, GRID_SIZE))

            # Draw scores
            score_text1 = font.render(f"Blue Score: {snake1.cumulative_score}", True, BLUE)
            score_text2 = font.render(f"Green Score: {snake2.cumulative_score}", True, GREEN)
            screen.blit(score_text1, (10, 10))
            screen.blit(score_text2, (10, 50))

            pygame.display.flip()
            clock.tick(10)

            # Small delay before resetting if game is over
            if not game_active:
                pygame.time.wait(1000)

if __name__ == "__main__":
    main() 