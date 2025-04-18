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

# Directions
UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)

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
            state = get_state(self, Snake(0, 0, BLUE), food_pos, obstacles)  # Create dummy opponent snake
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

def is_collision(pos: Tuple[int, int], snake1_body: List[Tuple[int, int]], snake2_body: List[Tuple[int, int]], obstacles: List[Tuple[int, int]]) -> bool:
    """Check if a position collides with snake bodies or obstacles"""
    x, y = pos
    # Check if position is out of bounds
    if x < 0 or x >= GRID_WIDTH or y < 0 or y >= GRID_HEIGHT:
        return True
    # Check collision with snake bodies and obstacles
    return pos in snake1_body or pos in snake2_body or pos in obstacles

def get_state(snake1, snake2, food_pos, obstacles):
    """Get state representation with prioritized features"""
    state = []
    
    # 1. Food position relative to snake1 (4 features)
    dx = food_pos[0] - snake1.body[0][0]
    dy = food_pos[1] - snake1.body[0][1]
    state.extend([dx / GRID_WIDTH, dy / GRID_HEIGHT, 
                 abs(dx) / GRID_WIDTH, abs(dy) / GRID_HEIGHT])
    
    # 2. Snake1's direction (4 features)
    state.extend([1 if snake1.direction == d else 0 for d in [UP, DOWN, LEFT, RIGHT]])
    
    # 3. Immediate danger detection (4 features)
    head = snake1.body[0]
    state.extend([
        1 if is_collision((head[0], head[1] - 1), snake1.body[1:], snake2.body, obstacles) else 0,  # Up
        1 if is_collision((head[0], head[1] + 1), snake1.body[1:], snake2.body, obstacles) else 0,  # Down
        1 if is_collision((head[0] - 1, head[1]), snake1.body[1:], snake2.body, obstacles) else 0,  # Left
        1 if is_collision((head[0] + 1, head[1]), snake1.body[1:], snake2.body, obstacles) else 0   # Right
    ])
    
    # 4. Distance to opponent's head (2 features)
    opp_head = snake2.body[0]
    state.extend([
        (opp_head[0] - head[0]) / GRID_WIDTH,
        (opp_head[1] - head[1]) / GRID_HEIGHT
    ])
    
    # 5. Length comparison (1 feature)
    state.append(1 if len(snake1.body) > len(snake2.body) else 0)
    
    # 6. Food availability (1 feature)
    state.append(1 if food_pos else 0)
    
    return np.array(state, dtype=np.float32)

def train_rl_agent(episodes=2000, batch_size=128, max_steps=2000):
    """Train the RL agent with improved parameters and curriculum"""
    agent = RLAgent(
        state_size=16,  # Matches the actual number of features in get_state
        action_size=4
    )
    
    best_score = -float('inf')
    scores = []
    avg_scores = []
    epsilons = []
    total_rewards = []
    
    # Enhanced curriculum learning
    for episode in range(episodes):
        # Initialize game state
        snake1 = Snake(GRID_WIDTH // 4, GRID_HEIGHT // 2, GREEN)
        snake2 = Snake(3 * GRID_WIDTH // 4, GRID_HEIGHT // 2, BLUE)
        obstacles = []
        food_pos = spawn_food(snake1.body, snake2.body, obstacles)
        last_obstacle_time = time.time()
        
        # Adjust obstacle spawn rate based on episode
        if episode < 200:  # First 200 episodes: no obstacles
            obstacle_interval = float('inf')
        elif episode < 400:  # Next 200 episodes: sparse obstacles
            obstacle_interval = 45
        elif episode < 600:  # Next 200 episodes: moderate obstacles
            obstacle_interval = 30
        else:  # Regular obstacles
            obstacle_interval = 15
        
        state = get_state(snake1, snake2, food_pos, obstacles)
        total_reward = 0
        steps = 0
        food_eaten = 0
        
        while snake1.alive and snake2.alive and steps < max_steps:
            steps += 1
            
            # Spawn obstacles based on curriculum schedule
            if obstacle_interval != float('inf'):
                current_time = time.time()
                if current_time - last_obstacle_time >= obstacle_interval:
                    new_obstacle = spawn_obstacle(snake1.body, snake2.body, obstacles)
                    obstacles.append(new_obstacle)
                    last_obstacle_time = current_time

            # Move snakes
            food_eaten1 = snake1.move(food_pos, snake2.body, obstacles, agent)
            food_eaten2 = snake2.move(food_pos, snake1.body, obstacles)

            # Calculate enhanced reward structure
            reward = -0.01  # Smaller penalty per step to encourage exploration
            
            # Food-related rewards
            if food_eaten1:
                reward += 5.0  # Reduced food reward
                # Additional reward for efficient path to food
                if steps < max_steps * 0.5:  # If food was eaten quickly
                    reward += 2.0
            
            # Survival rewards
            if snake1.alive:
                reward += 0.05  # Small reward for staying alive
                # Bonus for maintaining distance from opponent
                head1 = snake1.body[0]
                head2 = snake2.body[0]
                distance = abs(head1[0] - head2[0]) + abs(head1[1] - head2[1])
                if distance > GRID_WIDTH // 2:  # If maintaining good distance
                    reward += 0.1
            
            # Death penalties
            if not snake1.alive:
                reward -= 5.0  # Reduced death penalty
            elif not snake2.alive:
                reward += 2.5  # Reduced opponent death bonus
            
            # Get new state
            next_state = get_state(snake1, snake2, food_pos, obstacles)
            
            # Store experience
            done = not snake1.alive or not snake2.alive or steps >= max_steps
            agent.remember(state, agent.act(state), reward, next_state, done)
            
            state = next_state
            total_reward += reward

            # Spawn new food if eaten
            if food_eaten1 or food_eaten2:
                food_pos = spawn_food(snake1.body, snake2.body, obstacles)
                food_eaten = 1

            # Train on batch once warm-up buffer is filled
            if len(agent.memory) >= max(batch_size, 1000):  # Warm-up until 1000 experiences
                agent.replay(batch_size)

        # Store statistics
        scores.append(snake1.score)
        avg_scores.append(np.mean(scores[-100:]) if len(scores) >= 100 else np.mean(scores))
        epsilons.append(agent.epsilon)
        total_rewards.append(total_reward)
        
        # Save best model
        if snake1.score > best_score:
            best_score = snake1.score
            agent.save("snake_model_best.pth")
        
        # Print detailed progress
        print(f"\nEpisode: {episode+1}/{episodes}")
        print(f"Score: {snake1.score}")
        print(f"Best Score: {best_score}")
        print(f"Average Score (last 100): {avg_scores[-1]:.2f}")
        print(f"Steps: {steps}")
        print(f"Epsilon: {agent.epsilon:.3f}")
        print(f"Total Reward: {total_reward:.2f}")
        
        # Print learning progress every 50 episodes
        if (episode + 1) % 50 == 0:
            print("\nLearning Progress:")
            print(f"Last 50 episodes average score: {np.mean(scores[-50:]):.2f}")
            print(f"Epsilon decay: {epsilons[-50]:.3f} -> {epsilons[-1]:.3f}")
            print(f"Average total reward: {np.mean(total_rewards[-50:]):.2f}")
        
        # Save checkpoint every 100 episodes
        if (episode + 1) % 100 == 0:
            agent.save(f"snake_model_checkpoint_{episode+1}.pth")
            # Step learning rate scheduler for gradual decay
            try:
                agent.scheduler.step()
            except AttributeError:
                pass
    
    # Save the final model
    agent.save("snake_model.pth")
    
    # Print final statistics
    print("\nTraining Summary:")
    print(f"Final Average Score: {np.mean(scores):.2f}")
    print(f"Best Score Achieved: {best_score}")
    print(f"Final Epsilon: {agent.epsilon:.3f}")
    print(f"Average Total Reward: {np.mean(total_rewards):.2f}")
    
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