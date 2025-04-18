# Snake Battle

A Python implementation of a Snake game where two autonomous snakes battle each other for survival and food. Features both simple AI and reinforcement learning capabilities.

## Features

- Two autonomous snakes (blue and green) that battle each other
- Scoring system:
  - 1 point per second alive
  - 10 points for eating food
  - 50 points for surviving when opponent dies
- Grid-based movement
- Obstacles that appear every 3 seconds
- Wrap-around borders
- Cumulative score tracking
- Reinforcement Learning agent option
- Multiple game modes

## Requirements

- Python 3.x
- Pygame 2.5.2
- NumPy 1.24.3
- PyTorch 2.2.0
- Gymnasium 0.29.1

## Installation

1. Clone the repository:
```bash
git clone https://github.com/ilank-pro/Snake.git
cd Snake
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

## How to Play

The game can be run in several modes:

1. Simple AI mode (both snakes use basic AI):
```bash
python snake_battle.py --play
```

2. Train the RL agent (trains over 500 episodes):
```bash
python snake_battle.py --train
```

3. Use trained model for snake one (blue snake):
```bash
python snake_battle.py --one
```

4. Use trained model for snake two (green snake):
```bash
python snake_battle.py --two
```

The game features:
- Two autonomous snakes (blue and green)
- Red food squares
- Yellow obstacles that appear every 3 seconds
- Score display for both snakes
- Automatic reset when one or both snakes die
- Grid display for better visibility

## Game Rules

- Snakes move automatically using either simple AI or trained RL model
- Snakes die if they collide with:
  - Themselves
  - The other snake
  - Obstacles
- Food appears randomly on the board
- New obstacles appear every 3 seconds
- Game resets when one or both snakes die
- Cumulative scores are maintained between rounds

## Reinforcement Learning

The game includes a reinforcement learning agent implemented using PyTorch. The agent:
- Uses a Deep Q-Network (DQN)
- Takes into account:
  - Distance to food
  - Danger in each direction
  - Current direction
- Can be trained over multiple episodes
- Can be saved and loaded for later use 