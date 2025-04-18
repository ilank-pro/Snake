# Snake Battle

A Python implementation of a Snake game where two autonomous snakes battle each other for survival and food.

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

## Requirements

- Python 3.x
- Pygame 2.5.2
- NumPy 1.24.3

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

Run the game:
```bash
python snake_battle.py
```

The game features:
- Two autonomous snakes (blue and green)
- Red food squares
- Yellow obstacles that appear every 3 seconds
- Score display for both snakes
- Automatic reset when one or both snakes die

## Game Rules

- Snakes move automatically using AI
- Snakes die if they collide with:
  - Themselves
  - The other snake
  - Obstacles
- Food appears randomly on the board
- New obstacles appear every 3 seconds
- Game resets when one or both snakes die 