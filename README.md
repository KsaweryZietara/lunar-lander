# Lunar Lander

The goal of the project is to design an algorithm (agent) that can safely land a lunar lander on a designated area of the moon's surface. The challenge involves controlling the lander's speed, position, and tilt angle​​.

## Prerequisites

- [Python](https://www.python.org/)
- [PyTorch](https://pytorch.org/)
- [Gymnasium](https://gymnasium.farama.org/)

## Installation

- Clone the Repository.
```bash
git clone https://github.com/KsaweryZietara/lunar-lander.git
cd lunar-lander
```

- To train the model using the genetic algorithm, run the `genetic_algorithm.py` script. This will initialize a population of neural networks and evolve them over several generations to find the best-performing model.
```bash
python3 genetic_algorithm.py
```

- To test the trained model, run the `lunar_lander.py` script. This will load the best model and use it to control the lunar lander for 10 games, displaying the score for each game.
```bash
python3 lunar_lander.py
```

## Observation Space

The algorithm observes the following states:
- Horizontal position
- Vertical position
- Horizontal velocity
- Vertical velocity
- Tilt angle
- Rotational speed
- Left leg contact indicator
- Right leg contact indicator​​

## Action Space

There are four discrete actions available:
- 0: do nothing
- 1: fire left orientation engine
- 2: fire main engine
- 3: fire right orientation engine

## Rewards
After every step a reward is granted. The total reward of an episode is the sum of the rewards for all the steps within that episode.

For each step, the reward:
- is increased/decreased the closer/further the lander is to the landing pad.
- is increased/decreased the slower/faster the lander is moving.
- is decreased the more the lander is tilted (angle not horizontal).
- is increased by 10 points for each leg that is in contact with the ground.
- is decreased by 0.03 points each frame a side engine is firing.
- is decreased by 0.3 points each frame the main engine is firing.

The episode receive an additional reward of -100 or +100 points for crashing or landing safely respectively. An episode is considered a solution if it scores at least 200 points.

## Neural Network Architecture

![](https://github.com/KsaweryZietara/lunar-lander/blob/main/assets/neural_network.png)

The [neural network](https://github.com/KsaweryZietara/lunar-lander/blob/main/neural_network.py#L4) used in this project consists of two fully connected layers:
- Input Layer: 8 neurons corresponding to the observation space.
- First Fully Connected Layer: 16 neurons, with ReLU activation.
- Second Fully Connected Layer: 4 neurons, corresponding to the action space.

## Genetic Algorithm

#### Configuration

The genetic algorithm [parameters](https://github.com/KsaweryZietara/lunar-lander/blob/main/genetic_algorithm.py#L7) include:
- Population Size: Number of models in the population for solution searching.
- Mutation Rate: Probability of parameter mutation for solution exploration.
- Mutation Power: Degree of parameter change during mutation.
- Number of Generations: Iterations for evolving the population.
- Number of Games: Games each model plays to evaluate fitness​​.

#### Process

- Population Generation: Creating an initial population of neural network models.
- Fitness Calculation: Assessing each model by playing a series of games and calculating the average score.
- Selection: Using a roulette wheel method for model selection.
- Crossover: Combining the genetic information of two parent neural networks to create offspring. It involves selecting a crossover point and exchanging the segments of the parents' neural network weights and biases. 
- Mutation: Introducing random changes to the neural network's weights and biases. This helps in exploring the solution space and avoiding local optima.
- Elitism: Ensuring that the best-performing individual of each generation is carried over to the next generation without any modifications. This helps in retaining the best solutions found so far.

## Results

![](https://github.com/KsaweryZietara/lunar-lander/blob/main/assets/chart.png)

The performance of the algorithm is shown in the change in average scores over generations, indicating improvement in the models' ability to safely land the lunar module over time​​.

This comprehensive approach combines machine learning and evolutionary algorithms to solve the complex problem of autonomous lunar landing.
