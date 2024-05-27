import gymnasium as gym
import torch
import json
import random
from neural_network import NeuralNetwork

# Genetic algorithm configuration
population_size = 50
mutation_rate = 0.3
mutation_power = 0.1
num_generations = 700

# Environment configuration
env = gym.make("LunarLander-v2")
num_games = 10

def initialize_population():
    population = []
    for _ in range(population_size):
        model = NeuralNetwork()
        population.append(model)
    return population

def compute_fitness(model):
    score = 0
    for _ in range(num_games):
        observation, info = env.reset()
        terminated = False
        truncated = False
        while not (terminated or truncated):
            input_data = torch.tensor(observation).unsqueeze(0)
            output = model(input_data)
            action = torch.argmax(output).item()
            observation, reward, terminated, truncated, info = env.step(action)
            score += reward
    return score / num_games

def crossover(parent1, parent2):
    child1 = NeuralNetwork()
    child2 = NeuralNetwork()

    crossover_point_fc1 = random.randint(2, 14)

    # Crossover for weights of fc1 layer
    child1.fc1.weight.data = torch.cat((parent1.fc1.weight.data[:crossover_point_fc1], parent2.fc1.weight.data[crossover_point_fc1:]), dim=0)
    child2.fc1.weight.data = torch.cat((parent2.fc1.weight.data[:crossover_point_fc1], parent1.fc1.weight.data[crossover_point_fc1:]), dim=0)

    # Crossover for biases of fc1 layer
    child1.fc1.bias.data = torch.cat((parent1.fc1.bias.data[:crossover_point_fc1], parent2.fc1.bias.data[crossover_point_fc1:]))
    child2.fc1.bias.data = torch.cat((parent2.fc1.bias.data[:crossover_point_fc1], parent1.fc1.bias.data[crossover_point_fc1:]))

    crossover_point_fc2 = random.randint(1, 3)

    # Crossover for weights of fc1 layer
    child1.fc2.weight.data = torch.cat((parent1.fc2.weight.data[:crossover_point_fc2], parent2.fc2.weight.data[crossover_point_fc2:]), dim=0)
    child2.fc2.weight.data = torch.cat((parent2.fc2.weight.data[:crossover_point_fc2], parent1.fc2.weight.data[crossover_point_fc2:]), dim=0)

    # Crossover for biases of fc2 layer
    child1.fc2.bias.data = torch.cat((parent1.fc2.bias.data[:crossover_point_fc2], parent2.fc2.bias.data[crossover_point_fc2:]))
    child2.fc2.bias.data = torch.cat((parent2.fc2.bias.data[:crossover_point_fc2], parent1.fc2.bias.data[crossover_point_fc2:]))

    return child1, child2

def mutate(model):
    for param in model.parameters():
        if torch.rand(1).item() < mutation_rate:
            param.data += torch.randn_like(param.data) * mutation_power
    return model

# GENETIC ALGORITHM

population = initialize_population()
for generation in range(num_generations):
    highest_fitness = float('-inf')
    best_individual = None
    fitnesses = []

    for individual in population:
        fitness = compute_fitness(individual)
        fitnesses.append(fitness)
        if fitness > highest_fitness:
            highest_fitness = fitness
            best_individual = individual

    print("Highest fitness in generation", generation + 1, ":", highest_fitness)

    min_fitness = min(fitnesses)
    shifted_fitnesses = [fitness + abs(min_fitness) for fitness in fitnesses]
    total_fitness = sum(shifted_fitnesses)
    probabilities = [fitness / total_fitness for fitness in shifted_fitnesses]
    
    next_generation = []

    for i in range(population_size // 2):
        parent1 = random.choices(population, weights=probabilities)[0]
        parent2 = random.choices(population, weights=probabilities)[0]
        child1, child2 = crossover(parent1, parent2)
        child1 = mutate(child1)
        child2 = mutate(child2)
        next_generation.extend([child1, child2])

    population = next_generation

    # Elitism - move best individual to next population
    population[0] = best_individual

    if generation + 1 == num_generations:
        torch.save(best_individual.state_dict(), "best_individual")

        model_state = best_individual.state_dict()
        model_state_json = {key: value.tolist() for key, value in model_state.items()}
        with open("best_individual.json", 'w') as json_file:
            json.dump(model_state_json, json_file, indent=4)
