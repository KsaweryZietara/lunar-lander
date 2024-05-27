import gymnasium as gym
import torch
from neural_network import NeuralNetwork

env = gym.make("LunarLander-v2", render_mode="human")

model = NeuralNetwork()
model.load_state_dict(torch.load("best_individual"))

for _ in range(10):
    score = 0
    observation, info = env.reset()
    terminated = False
    truncated = False
    while not (terminated or truncated):
        input_data = torch.tensor(observation).unsqueeze(0)
        output = model(input_data)
        action = torch.argmax(output).item()
        observation, reward, terminated, truncated, info = env.step(action)
        score += reward
    print("Score:", score)
