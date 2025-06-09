from collections import deque
from dqn import DQN, DQN_Trainer
from game import SnakeAI, CELLSIZE, SCREEN_SIZE
from helperClass import get_dangers, get_direction, get_fruit_position
import numpy as np
import random
import torch

EPSILON_MIN = 0.05
BATCH_SIZE = 1000

class Agent:
    def __init__(self):
        self.epsilon = 1
        self.gamma = 0.9
        self.memory = deque(maxlen=10000)
        self.model = DQN(11,64,3) # try smaller hidden layer and see how it goes
        self.trainer = DQN_Trainer(self.model, 0.001, self.gamma)

    def get_state(self, game: SnakeAI):
        state = [
            # get direction 
            *get_direction(game.moveDir),
            # get dangers
            *get_dangers(game.snake_segments, game.moveDir, CELLSIZE, SCREEN_SIZE),
            # get fruit direction
            *get_fruit_position(game.head,game.fruit)
        ]
        #print(np.array(state,dtype=float))
        return np.array(state,dtype=float)

    def add_memory(self, state, action, reward, state_next, done):
        # state, action, reward, state_next, done
        self.memory.append((state,action,reward,state_next,done))

    def get_action(self, state):
        self.epsilon *= 0.995
        direction = [0,0,0]
 
        # explore
        if random.random() < self.epsilon:
            direction[random.randint(0,2)] = 1
        else:
            # exploit
            state = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state)
            action = torch.argmax(prediction).item()
            direction[action] = 1
            #print(prediction, direction)
        return direction
    
    def train_main_network(self, state, action, reward, state_next, done):
        if len(self.memory) > BATCH_SIZE:
            sample = random.sample(self.memory, BATCH_SIZE)
            state, action, reward, state_next, done = zip(*sample)

        self.trainer.train_step(state, action, reward, state_next, done)





if __name__ == '__main__':
    game = SnakeAI(640,640,5)
    agent = Agent()
    while True:
        state = agent.get_state(game)

        action = agent.get_action(state)

        reward, done, score = game.gameLoop(action)

        state_next = agent.get_state(game)

        agent.add_memory(state, action, reward, state_next, done)

        agent.train_main_network(state, action, reward, state_next, done)

        





