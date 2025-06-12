from collections import deque
from dqn import DQN, DQN_Trainer
from game import SnakeAI, CELLSIZE, SCREEN_SIZE
from helperClass import get_dangers, get_direction, get_fruit_position
import numpy as np
import random
import torch
import matplotlib.pyplot as plt

EPSILON_MIN = 0.05
BATCH_SIZE = 1000
TARGET_NETWORK_UPDATE_RATE = 100

plt.ion()
fig, ax = plt.subplots()
num_game = 0
game_count = []
score_history = []
avg_score = 0


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
        print(self.epsilon)
        self.epsilon = max(EPSILON_MIN, self.epsilon - 1/10000)
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
    
    def train_long_memory(self, state, action, reward, state_next, done):
            current_memory_size = min(BATCH_SIZE, len(self.memory))
            sample = random.sample(self.memory, current_memory_size)
            state, action, reward, state_next, done = zip(*sample)
            self.trainer.train_step(state, action, reward, state_next, done)

    def train_short_memory(self, state, action, reward, state_next, done):
        self.trainer.train_step(state, action, reward, state_next, done)

    def draw_graph_score(self):
        ax.clear()
        ax.plot(game_count, score_history, label='Score')
        ax.set_title("Score Over Time")
        ax.set_xlabel("Game #")
        ax.set_ylabel("Score")
        ax.legend()
        plt.axhline(y=avg_score , color='red', linestyle='--', label='Avg. Score')
        plt.legend()
        plt.show()
        plt.pause(0.001)






if __name__ == '__main__':
    agent = Agent()
    game = SnakeAI(640,640,2000, agent)
    counter = 0
    while True:
        state = agent.get_state(game)

        action = agent.get_action(state)

        reward, done, score = game.gameLoop(action)

        state_next = agent.get_state(game)

        agent.add_memory(state, action, reward, state_next, done)
        
        # train network on the past and next (current) frame
        agent.train_short_memory(state, action, reward, state_next, done)

        if done:
            agent.train_long_memory(state, action, reward, state_next, done)
            num_game += 1
            score_history.append(score)
            game_count.append(num_game)
            avg_score = sum(score_history)/num_game
            agent.draw_graph_score()
            


        counter += 1
        if counter == TARGET_NETWORK_UPDATE_RATE:
            agent.trainer.update_target_network()
            counter = 0


    
        
        


        





