import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class DQN(nn.Module):
    
    def __init__(self, input_num, hidden_num, output_num):
        super().__init__()
        self.layer1 = nn.Linear(input_num, hidden_num)
        self.layer2 = nn.Linear(hidden_num, output_num)

    def forward(self,x):
        x = F.relu(self.layer1(x))
        x = self.layer2(x)
        return x
    

class DQN_Trainer:
    def __init__(self, model, lr ,gamma):
        self.model = model
        self.lr = lr
        self.gamma = gamma  
        self.optimizer = optim.Adam(self.model.parameters(),self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, state_next, done):
        state = torch.tensor(state,dtype=torch.float)
        action = torch.tensor(action,dtype=torch.long)
        reward = torch.tensor(reward,dtype=torch.float)
        state_next = torch.tensor(state_next,dtype=torch.float)

        if len(state.shape) == 1:
            state = torch.unsqueeze(state,0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            state_next = torch.unsqueeze(state_next, 0)
            done = (done,)

        prediction = self.model(state)
        target = prediction.clone()

        # target = reward + gamma * Qopt(state_next) * (1 - done)
        for i in range(len(state)):
            Q_new = reward[i]
            if not done[i]:
                Q_new = reward[i] + (self.gamma * torch.max(self.model(state_next[i])))

            # updates predicted q-value of the main model with the target q-value
            target[i][torch.argmax(action[i]).item()] = Q_new
        
        self.optimizer.zero_grad()
        loss = self.criterion(target, prediction)
        loss.backward()
        self.optimizer.step()
        