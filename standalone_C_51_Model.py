import gymnasium as gym
import tensorflow as tf
import numpy as np
import torch
import torch.nn as nn
import random
import matplotlib.pyplot as plt
from gymnasium.experimental.wrappers import HumanRenderingV0
from statistics import mean
from game.env import QWOPEnv
import recorder






import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'






class C51:

    def __init__(self, action, state):
        #defining the parameters
        self.num_atoms = 51
        self.action_space = action
        self.state_space = state
        self.reward = 0
        self.num_actions = 11

        self.curr_action = np.zeros([self.num_actions,51])
        self.curr_rew = 0

        self.gamma = .9
        self.epsilon = 50
        self.V_min = -10
        self.V_max = 10
        self.d_z = (self.V_max - self.V_min) / (self.num_atoms - 1)
        self.z = np.linspace(self.V_min, self.V_max, self.num_atoms, dtype=np.float32)

        # neural network network
        self.model = nn.Sequential(nn.Linear(71, 128), nn.ReLU(),
                                   nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, self.num_actions*self.num_atoms))

        #Defining the optimizer and Loss functions
        self.lr = .0002
        self.mseloss = nn.MSELoss()
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.lr)
    def target(self, z):

        #creating the target distributions

        prediction = np.zeros([self.num_actions,51])
        i = 0
        #print(z.shape)
        while i < self.num_actions:
            j = 0
            while j < self.num_atoms:
                dist = min(self.V_max, max(self.V_min, (self.reward + (self.gamma*self.curr_action[i][j]))))
                bdist = (dist - self.V_min)/self.d_z

                prediction[i][j] == bdist
                j+=1
            i+=1

        #Shifting the prediction distribution using loss
        prediction = torch.tensor(prediction, dtype=torch.float32, requires_grad=True)
        #print(prediction)
        prediction = prediction.view(-1,self.num_actions,self.num_atoms)
        self.curr_action = z
        target = torch.tensor(z, dtype=torch.float32, requires_grad=True)
        #print(target)
        #print(prediction.shape)
        #print(target.shape)
        loss = self.mseloss(target, prediction)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        return z

    def optimal_act(self, state):

        #Using the Neural Network to find the optimal prediction distribution

        state = torch.tensor(state, dtype = torch.float32, requires_grad = False)
        z = self.model(state)
        z = z.view(-1, self.num_actions, self.num_atoms)
        z = z.squeeze()
        z = z.detach().numpy()
        #print(z.shape)
        distr = self.target(z)
        #print(z.shape)
        q = np.sum(np.multiply(z, np.array(self.z)), axis = 1)
        next_action = np.argmax(q)

        return next_action
    def next_action(self, state):
        random_number = np.random.randint(0,100)
        #print(random_number)
        if 35 >= random_number:

            #Creating the random Distribution for exploration

            i = 0
            action = np.zeros([self.num_actions,51])
            while i < self.num_actions:
                j = 0
                while j < self.num_atoms:
                    num = random.randrange(0,2)

                    action[i,j] = num
                    j+=1
                i+=1

            distri = self.target(action)
            q = np.sum(np.multiply(action, np.array(self.z)), axis=1)
            next_action = np.argmax(q)
        else:

            #Getting the optimal action for exploitation

            next_action = self.optimal_act(state)
        #print(next_action)
        return next_action

    def scheduler(self, epsilon):
        return epsilon/4

    '''

    def replay():
    '''

def run_model(game_env, act_num):

    #Starting the environment and the agent

    walk = game_env(render_mode = 'human', fine_tune = False, intermediate_rewards = True)

    obs = walk.reset()[0]

    current_state = obs


    act = walk.action_space
    #print(act.shape)
    #print(act)
    state = walk.observation_space
    agent = C51(act,state)


    reward_plot = []
    mean_reward_plot = []
    print(act)
    for i in range(200):

        #reset for each episode

        state = walk.reset()[0]
        end = False
        reward = 0

        while end == False:

            #finding and taking the next action

            action = agent.next_action(state)
            obs, rew, term, trun, info = walk.step(action)
            agent.reward = rew
            reward += rew
            if i % 6000 == 0:
                agent.epsilon = agent.scheduler(agent.epsilon)
            if term or trun:

                #the end of the episode

                obs = walk.reset()[0]
                current_state = obs
                end = True

            state = obs
        # agent.curr_action = action

        #computing the rewards per episode

        reward_plot.append(reward)
        rewa = mean(reward_plot)
        mean_reward_plot.append(rewa)
        print('epochs: ', i, ' reward:', rewa)
        # reward_plot.append(reward)
    # recording.close()
    walk.close()

    #Creating a plot to show the mean rewards per episode
    reward_plot = np.array(reward_plot)
    # print(reward_plot)
    plt.plot(np.arange(1, 201), mean_reward_plot)
    plt.title('reward v episodes')
    plt.show()

act_num = 4
run_model(QWOPEnv, 11)


