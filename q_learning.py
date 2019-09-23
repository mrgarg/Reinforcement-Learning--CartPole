# Q_Learning is a way to measure the reward that we would get when taking a particular action 'a' in a state 's'. 
# It is not only a measurment of the immediate reward but a summation of the entire future reward we would get from 
# consequent actions as well. 
# Q(s,a) = r + Y*max(Q(s',a')); where, r is the immediate reward
# Input will have a state matrix, the output matrix from the Neural Network would be a matrix of how good 
# each action is


import numpy as np
import matplotlib.pyplot as plt
import os
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import random

class Agent:
    
    def __init__(self,state_size,action_size):
        
        self.state_size= state_size
        self.action_size = action_size
        self.gamma = 0.95  # discount factor
        self.memory= deque(maxlen=2000)
        
        # exploration vs exploitation Trade off
        # exploration: good in the begining -> helps you to try various random things
        # exploitation: sample good experience from the past(memory)--> good in the end
        
        self.epsilon=1.0  #100 % exploration in the begining
        self.epsilon_decay=0.995
        self.epsilon_min=0.01
        self.learning_rate=0.001
        self.model= self._create_model()
    
    def _create_model(self):
        
        model=Sequential()
        model.add(Dense(24,activation='relu',input_dim=self.state_size))
        model.add(Dense(24,activation='relu'))
        model.add(Dense(self.action_size,activation='linear'))
        model.compile(loss='mse',optimizer=Adam(lr=self.learning_rate))
        return model
    
    
    # We need data to train the neural network, thus we use Replay Buffer Technique to generate data on the fly 
    # and use it for training 
    def remember(self,state,action,reward,next_state,done):
        self.memory.append((state,action,reward,next_state,done))
    
    
    def act(self,state):
        
        # Exploration vs Exploitation
        # sampling according to epsilon greedy method
        
        if np.random.rand()<=self.epsilon:
            return random.randrange(self.action_size)
        else:      ## in else case: ask neural network to give me the suitable action
            return np.argmax(self.model.predict(state)[0])
        
    def train(self,batch_size=32):
        minibatch=random.sample(self.memory,batch_size)
        for state,action,reward,next_state,done in minibatch:
            
            if not done:
                target= reward+ self.gamma*np.max(self.model.predict(next_state)[0])
            else:
                target=reward
            
            target_f = self.model.predict(state)
            target_f[0][action]=target
            
            self.model.fit(state,target_f,epochs=1,verbose=0)
        if self.epsilon > self.epsilon_min:
            
            self.epsilon*=self.epsilon_decay
    def load(self,name):
        self.model.load_weights(name)
        
    def save(self,name):
        self.model.save_weights(name)
    