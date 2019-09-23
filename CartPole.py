import q_learning

import gym

import numpy as np

env= gym.make('CartPole-v0')

n_episodes = 500


output_dir = "cartpole_model/"

state_size = 4
action_size =2
batch_size = 32

agent = q_learning.Agent(state_size, action_size) 

done = False  # initialy game is not completed

for e in range(n_episodes): 
    state = env.reset()
    state = np.reshape(state,[1,state_size])  ## state ko reshape kiya (1,4) shape mein
    
    for time in range(500):
        env.render()
        
        action = agent.act(state) #action is 0 or 1
        
        next_state,reward,done,other_info = env.step(action) 
        
        
        reward = reward if not done else -10
        
        next_state = np.reshape(next_state,[1,state_size])
        
        agent.remember(state,action,reward,next_state,done)
        
        state = next_state
        
        if done:
            print("Game Episode :{}/{}, High Score:{},Exploration Rate:{:.2}".format(e,n_episodes,time,agent.epsilon))
            break
            
    if len(agent.memory)>batch_size:
        agent.train(batch_size)
    
   # if e%50==0:
    #    agent.save(output_dir+"weights_"+'{:04d}'.format(e)+".hdf5")
        
print("Deep Q-Learner Model Trained!")
env.close()