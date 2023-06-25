#%%
import time
from copy import deepcopy

import gym
import gym.envs.classic_control
import gym.wrappers
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from UnbalancedDisk3 import *


class Qfunction(nn.Module):
    def __init__(self, env):
        super(Qfunction,self).__init__()
        self.lay1 = nn.Linear(2, 40)  
        self.F1 =  nn.Tanh()  
        self.lay2 = nn.Linear(40,env.action_space.n)  
    
    def forward(self, obs):
        return self.lay2(self.F1(self.lay1(obs)))  

def rollout(Q, env, epsilon=0.1, N_rollout=10_000): 
    #save the following (use .append)
    Start_state = np.zeros((2,N_rollout)) #hold an array of (x_t)
    Actions = np.zeros((1,N_rollout)) #hold an array of (u_t)
    Rewards =  np.zeros((1,N_rollout)) #hold an array of (r_{t+1})
    End_state =  np.zeros((2,N_rollout)) #hold an array of (x_{t+1})
    Terminal =  np.zeros((2,N_rollout)) #hold an array of (terminal_{t+1})
    # Qfun( a numpy array of the obs) -> a numpy array of Q values
    Qfun = lambda x: Q(torch.tensor(x[None,:],dtype=torch.float32))[0].numpy() 
    with torch.no_grad():
        
        obs_temp = env.reset() 
        obs = convert_to_angle(obs_temp)
        for i in range(N_rollout):  
            if np.random.uniform()>epsilon:  
                Qnow = Qfun(obs)  
                action = np.argmax(Qnow)  
            else:  
                action = env.action_space.sample()  
            Start_state[:,i] = obs
            Actions[0,i]=action

            obs_next_temp, reward, done, info = env.step(action)  
            obs_next = convert_to_angle(obs_next_temp)
 
            Rewards[0,i] = reward
            End_state[:,i] = obs_next
            obs = obs_next  
            
        obs = env.reset()  
            
                
                
    #error checking:
    return np.array(Start_state), np.array(Actions), np.array(Rewards), np.array(End_state), np.array(Terminal).astype(int)



def eval_Q(Q,env,nsteps=250):
    with torch.no_grad():
        Qfun = lambda x: Q(torch.tensor(x[None,:],dtype=torch.float32))[0].numpy()
        rewards_acc = 0  
        obs_temp = env.reset()  
        obs = convert_to_angle(obs_temp)
        for i in range(nsteps):  
            action = np.argmax(Qfun(obs))  
            obs_temp, reward, done, info = env.step(action)  
            obs = convert_to_angle(obs_temp)
            rewards_acc += reward  
           
        return rewards_acc  



def DQN_rollout(Q, optimizer, env, gamma=0.98, use_target_net=True, N_iterations=21, N_rollout=20000, \
                N_epochs=10, batch_size=32, N_evals=10, target_net_update_feq=100):
    best = -float('inf')
    torch.save(Q.state_dict(),'Q-checkpoint')
    
    for iteration in range(N_iterations):
        epsilon = 1.0 - iteration/(N_iterations-1) 
        print(f'rollout iteration {iteration} with epsilon={epsilon:.2%}...')
        
        #2. rollout
        Start_state, Actions, Rewards, End_state, Dones = rollout(Q, env, epsilon=epsilon, N_rollout=N_rollout)   
        
        #Data conversion, no changes required
        convert = lambda x: [torch.tensor(xi,dtype=torch.float32) for xi in x]
        Start_state, Rewards, End_state, Dones = convert([Start_state, Rewards, End_state, Dones])
        Actions = Actions.astype(int)

        print('starting training on rollout information...')
        t = 0
        for epoch in range(N_epochs): 
            for i in range(batch_size,len(Start_state)+1,batch_size): 
                if t%target_net_update_feq==0:
                    Qtarget = deepcopy(Q)  
                    
                t += 1
                
                Start_state_batch, Actions_batch, Rewards_batch, End_state_batch, Dones_batch = [d[i-batch_size:i] for d in [Start_state, Actions, Rewards, End_state, Dones]] 
                
                with torch.no_grad(): 
                    if use_target_net:
                        
                        maxQ = torch.max(Qtarget(End_state_batch),dim=1)[0]  
                    else:
                        maxQ = torch.max(Q(End_state_batch),dim=1)[0]
                
                action_index = np.stack((np.arange(batch_size),Actions_batch),axis=0)
                Qnow = Q(Start_state_batch)[action_index]
                
                Loss = torch.mean((Rewards_batch + gamma*maxQ - Qnow)**2)  
                print(f">>>>>>>>>>>>>>>>>>Loss: {Loss:.2f}<<<<<<<<<<<<<<<<<<<<<<")
                optimizer.zero_grad() 
                Loss.backward()   
                optimizer.step() 
            
            score = np.mean([eval_Q(Q,env) for i in range(N_evals)]) 
            
            print(f'iteration={iteration} epoch={epoch} Average Reward per episode:',score)
            if score>best:
                best = score
                print('################################# \n new best',best,'saving Q... \n#################################')
                torch.save(Q.state_dict(),'Q-checkpoint')
        
        print('loading best result')
        Q.load_state_dict(torch.load('Q-checkpoint'))
    
    print('loading best result')
    Q.load_state_dict(torch.load('Q-checkpoint'))
def convert_to_angle(obs):
    return np.array([round(np.arctan2(obs[0], obs[1]),1),round(obs[2],1)])
#%%
def visualize(env, nsteps=10000,visualize=True):
     #any new argument be set to zero
    
    obs_list = np.zeros((nsteps,2))
    reward_list = np.zeros((nsteps,1))
    highest_reward = -1000
    cur_nsteps = 0   
    obs_temp = env.reset()

    obs = convert_to_angle(obs_temp)
    #load the Qfunction weights
    Qfunc = Qfunction(env)
    Qfunc.load_state_dict(torch.load('Q-checkpoint'))
    convert = lambda x: [torch.tensor(xi,dtype=torch.float32) for xi in x]    
    for z in range(nsteps):
        obs = torch.tensor(convert(obs))
        
        action = torch.argmax(Qfunc(obs))

        obs_new_temp, reward, done, _ = env.step(action)
        obs_new = convert_to_angle(obs_new_temp)
        highest_reward = max(highest_reward, reward)
        obs_list[z] = obs
        reward_list[z] = reward
        obs = obs_new
        if visualize:
            env.render()
            time.sleep(1/60)
            print(f"{z}/{nsteps} -cur_angle {(obs[0]):3f}-cur_vel {obs[1]:.3f}- highest reward: {highest_reward:.4f} cur reward ={reward:.4f} ", end='\r')

#%%
max_episode_steps = 250
env = UnbalancedDisk_sincos_cus()

Q = Qfunction(env)
#test validity:
obs_temp = env.reset()
obs = convert_to_angle(obs_temp)
#%%
obs_tensor = torch.tensor(obs,dtype=torch.float32)[None,:] #convert to an torch tensor with size (1, Nobs=6)

gamma = 0.98; batch_size = 16; N_iterations = 2; N_rollout = 20000; N_epochs = 20; N_evals = 10; lr = 0.0005

Start_state, Actions, Rewards, End_state, Terminal = rollout(Q,env,N_rollout=300)
print(Start_state, Actions, Rewards, End_state, Terminal)


Q = Qfunction(env)
optimizer = torch.optim.Adam(Q.parameters(),lr=lr) #low learning rate
DQN_rollout(Q, optimizer, env, use_target_net=True, gamma=gamma, N_iterations=N_iterations, \
            N_rollout=N_rollout, N_epochs=N_epochs, N_evals=N_evals)



# %%

visualize(env, nsteps=10000,visualize=True)


# %%
