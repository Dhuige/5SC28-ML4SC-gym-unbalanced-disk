from collections import defaultdict
import gym, time
import numpy as np
import os
from UnbalancedDisk2 import *
import matplotlib.pyplot as plt 
from collections import defaultdict

def argmax(a):
    a = np.array(a)
    return np.random.choice(np.arange(a.shape[0],dtype=int)[a==np.max(a)])

def Qlearn(env,Qmat, alpha=0.1, epsilon=0.1, gamma=1.0, nsteps=10000,visualize=False, epsilon_decay=False):
     #any new argument be set to zero
    obs_list = np.zeros((nsteps,2))
    reward_list = np.zeros((nsteps,1))
    highest_reward = -1000
    obs_temp = env.reset()
    cur_nsteps = 0
    obs = np.array([round(np.arctan2(obs_temp[0], obs_temp[1]),1),round(obs_temp[2],1)])
    for z in range(nsteps):
        if epsilon_decay:
            epsilon = 1- (z / nsteps)

        if np.random.uniform()<epsilon:
            action = env.action_space.sample()
        else:
            action = argmax([Qmat[obs[0],obs[1],a] for a in range(env.action_space.n)])

        obs_new_temp, reward, done, _ = env.step(action)
        obs_new = np.array([round(np.arctan2(obs_new_temp[0], obs_new_temp[1]),1), round(obs_new_temp[2],1)])
        if done:
            TD = reward - Qmat[obs[0],obs[1],action]
            Qmat[obs[0],obs[1],action] += alpha*TD
            obs_temp = env.reset()
        else:
            Qmax = max(Qmat[obs_new[0], obs_new[1], action_next] for action_next in range(env.action_space.n))
            TD = reward + gamma*Qmax - Qmat[obs[0],obs[1],action]
            Qmat[obs[0],obs[1],action] += alpha*TD
            obs_temp = obs_new
        if z == cur_nsteps + 1000:
            print(f"{z}/{nsteps} -cur_angle {(obs[0]):3f}-cur_vel {obs[1]:.3f}- highest reward: {highest_reward:.4f} cur reward ={reward:.4f} ", end='\r')
            cur_nsteps = z
        highest_reward = max(highest_reward, reward)
        obs_list[z] = obs_temp
        reward_list[z] = reward
        obs = obs_new
        if visualize:
            env.render()
            time.sleep(1/60)
            print(f"{z}/{nsteps} -cur_angle {(obs[0]):3f}-cur_vel {obs[1]:.3f}- highest reward: {highest_reward:.4f} cur reward ={reward:.4f} ", end='\r')

    return Qmat, obs_list, reward_list

fulldata = []
for i, data1 in enumerate(np.arange(0,1.1,0.1)):
    rew_arg = 0
    max_rew = 0

    Qmat = defaultdict(lambda: float(0))
    env = UnbalancedDisk_sincos_cus()
    Qmat,obs_list, reward_list = Qlearn(env, Qmat, alpha=data1, epsilon=0.2, gamma=0.99, nsteps=1_000_00, epsilon_decay=False)
    print("reached the top after: ", np.argmax(reward_list), "iterations")
    pdata = [data1, 0.2, np.argmax(reward_list)]
    for j in range(5):
        Qmat,obs_list, reward_list = Qlearn(env,Qmat, alpha=0.1, epsilon=0, gamma=0.99, nsteps=300,visualize=False)
        rew_arg = np.argmax(reward_list)
        max_rew = np.max(reward_list)
    
    print("reached the top after:", rew_arg/5, "iterations")
    print("max reward:", max_rew/5)
    pdata.append(rew_arg/5)
    pdata.append(max_rew/5)
    env.close()
    fulldata.append(pdata)
    np.savetxt("data.csv", fulldata, delimiter=",")
