
import pickle
import torch
import time

import gym
import matplotlib.pyplot as plt
import numpy as np

from agents.eqrdqn.eqrdqn import EQRDQN
from agents.common.networks.mlp import MLP_Multihead

AGENT_PATH = '???'

env = gym.make("CartPole-v0")

agent = EQRDQN(env,MLP_Multihead,n_quantiles=50)

agent.load(AGENT_PATH)

obs = env.reset()
returns = 0
for i in range(10000):
    net1,net2 = agent.network(torch.FloatTensor(obs))
    net1 = net1.view(agent.env.action_space.n,agent.n_quantiles)
    net2 = net2.view(agent.env.action_space.n,agent.n_quantiles)
    
    plt.cla()
    #plt.axis([0,20,0,10])
    pred1 = (np.array(net1[0,:].detach()) + np.array(net2[0,:].detach()))/2
    pred2 = (np.array(net1[1,:].detach()) + np.array(net2[1,:].detach()))/2
    plt.plot(np.array(net1[0,:].detach()), 'r', label="left")
    plt.plot(np.array(net2[0,:].detach()), 'r', label="left")
    plt.plot(np.array(net1[1,:].detach()), 'g', label="right")
    plt.plot(np.array(net2[1,:].detach()), 'g', label="right")
    plt.legend()
    plt.draw()
    plt.pause(0.01) 
    
    uncertainty = torch.sqrt(torch.mean((net1-net2)**2,dim=1)/2).detach()
    means = torch.mean((net1+net2)/2,dim=1).detach()
    action = agent.predict(torch.FloatTensor(obs),thompson_sampling=True)
    obs, rew, done, info = env.step(action)
    env.render()
    time.sleep(0.05)
    returns += rew
    print(action, "means",means,"uncertainties",uncertainty)
    if done:
        obs = env.reset()
        print(returns)
        returns = 0