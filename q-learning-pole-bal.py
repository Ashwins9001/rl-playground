import numpy as np
import gym
import matplotlib.pyplot as plt


#Apply Q-learning with Epsilon-Greedy strategy to find better policies (paths) via exploration by occasionally choosing non-greedy actions
#Discount rewards to prioritise recents and apply policy iteration per each added reward as agent samples state-action pairs (rather than episodically as Monte Carlo)
#Q-learning off-policy therefore behavioral policy formed to estimate optimal target policy
env = gym.make('CartPole-v0')

#Define discount factor gamma (closer to 1 value immediate rewards more) and step size param alpha (scales/normalizes error), set of discrete states for agent
MAXSTATES = 10**4
GAMMA = 0.9
ALPHA = 0.01

#Helper func to find optimal state-action pair from value func
def max_dict(d):
    max_v = float('-inf')
    for key, val in d.items():
        if val > max_v:
            max_v = val
            max_key = key 
    return max_key, max_v

#Require conversion from continuous to discrete space for states 