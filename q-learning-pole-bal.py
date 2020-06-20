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
def create_bins():
    bins = np.zeros((4,10)) #create space for 4 quantities describing observations, 10 possible vals each
    bins[0] = np.linspace(-4.8, 4.8, 10) #defn range of values over interval
    bins[1] = np.linspace(-5, 5, 10) #to improve can check vel distribution and set ranges upon that 
    bins[2] = np.linspace(-0.418, 0.418, 10)
    bins[3] = np.linspace(-5, 5, 10)
    return bins

#Rather than work with raw numerical obs as done prior, instead work with normalized scale positions to form state 
def assign_bins(observations, bins):
    #Map continuous observation quantities, scale down by bin ranges & assign discretized state; bin range must exceed obs range 
    state = np.zeros(4)
    #Formulate four elem-state vec containing bin index; each index rep numerical location on normalized scale for: cart pos, cart vel, pole angle, pole vel 
    #E.g. state = {0, 3, 1, 8} meaning per each scale defined by bins[], state[0] = 0, closer to start of cart pos scale, whereas state[3] = 8, closer to end of pole vel scale
    for i in range(4):
        state[i] = np.digitize(observations[i], bins[i])
    return state

#Convert state-vec from int to string
def get_state_as_string(state):
    string_state = ''.join(str(int(e)) for e in state)
    return string_state

def get_all_states_as_string():
    states = []
    for i in range(MAXSTATES):
        states.append(str(i).zfill(4)) #fill each state as a empty four-elem vec ('0000'), store all in init-states array
    return states

