import numpy as np
import gym


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
#Policy used to define state transitions
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
        states.append(str(i).zfill(4)) #encode each state using zfill (add trailing zeros make it four-elem num), e.g. str(1).zfill(4) = 0001, str(23).zfill(4) = 0023
    return states

#Create dict of state-reward key/val pairs by relating state->action->reward through dict inside dict 
def initialize_Q():
    Q = {}  
    all_states = get_all_states_as_string()
    for state in all_states: #per each possible state add set of rewards per possible actions from state; 0 or 1 for cart; thus Q is matrix size: [state] x 2
        Q[state] = {} #access key for Q by state-string 
        for action in range(env.action_space.n):
            Q[state][action] = 0 #each Q[state] also a dictionary containing state-action key/val pairs 
    return Q

def play_one_game(bins, Q, eps = 0.5):
    observation = env.reset()
    done = False
    count = 0
    state = get_state_as_string(assign_bins(observation, bins)) #create state-vec per rand obs reset, convert to str
    total_reward = 0
    
    while not done:
        count += 1
        #Epsilon set to 0.5 yield random, sub-optimal action 50% of time; select particular action from set of possible actions in current state
        if np.random.uniform() < eps:
            action = env.action_space.sample() #randomly go left or right
        else: #greedy action 
            action = max_dict(Q[state])[0] #go to state, return action associated with max reward (return either 0 for left, 1 for right and reward per each), from tuple access first elem for specific action
            
        observation, reward, done, information = env.step(action)
        total_reward += reward #reward +1 for each balanced step 
        
        if done and count < 200: #if pole-balance terminates before 200 steps, lower value of state-action pairs; max-reward = 200 thus heavily penalize
            reward = -300
        
        next_state = get_state_as_string(assign_bins(observation, bins)) #next state chosen as per bin assignment policy since agent learns from experience; no model or markov decision process exists to define transitions
        next_state_action, next_state_return = max_dict(Q[next_state]) #return optimal action & associated reward of following state based on observations from current
        Q[state][action] += ALPHA * (reward + GAMMA * next_state_return - Q[state][action]) #update value of current state based on future discounted returns 
        state, action = next_state, next_state_action #continue for following time-step until either pole tips over or cart reaches 200 points
        
    return total_reward, count 


    