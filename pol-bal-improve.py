
import gym
import numpy as np
from gym import wrappers

env = gym.make('CartPole-v0')
bestEpsLength = 0
allEpsLengths = []
optimalWeights = np.zeros(4)

for i in range(100):
    #Randomly weight states and policy will be if dot prod > 0 go right else left 
    #No update on policies or application of state-action pairs for improvement
    #Per each random weighting, 100 time-steps to test env; 100 varn of weights; 10 000 total increments
    #Idea is randomly select policy and choose optimal one (albeit mathematically may be sub-optimal, better policies can exist)
    newWeights = np.random.uniform(-1.0, 1.0, 4)
    lengthPerEpsIterations = [] #Track time-steps each iteration of pole-bal makes 
    for j in range(100): #iterate eps and update state vals at end
        observation = env.reset()
        done = False
        count = 0
        while not done:
            count += 1
            action = 1 if np.dot(observation, newWeights) > 0 else 0 #remember 1 = right, 0 = left in CartPole-v0 env
            observation, reward, done, info = env.step(action) 
            if done:
                break
        lengthPerEpsIterations.append(count) 
    avgEpsLength = float(sum(lengthPerEpsIterations)/len(lengthPerEpsIterations)) 
    if bestEpsLength < avgEpsLength:
        bestEpsLength = avgEpsLength
        optimalWeights = newWeights
    allEpsLengths.append(avgEpsLength)
    if i % 10 == 0:
        print('Best Average Episode Length: ', bestEpsLength, ' With Actions on Most Recent Iteration in Episode: ', count)
print('All episodes: ', allEpsLengths)
print('Game Lasted: ', count)