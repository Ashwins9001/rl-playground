import gym
import numpy as np


#Select env for agent to play in; cart-pole goal is to self-balance cart
#Action space: '0' for pushing left, '1' for pushing right
#State space: '0' for cart pos, '1' for cart vel, '2' for pole angle, '3' for pole vel
#Reward given for time-step pole balanced
env = gym.make('CartPole-v0')
count = 0
env.reset() #reset all states & return observation which displays state vals
for i in range(10000):
    env.render() #render frame to show
    action = env.action_space.sample()
    print('Action: ', action)
    observation, reward, done, info = env.step(action) #apply action and step env by timestep; policy to select action rand
    print('Observation: ', observation)
    print('Reward: ', reward)
    print('Info: ', info)
    count += 1
    if(done): #ensure program ends if terminal state reached 
        break
print(count)
env.close()
