import tensorflow as tf
import numpy as np
import retro
from skimage import transform
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
from collections import deque
import random
import warnings

env = retro.make(game='SpaceInvaders-Atari2600')
print("Frame size: ", env.observation_space)
print("Actions available: ", env.action_space.n)

def preprocess_frame(frame):
    gray = rgb2gray(frame)
    cropped_frame = gray[8:-12, 4:-12]
    normalized_frame = cropped_frame/255.0
    preprocessed_frame = transform.resize(normalized_frame, [110, 84])
    return preprocessed_frame

#Skip four frames each timestep and stack those frames into queue to provide network sense of position, velocity, acceleration
#Appending frame to deque removes oldest frame on stack; formulate state
def stack_frames(stacked_frames, state, is_new_eps):
    frame = preprocess_frame(state)
    if is_new_eps: #on new eps
        stacked_frames = deque([np.zeros((110, 84), dtype=np.int) for i in range(stack_size)], maxlen=4) #clear stack 
        stacked_frames.append(frame) #init stack on eps
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)

        
        stacked_state = np.stack(stacked_frames, axis=2) #joins frames
    else:
        stacked_frames.append(frame)
        stacked_state = np.stack(stacked_frames, axis=2)
    return stacked_state, stacked_frames

#Model params defining MDP
state_size = [110, 84, 4] #provide stack 4 frames each 110 x 84; each action repeatedly performed for four frames of stack
action_size = env.action_space.n #8 possible actions
learning_rate = 0.00025 #alpha

#Training params defining how DQN will learn
total_eps = 50
max_steps = 50000
batch_size = 64

#Exploration params for epsilon-greedy action selection
explore_start = 1.0 #max exploration prob
explore_stop = 0.01 #min exploration prob
decay_rate = 0.00001 #gamma param extremely low thus agent will value actions taken long ago

#Q-learning params
pretrain_length = batch_size #experiences in mem at init
memory_size = 1000000 #experiences stored in mem to improve convergence time to optimal state-action values

#Preprocessing params
stack_size = 4

training = False
eps_render = False