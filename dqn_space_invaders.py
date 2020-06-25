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

#Class to define architecture of DNN, no training/learning/init done 
class DQNetwork:
    def __init__(self, state_size, action_size, learning_rate, name='DQNetwork'):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        
        with tf.variable_scope(name):
            self.inputs_ = tf.placeholder(tf.float32, [None, *state_size], name="inputs")
            self.actions_ = tf.placeholder(tf.float32, [None, self.action_size], name="actions_")
            
            #sample episilon-greedy action step into env, observe next states, rewards and provide to DNN upon init
            #purpose of stepping through env is to gain exp and iteratively train DNN by minimizing loss 
            #receive set of q-values per actions from DNN, select max and set as target_Q
            #store experience (reward, action, state, next_state) at time step in replay memory
            #using experiences, compute target Q-value which is desirable distribution DNN must approximate
            #iterate to following state
            #load up experiences and provide as vector of inputs to DNN and compute prediction per each experience
            #apply loss function that computes error between DNN output and optimal distribution from stored experiences in memory
            #backpropagate computations to tune weights & repeat; DNN is learning 
            self.target_Q = tf.placeholder(tf.float32, [None], name="target") 
            
            self.conv1 = tf.layers.conv2d(inputs=self.inputs_, 
                                          filters=32,
                                          kernel_size=[8,8],
                                          strides=[4,4],
                                          padding="VALID",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                          name="conv1")
            self.conv1_out = tf.nn.elu(self.conv1, name="conv1_out")
            
            self.conv2 = tf.layers.conv2d(inputs=self.conv1_out, 
                                          filters=64,
                                          kernel_size=[4,4],
                                          strides=[2,2],
                                          padding="VALID",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                          name="conv2")
            self.conv2_out = tf.nn.elu(self.conv2, name="conv2_out")

            
            self.conv1 = tf.layers.conv2d(inputs=self.conv2_out, 
                                          filters=64,
                                          kernel_size=[3,3],
                                          strides=[2,2],
                                          padding="VALID",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                          name="conv3")
            self.conv3_out = tf.nn.elu(self.conv3, name="conv3_out")
            
            self.flatten = tf.contrib.layers.flatten(self.conv3_out)
            
            self.fc = tf.layers.dense(inputs=self.flatten,
                                      units=512,
                                      activation=tf.elu.nn,
                                      kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                      name="fc1")
            
            self.output = tf.layers.dense(inputs=self.fc,
                                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                          units=self.action_size,
                                          activation=None)
            
            self.Q = tf.reduce_sum(tf.multiply(self.output, self.actions_)) #predicted Q-value computed by DNN
            self.loss = tf.reduce_mean(tf.square(self.target_Q - self.Q))
            
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            