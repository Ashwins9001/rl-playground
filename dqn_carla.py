import glob
import os
import sys

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass


import carla
import time
import random
import numpy as np
import argparse
import logging
import pygame
import tensorflow as tf
import manual_control
import queue
from PIL import Image
from multiprocessing import Process
from collections import deque

state_size = [84, 84, 1]
# discrete action-space described as (throttle, steer, brake)
action_space = np.array([(0.0, 0.0, 1.0), (0.5, 0.0, 0.0), (1.0, 0.0, 0.0),
                        (0.5, 0.25, 0.0), (0.5, -0.25, 0.0), (0.5, 0.5, 0.0), (0.5, -0.5, 0.0)])
learning_rate= 0.00025

# Training parameters
total_episodes = 5001  # INTIALLY  5000
max_steps = 5000
batch_size = 64

# Fixed Q target hyper parameters
max_tau = 5000  # tau is the C step where we update out target network -- INTIALLY 10000

# exploration hyperparamters for ep. greedy. startegy
explore_start = 1.0  # exploration probability at start
explore_stop = 0.01  # minimum exploration probability
decay_rate = 0.00005  # exponential decay rate for exploration prob


# Q LEARNING hyperparameters
gamma = 0.95  # Discounting rate
pretrain_length = 100000  ## Number of experiences stored in the Memory when initialized for the first time --INTIALLY 100k
memory_size = 100000  # Number of experiences the Memory can keep  --INTIALLY 100k

class DQNetwork():
    def __init__(self, state_size, action_size, learning_rate, name='DQNetwork'):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        
        with tf.variable_scope(name):
            #inputs define image fed into NN
            self.inputs_ = tf.placeholder(tf.float32, [None, *state_size], name="inputs")
            
            #actions define array containing tuple of actions taken by system
            self.actions_ = tf.placeholder(tf.float32, [None, self.action_size], name="actions_")
            
            
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

            
            self.conv3 = tf.layers.conv2d(inputs=self.conv2_out, 
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
            
            self.Q = tf.reduce_sum(tf.multiply(self.output, self.actions_)) #predicted Q-value computed by DNN by associating output of DNN w/ action tuples
            self.loss = tf.reduce_mean(tf.square(self.target_Q - self.Q)) #compute loss per each action-val 
            
            #return gradients for each weight of NN (change in weights after minimizing loss)
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
            

    def predict_action(self, sess, explore_start, explore_stop, decay_rate, decay_step, state):
        # Epsilon greedy strategy: given state s, choose action a ep. greedy
        exp_tradeoff = np.random.rand()
        explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)

        if (explore_probability > exp_tradeoff):
            action_int = np.random.choice(self.action_size)
            action = self.possible_actions[action_int]
        else:
            # get action from Q-network: neural network estimates the Q values
            Qs = sess.run(self.output, feed_dict={self.inputs_: state.reshape((1, *state.shape))})

            # choose the best Q value from the discrete action space (argmax)
            action_int = np.argmax(Qs)
            action = self.possible_actions[int(action_int)]

        return action_int, action, explore_probability

   
class Memory():
    def __init__(self, max_size):
        self.buffer = deque(maxlen = max_size)
    def add(self, experience):
        self.buffer.append(experience)
    def sample(self, batch_size):
        buffer_size = len(self.buffer)
        index = np.random.choice(np.arrange(buffer_size),
                                 size = batch_size,
                                 replace = False)
        return [self.buffer[i] for i in index]
    


#carla sensors special actors to measure & stream data using listen() method
#can retrieve data upon timestep/action; must be attached to parent actor (vehicle)
#listen() method employs lambda func which recursively callback

class Sensors(object):
    """Class to keep track of all sensors added to the vehicle"""

    def __init__(self, world, vehicle):
        super(Sensors, self).__init__()
        self.world = world
        self.vehicle = vehicle
        self.camera_queue = queue.Queue() # queue to store images from buffer
        self.collision_flag = False # Flag for colision detection
        self.lane_crossed = False # Flag for lane crossing detection
        self.lane_crossed_type = '' # Which type of lane was crossed

        self.camera_rgb = self.add_sensors(world, vehicle, 'sensor.camera.rgb')
        self.collision = self.add_sensors(world, vehicle, 'sensor.other.collision')
        self.lane_invasion = self.add_sensors(world, vehicle, 'sensor.other.lane_invasion', sensor_tick = '0.5')

        self.sensor_list = [self.camera_rgb, self.collision, self.lane_invasion]

        #sensor uses lambda func to constantly retrieve data and return it
        #secondary func that uses lambda func will then manipulate data for use
        #such as by setting collision flag on 
        
        self.collision.listen(lambda collisionEvent: self.track_collision(collisionEvent))
        self.camera_rgb.listen(lambda image: self.camera_queue.put(image))
        self.lane_invasion.listen(lambda event: self.on_invasion(event))

    def add_sensors(self, world, vehicle, type, sensor_tick = '0.0'):

        sensor_bp = self.world.get_blueprint_library().find(type)
        try:
            sensor_bp.set_attribute('sensor_tick', sensor_tick)
        except:
            pass
        if type == 'sensor.camera.rgb':
            sensor_bp.set_attribute('image_size_x', '100')
            sensor_bp.set_attribute('image_size_y', '100')

        sensor_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        sensor = self.world.spawn_actor(sensor_bp, sensor_transform, attach_to=vehicle)
        return sensor

    def track_collision(self, collisionEvent):
        '''Whenever a collision occurs, the flag is set to True'''
        self.collision_flag = True

    def reset_sensors(self):
        '''Sets all sensor flags to False'''
        self.collision_flag = False
        self.lane_crossed = False
        self.lane_crossed_type = ''

    def on_invasion(self, event):
        '''Whenever the car crosses the lane, the flag is set to True'''
        lane_types = set(x.type for x in event.crossed_lane_markings)
        text = ['%r' % str(x).split()[-1] for x in lane_types]
        self.lane_crossed_type = text[0]
        self.lane_crossed = True

    def destroy_sensors(self):
        '''Destroy all sensors (Carla actors)'''
        for sensor in self.sensor_list:
            sensor.destroy()


def reset_environment(map, vehicle, sensors):
    vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
    time.sleep(1)
    spawn_points = map.get_spawn_points()
    spawn_point = random.choice(spawn_points) if spawn_points else carla.Transform()
    vehicle.set_transform(spawn_point)
    time.sleep(2)
    sensors.reset_sensors()

def process_image(queue):
    '''get the image from the buffer and process it. It's the state for vision-based systems'''
    image = queue.get()
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    image = Image.fromarray(array).convert('L') # grayscale conversion
    image = np.array(image.resize((84, 84))) # convert to numpy array
    image = np.reshape(image, (84, 84, 1)) # reshape image
    return image


def map_action(action, action_space):
    """ maps discrete actions into actual values to control the car"""
    control = carla.VehicleControl()
    control_sequence = action_space[action]
    control.throttle = control_sequence[0]
    control.steer = control_sequence[1]
    control.brake = control_sequence[2]

    return control

def compute_reward(vehicle, sensors):#, collision_sensor, lane_sensor):
    max_speed = 14
    min_speed = 2
    speed = vehicle.get_velocity()
    vehicle_speed = np.linalg.norm([speed.x, speed.y, speed.z])

    speed_reward = (abs(vehicle_speed) - min_speed) / (max_speed - min_speed)
    lane_reward = 0

    if (vehicle_speed > max_speed) or (vehicle_speed < min_speed):
        speed_reward = -0.05

    if sensors.lane_crossed:
        if sensors.lane_crossed_type == "'Broken'" or sensors.lane_crossed_type == "'NONE'":
            lane_reward = -0.5
            sensors.lane_crossed = False

    if sensors.collision_flag:
        return -1

    else:
        return speed_reward + lane_reward
    
def isDone(reward):
    '''Return True if the episode is finished'''
    if reward <= -1:
        return True
    else:
        return False



def render(clock, world, display):
    clock.tick_busy_loop(30) # this sets the maximum client fps
    world.tick(clock)
    world.render(display)
    pygame.display.flip()
    

def training(map, vehicle, sensors):
    
    
    tf.reset_default_graph()
    agent = DQNetwork(state_size, action_space, learning_rate, name="Agent")
    print("NN init")
    writer = tf.summary.FileWriter("summary")
    tf.summary.scalar("Loss", agent.loss)
    write_op = tf.summary.merge_all()
    saver = tf.train.Saver()
    
    #init memory 
    memory = Memory(max_size = memory_size)
    print("memory init")
    
    with tf.Session() as sess:
        print("session beginning")
        sess.run(tf.global_variables_initializer())
        writer.add_graph(sess.graph)
        m = 0
        decay_step = 0
        tau = 0
        print("beginning training")
        for episode in range(1, total_episodes):
            #init episode
            print("env reset")
            reset_environment(map, vehicle, sensors)
            state = process_image(sensors.camera_queue)
            done = False
            start = time.time()
            episode_reward = 0
        
            #step through episode & retrieve data from DNN
            for step in range(max_steps):
                tau += 1
                decay_step += 1
                action_int, action, explore_probability = agent.predict_action(explore_start, explore_stop, decay_rate, decay_step, state)
                print("action from NN received")
                car_controls = map_action(action_int, action_space)
                vehicle.apply_control(car_controls)
                print("action applied to car")
                time.sleep(0.25)
                next_state = process_image(sensors.camera_queue)
                reward = compute_reward(vehicle, sensors)
                print("reward computed: " + reward)
                
                
                episode_reward += reward
                done = isDone(reward)
                memory.add((state, action, reward, next_state, done))
                state = next_state
            
                #begin learning by sampling a batch from memory
                batch = memory.sample(batch_size)
                
                s_mb = np.array([each[0] for each in batch], ndmin = 3)
                a_mb = np.array([each[1] for each in batch])
                r_mb = np.array([each[2] for each in batch])
                next_s_mb = np.array([each[3] for each in batch], ndmin = 3)
                dones_mb = np.array([each[4] for each in batch])
                
                target_Qs_batch = []

                #q-val for all next states to compute target q-val for current state
                Qs_next_state = sess.run(agent.output, feed_dict={agent.inputs_: next_s_mb})
                
                for i in range(0, len(batch)):
                    terminal = dones_mb[i] #check if on last state of eps
                    if terminal:
                        target_Qs_batch.append((r_mb[i])) #if last state, append reward
                    else:
                        #formulate target q-vals by feed-fwd in network, using old weights for comparison 
                        target = r_mb[i] + gamma*np.max(Qs_next_state[i])
                        target_Qs_batch.append(target)
                targets_mb = np.array([each for each in target_Qs_batch])
                
                #run session to compute loss & change NN weights, "newly" learned weights compared against old ones in training
                #feed in state inputs, actions to associate q-vals w/, and target q-vals for loss computation
                loss, _  = sess.run([agent.loss, agent.optimizer], feed_dict={agent.inputs_: s_mb, agent.target_Q: targets_mb, agent.actions_:a_mb})
                summary = sess.run(write_op, feed_dict={agent.inputs_: s_mb, agent.target_Q: targets_mb, agent.actions_:a_mb})
                writer.add_summary(summary, episode)
                writer.flush
                
                print('Episode: {}'.format(episode),
                                  'Total reward: {}'.format(episode_reward),
                                  'Explore P: {:.4f}'.format(explore_probability),
                                'Training Loss {:.4f}'.format(loss))
               
                    
def control_loop(vehicle_id, host, port):
    actor_list = []
    try:
        #setup Carla
        client = carla.Client(host, port)
        client.set_timeout(15.0)
        world = client.get_world()
        map = world.get_map()
        vehicle = next((x for x in world.get_actors() if x.id == vehicle_id), None) #get the vehicle actor according to its id
        sensors = Sensors(world, vehicle)
        print("beginning training loop")
        training(map, vehicle, sensors)   
        
    finally:
        print("done")
        #sensors.destroy_sensors()       

def render_loop(args):
    #loop responsible for rendering the simulation client
    pygame.init()
    pygame.font.init()
    world = None
    
    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(15.0)

        hud = manual_control.HUD(args.width, args.height)
        world = manual_control.World(client.get_world(), hud, args)
        print("beginning process loop")
        control_loop(world.player.id, args.host, args.port)
        
    finally:

        pygame.quit()


def main():
    argparser = argparse.ArgumentParser(
        description='CARLA RL')
    argparser.add_argument(
        '--test',
        action='store_true',
        dest='test',
        help='test a trained model')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=5000,
        type=int,
        help='TCP port to listen to (default: 5000)')
    argparser.add_argument(
        '-a', '--autopilot',
        action='store_true',
        help='enable autopilot')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='800x600',
        help='window resolution (default: 800x600)')
    argparser.add_argument(
        '--filter',
        metavar='PATTERN',
        default='vehicle.audi.tt',
        help='actor filter (default: "vehicle.audi.tt")')
    argparser.add_argument(
        '--rolename',
        metavar='NAME',
        default='hero',
        help='actor role name (default: "hero")')
    argparser.add_argument(
        '--gamma',
        metavar='NAME',
        default='0',
        help='gamma correction')
    
    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split('x')]

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    print(__doc__)

    try:

        render_loop(args)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':

    main()