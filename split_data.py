import tensorflow as tf
import airsim
from collections import deque

import random
import numpy as np
import time
import os
import pickle
from collections import deque
from tqdm import tqdm
import cv2
import copy

# basic setting
ACTION_NUMS = 13        # number of valid actions
GAMMA = 0.99            # decay rate of past observations
OBSERVE = 50            # time steps to observe before training
EXPLORE = 20000.        # frames over which to anneal epsilon
FINAL_EPSILON = 0.0001  # final value of epsilon
INITIAL_EPSILON = 0.1   # starting value of epsilon
EPSILON_DECAY_START = 20
MEMORY_SIZE = 200    # number of previous transitions to remember
MINI_BATCH = 32         # size of mini batch
MAX_EPISODE = 20000
DEPTH_IMAGE_WIDTH = 256
DEPTH_IMAGE_HEIGHT = 144

TAU = 0.001             # Rate to update target network toward primary network
flatten_len = 9216      # the input shape before full connect layer
NumBufferFrames = 4     # take the latest 4 frames as input



def store_transition(replay_experiences,store_or_read):
    store_path = 'replay_experiences_new.pkl'
    if(store_or_read=='read'):
        if not os.path.exists(store_path) or os.path.getsize(store_path)==0:
        # if not os.path.exists(store_path):
            print('Not Found the pkl file!')
            return replay_experiences
        else:
            store_file = open(store_path,'rb')
            replay_experiences = pickle.load(store_file)
            store_file.close()
            memory_len = len(replay_experiences)
            print('Successfully load the replay_experiences.pkl, %05d memory'%memory_len)
            return replay_experiences
    elif(store_or_read=='store'):
        store_file = open(store_path, 'wb')
        pickle.dump(replay_experiences, store_file)
        store_file.close()
        return 1
    else:
        return 0

# store_path = 'replay_experiences_new2.pkl'
store_path = 'imitaion_data.pkl'
# store_path = 'augmented_data.pkl'
store_file = open(store_path,'rb')
# global replay_experiences
replay_experiences = pickle.load(store_file)
store_file.close()
memory_len = len(replay_experiences)
print('Successfully load the replay_experiences.pkl, %05d memory'%memory_len)


index = list(range(0,int(len(replay_experiences))))
random.shuffle(index)

# data = random.shuffle(replay_experiences)
train_data = []
train_data_len = 0.9*len(replay_experiences)
for i in range(int(train_data_len)):
    train_data.append(replay_experiences[index[i]])

test_data = []
test_data_len = 0.1*len(replay_experiences)
for i in range(int(test_data_len)):
    test_data.append(replay_experiences[index[int(0.9*len(replay_experiences))+i]])

print(len(train_data),len(test_data))


store_file = open('training_augmented.pkl', 'wb')
pickle.dump(train_data, store_file)
store_file.close()

store_file = open('validation_augmented.pkl', 'wb')
pickle.dump(test_data, store_file)
store_file.close()