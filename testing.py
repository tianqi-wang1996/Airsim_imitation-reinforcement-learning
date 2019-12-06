from tensorboardX import SummaryWriter

import airsim
from collections import deque
import torch
import random
import numpy as np
import time
import os
import pickle
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from model.baseline import ResNet34, ResNet50
import copy
from tqdm import tqdm

from ddpg import DDPG, OUNoise

num_steps = 1000
num_episodes = 1000

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


net = ResNet34()
print("Model Instantiated")


checkpoint = torch.load('NEW_ResNet34_augmented-80-0.0034.pth')

net.load_state_dict(checkpoint)
print('Restoring parameters')

net = net.to(device)
net.eval()




client = airsim.CarClient()
client.confirmConnection()
print('Connect succcefully!')
client.enableApiControl(True)

client.simEnableWeather(True)

car_controls = airsim.CarControls()
car_controls.throttle = 0
car_controls.steering = 0
client.reset()
print('Environment initialized!')


def get_image_velocity_done(client):
    responses = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)])
    response = responses[0]
    kinematics  = client.getCarState().kinematics_estimated
    collision_info = client.simGetCollisionInfo()
    done = int(collision_info.has_collided == True)

    img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)
    img = img1d.reshape(response.height, response.width, 4)
    img = img[:, :, 0:3]
    
    velocity_x = kinematics.linear_velocity.x_val+1.6
    velocity_y = kinematics.linear_velocity.y_val-15
    velocity = np.sqrt(velocity_x**2+velocity_y**2) 
    velocity  = np.array([velocity])[0] 
    velocity = torch.tensor(velocity).float().unsqueeze(0).unsqueeze(0)

    return img, velocity, done


def reset_initial_position_weather(client):
    client.reset()
    client.enableApiControl(True)
    client.simSetWeatherParameter(airsim.WeatherParameter.Rain, 0)
    client.simSetWeatherParameter(airsim.WeatherParameter.Snow, 0)
    client.simSetWeatherParameter(airsim.WeatherParameter.MapleLeaf, 0)
    road_choice = random.randint(1,7)
    direction_choice = random.randint(1,2)
    road_position = random.random()
    # print(road_choice, direction_choice, road_position)
    x = 0
    y = 0
    z_angle = 0
    if road_choice == 1:
        x = -127.6 + (128.4+127.6)*road_position
        y = -129
        z_angle = (direction_choice-2)*0 + (direction_choice-1)*3.1416
    elif road_choice == 2:
        x = -127.6 + (128.4+127.6)*road_position
        y = -1.1
        z_angle = (direction_choice-2)*0 + (direction_choice-1)*3.1416
    elif road_choice == 3:
        x = -127.6 + (128.4+127.6)*road_position
        y = 127
        z_angle = (direction_choice-2)*0 + (direction_choice-1)*3.1416
    elif road_choice == 4:
        x = 128.4
        y = -129 + (127+129)*road_position
        z_angle = (direction_choice-2)*1.5708 + (direction_choice-1)*1.5708
    elif road_choice == 5:
        x = 80.3
        y = -129 + (129-1.1)*road_position
        z_angle = (direction_choice-2)*1.5708 + (direction_choice-1)*1.5708  
    elif road_choice == 6:
        x = 0.5
        y = -129 + (127+129)*road_position 
        z_angle = (direction_choice-2)*1.5708 + (direction_choice-1)*1.5708 
    else:
        x = -127.6
        y = -129 + (127+129)*road_position 
        z_angle = (direction_choice-2)*1.5708 + (direction_choice-1)*1.5708 

    position = airsim.Vector3r(x, y, -0.6031363606452942)
    heading = airsim.utils.to_quaternion(0, 0, z_angle)
    pose = airsim.Pose(position, heading)
    client.simSetVehiclePose(pose, True)   

    weather_choice = random.randint(1,5)
    # print(weather_choice)
    if weather_choice == 1:
        print('No weather')
        client.simSetWeatherParameter(airsim.WeatherParameter.Rain, 0)
        client.simSetWeatherParameter(airsim.WeatherParameter.Snow, 0)
        client.simSetWeatherParameter(airsim.WeatherParameter.MapleLeaf, 0)
    elif weather_choice == 2:
        print('Heavy Rain')
        client.simSetWeatherParameter(airsim.WeatherParameter.Rain, 0.99)  
    elif weather_choice == 3:
        print('Heavy Leaf')
        client.simSetWeatherParameter(airsim.WeatherParameter.MapleLeaf, 0.99) 
    elif weather_choice == 4:
        print('Heavy Rain and Snow')
        client.simSetWeatherParameter(airsim.WeatherParameter.Rain, 0.99)
        client.simSetWeatherParameter(airsim.WeatherParameter.Snow, 0.99)
    else:
        print('Heavy Snow and MapleLeaf')
        client.simSetWeatherParameter(airsim.WeatherParameter.Snow, 0.99)
        client.simSetWeatherParameter(airsim.WeatherParameter.MapleLeaf, 0.99) 


updates = 0

TRAIN_MEAN = (0.1840, 0.1658, 0.1612)
TRAIN_STD = (0.2539, 0.2384, 0.2597)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(TRAIN_MEAN, TRAIN_STD)
])

for i_episode in range(1, num_episodes):
    reset_initial_position_weather(client)
    episode_reward = 0
    steps = 0
    stop_count = 0
    while steps < num_steps:
        steps += 1
        img, velocity, done = get_image_velocity_done(client)

        img_transform = transform(img).to(device).float().unsqueeze(0)

        with torch.no_grad(): 
            out = net(img_transform, velocity.to(device))
            action = out.squeeze(0).cpu().numpy()

            if action[0] < 0:
                action[0] = 0
            elif action[0] > 1:
                action[0] = 1          
            if action[1] < 0:
                action[1] = 0
            elif action[1] > 1:
                action[1] = 1
            if action[2] < -1:
                action[2] = -1
            elif action[2] > 1:
                action[2] = 1  
            if action[0] < action[1]:
                action[0] = 0
            else:
                action[1] = 0               


            car_controls.throttle = float(action[0])
            car_controls.brake = float(action[1])
            car_controls.steering = float(action[2])
            print("Throttle:%f Brake:%f Steering:%f" %(action[0],action[1],action[2]))
            client.setCarControls(car_controls)
            time.sleep(0.1)

        if done:
            client.reset()
            client.enableApiControl(True)
            break

    print('Episode steps: %d'%steps)
