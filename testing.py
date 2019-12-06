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
from baseline import ResNet34, ResNet50
import copy
from tqdm import tqdm

from ddpg import DDPG, OUNoise
# from replay_memory import ReplayMemory, Transition

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net = ResNet34()
print("Model Instantiated")

# uncomment this part to restore weights

# checkpoint = torch.load('NEW_ResNet34_augmented-80-0.0034.pth')
# checkpoint = torch.load('RL_ResNet34_unfreeze2-90-0.0034.pth')

checkpoint = torch.load('checkpoint_reinforcement.pth')
# checkpoint = torch.load('RL_ResNet34_freeze-100-0.0035.pth')
# RL_ResNet34_unfreeze2-100-0.0035.pth

# checkpoint = torch.load('./models/ddpg_actor_pretrained_18_0.020736435340283528.pth')
# ./models/ddpg_actor_pretrained_18_0.020736435340283528.pth
net.load_state_dict(checkpoint)
print('Restoring parameters')

net = net.to(device)
net.eval()

gamma = 0.99
tau = 0.001
noise_scale = 1
final_noise_scale = 0.001
exploration_end = 200
MEMORY_SIZE = 30000 - 1
# batch_size = 20
batch_size = 10
num_steps = 1000
num_episodes = 1000
# updates_per_episode = 100
validation_times = 5




client = airsim.CarClient()
client.confirmConnection()
print('Connect succcefully!')
# client.enableApiControl(True)

client.simEnableWeather(True)

car_controls = airsim.CarControls()
car_controls.throttle = 0
car_controls.steering = 0
# client.reset()
print('Environment initialized!')

def compute_center_distance(pt, car_obs_num):
    pt_obs = client.simGetObjectPose("Car_"+car_obs_num).position
    dist = np.sqrt((pt.x_val-pt_obs.x_val)**2 + (pt.y_val-pt_obs.y_val)**2)
    return dist

def compute_distance_element(pt, A, B):
    v1 = B - A
    v2 = pt - A
    u1 = v1/(np.sqrt(v1[0]**2+v1[1]**2))
    projection_dis = v2[0]*u1[0] + v2[1]*u1[1]
    AB_dis = np.sqrt(v1[0]**2+v1[1]**2)
    if projection_dis < 0:
        dist = np.sqrt(v2[0]**2+v2[1]**2)
    elif projection_dis > AB_dis:
        v3 = pt - B
        dist = np.sqrt(v3[0]**2+v3[1]**2)
    else:
        v2_substracted = v2 - projection_dis*u1
        dist = np.sqrt(v2_substracted[0]**2+v2_substracted[1]**2)
    return dist


def compute_distance_obstacles(pt, angle, car_obs_num):
    car_width_half = 1.05
    # car_width_half = 1.025
    car_length_half = 2.6
    # car_length_half = 2.5
    # obs_width_half = 1.025
    obs_width_half = 1.1
    obs_length_half = 2.65
    pt_car = np.array([[pt.x_val],[pt.y_val]])

    pose_obs = client.simGetObjectPose("Car_"+car_obs_num)
    pt_obs = np.array([[pose_obs.position.x_val],[pose_obs.position.y_val]])
    _,_,angle_obs = airsim.utils.to_eularian_angles(pose_obs.orientation)
    angle_obs = -(angle_obs+1.57079632679)

    car_vertix = np.array([[car_length_half,car_length_half,-car_length_half,-car_length_half], [-car_width_half,car_width_half,car_width_half,-car_width_half]])
    obs_vertix = np.array([[obs_length_half,obs_length_half,-obs_length_half,-obs_length_half], [-obs_width_half,obs_width_half,obs_width_half,-obs_width_half]])

    car_rotation_mat = np.array([[np.cos(angle),np.sin(angle)], [-np.sin(angle),np.cos(angle)]])
    obs_rotation_mat =np.array([[np.cos(angle_obs),np.sin(angle_obs)], [-np.sin(angle_obs),np.cos(angle_obs)]])

    car_vertix = np.matmul(car_rotation_mat, car_vertix) + pt_car
    obs_vertix = np.matmul(obs_rotation_mat, obs_vertix) + pt_obs

    dist_obs = 1000

    for i in range(4):
        for j in range(4):
            if j == 3:
                dist_obs = min(dist_obs, compute_distance_element(car_vertix[:,i], obs_vertix[:,3], obs_vertix[:,0]))
                dist_obs = min(dist_obs, compute_distance_element(obs_vertix[:,i], car_vertix[:,3], car_vertix[:,0]))
            else:
                dist_obs = min(dist_obs, compute_distance_element(car_vertix[:,i], obs_vertix[:,j], obs_vertix[:,j+1]))
                dist_obs = min(dist_obs, compute_distance_element(obs_vertix[:,i], car_vertix[:,j], car_vertix[:,j+1]))

    return dist_obs




def compute_reward(kinematics):
    pd = kinematics.position
    quaternion = kinematics.orientation

    _,_,angle = airsim.utils.to_eularian_angles(quaternion)
    angle = -angle
    # car_pt = np.array([pd.x_val, pd.y_val])
    x = pd.x_val
    y = pd.y_val
    car_width_half = 1.025
    road_width_half = 4.25
    x1 = np.array([128.4, 80.3, 0.5, -127.6])
    y1 = np.array([-129, -1.1, 127]) 
    dist1 = 21
    dist = 22
    if x<(x1[0]+road_width_half-car_width_half) and x>(x1[3]-road_width_half+car_width_half) and y<(y1[0]+road_width_half-car_width_half) and y>(y1[0]-road_width_half+car_width_half):
        road = 1
        dist = min((y1[0]+road_width_half-car_width_half)-y, y-(y1[0]-road_width_half+car_width_half))
        obstalcles = ["60","112","62","63","117","65"]
        for i in range(len(obstalcles)):
            if(compute_center_distance(pd, obstalcles[i]) < 15):
                dist1 = min(dist1, compute_distance_obstacles(pd, angle, obstalcles[i]))

    elif x<(x1[0]+road_width_half-car_width_half) and x>(x1[3]-road_width_half+car_width_half) and y<(y1[1]+road_width_half-car_width_half) and y>(y1[1]-road_width_half+car_width_half):
        road = 2
        dist = min((y1[1]+road_width_half-car_width_half)-y, y-(y1[1]-road_width_half+car_width_half))
        obstalcles = ["58","59","56","107"]
        for i in range(len(obstalcles)):
            if(compute_center_distance(pd, obstalcles[i]) < 15):
                dist1 = min(dist1, compute_distance_obstacles(pd, angle, obstalcles[i]))

    elif x<(x1[0]+road_width_half-car_width_half) and x>(x1[3]-road_width_half+car_width_half) and y<(y1[2]+road_width_half-car_width_half) and y>(y1[2]-road_width_half+car_width_half):
        road = 3
        dist = min((y1[2]+road_width_half-car_width_half)-y, y-(y1[2]-road_width_half+car_width_half))
        obstalcles = ["97","50","100","51","103","53","55"]
        for i in range(len(obstalcles)):
            if(compute_center_distance(pd, obstalcles[i]) < 15):
                dist1 = min(dist1, compute_distance_obstacles(pd, angle, obstalcles[i]))

    elif y<(y1[2]+road_width_half-car_width_half) and y>(y1[0]-road_width_half+car_width_half) and x<(x1[0]+road_width_half-car_width_half) and x>(x1[0]-road_width_half+car_width_half):
        road = 4
        dist = min((x1[0]+road_width_half-car_width_half)-x, x-(x1[0]-road_width_half+car_width_half))
        obstalcles = ["87","41","44","43","46","92","48","47"]
        for i in range(len(obstalcles)):
            if(compute_center_distance(pd, obstalcles[i]) < 15):
                dist1 = min(dist1, compute_distance_obstacles(pd, angle, obstalcles[i]))

    elif y<(y1[1]+road_width_half-car_width_half) and y>(y1[0]-road_width_half+car_width_half) and x<(x1[1]+road_width_half-car_width_half) and x>(x1[1]-road_width_half+car_width_half):
        road = 5
        dist = min((x1[1]+road_width_half-car_width_half)-x, x-(x1[1]-road_width_half+car_width_half))
        obstalcles = ["40","39","80","38","81"]
        for i in range(len(obstalcles)):
            if(compute_center_distance(pd, obstalcles[i]) < 15):
                dist1 = min(dist1, compute_distance_obstacles(pd, angle, obstalcles[i]))

    elif y<(y1[2]+road_width_half-car_width_half) and y>(y1[0]-road_width_half+car_width_half) and x<(x1[2]+road_width_half-car_width_half) and x>(x1[2]-road_width_half+car_width_half):
        road = 6
        dist = min((x1[2]+road_width_half-car_width_half)-x, x-(x1[2]-road_width_half+car_width_half))
        obstalcles = ["30","77","35","01_32","37","7"]
        for i in range(len(obstalcles)):
            if(compute_center_distance(pd, obstalcles[i]) < 15):
                dist1 = min(dist1, compute_distance_obstacles(pd, angle, obstalcles[i]))

    elif y<(y1[2]+road_width_half-car_width_half) and y>(y1[0]-road_width_half+car_width_half) and x<(x1[3]+road_width_half-car_width_half) and x>(x1[3]-road_width_half+car_width_half):
        road = 7
        dist = min((x1[3]+road_width_half-car_width_half)-x, x-(x1[3]-road_width_half+car_width_half))
        obstalcles = ["66","69","74","32","27","64"]
        for i in range(len(obstalcles)):
            if(compute_center_distance(pd, obstalcles[i]) < 15):
                dist1 = min(dist1, compute_distance_obstacles(pd, angle, obstalcles[i]))
    else:
        road = 0
        dist1 = 0

    # if dist1 < 0.2:
    #     dist1 = 0
    dist_min = min(dist, dist1)

    # return road, dist_min
    return dist_min

def get_image_velocity_reward_done(client):
    responses = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)])
    response = responses[0]
    kinematics  = client.getCarState().kinematics_estimated
    distance = compute_reward(kinematics)
    if distance > 3.5:
        reward_distance = 1
    reward_distance = (distance/3.5)%1
    collision_info = client.simGetCollisionInfo()
    done = int(collision_info.has_collided == True)


    img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)
    img = img1d.reshape(response.height, response.width, 4)
    img = img[:, :, 0:3]
    
    velocity_x = kinematics.linear_velocity.x_val+1.6
    velocity_y = kinematics.linear_velocity.y_val-15
    velocity = np.sqrt(velocity_x**2+velocity_y**2)
    if velocity > 20:
        reward_velocity = 1
    else:
        reward_velocity = (velocity/20)%1
    reward = 0.5*reward_distance + 0.5*reward_velocity
    if reward_distance < 0.1 or reward_velocity < 0.1:
        reward = 0

    
    velocity  = np.array([velocity])[0] 

    # TRAIN_MEAN = (0.1840, 0.1658, 0.1612)
    # TRAIN_STD = (0.2539, 0.2384, 0.2597)

    # transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize(TRAIN_MEAN, TRAIN_STD)
    # ])
    # img = transform(img).float().unsqueeze(0)
    velocity = torch.tensor(velocity).float().unsqueeze(0).unsqueeze(0)
    if done == 1:
        reward = 0

    return img, velocity, reward, done


def reset_initial_position_weather(client):
    client.reset()
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
        img, velocity, reward, done = get_image_velocity_reward_done(client)
        # print('reward: %f'%reward)
        episode_reward += reward
        if reward < 0.01:
            stop_count += 1
        if reward > 0.2:
            stop_count = 0
        if stop_count > 30:
            print('too many zero rewards, stop!!')
            client.reset()
            break

        img_transform = transform(img).to(device).float().unsqueeze(0)
        # velocity = velocity
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
            print(action[0],action[1],action[2])
            # client.setCarControls(car_controls)
            time.sleep(0.1)

        if done:
            client.reset()
            break

    print('Episode steps: %d'%steps)
    print("[Train] Episode: {}, reward: {}".format(i_episode, episode_reward))
