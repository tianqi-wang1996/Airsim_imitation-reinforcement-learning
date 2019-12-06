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
from xbox360controller import Xbox360Controller
import setup_path 
import airsim

# basic setting
MEMORY_SIZE = 12000    # number of previous transitions to remember
MINI_BATCH = 32         # size of mini batch
DEPTH_IMAGE_WIDTH = 256
DEPTH_IMAGE_HEIGHT = 144

client = airsim.VehicleClient()
client.confirmConnection()

client.simEnableWeather(True)

times_a = 0
times_b = 0
times_x = 0
times_y = 0
Recording = 0


def A_button_pressed(button):
    global client
    global times_a
    times_a += 1
    times_a = times_a%4
    client.simSetWeatherParameter(airsim.WeatherParameter.Rain, times_a*0.25)
    
def B_button_pressed(button):
    global client
    global times_b
    times_b += 1
    times_b = times_b%4
    client.simSetWeatherParameter(airsim.WeatherParameter.Snow, times_b*0.25)    

def X_button_pressed(button):
    global client
    global times_x
    times_x += 1
    times_x = times_x%4
    client.simSetWeatherParameter(airsim.WeatherParameter.MapleLeaf, times_x*0.25)  

def Y_button_pressed(button):
    global client
    global times_y
    times_y += 1
    times_y = times_y%2
    if times_y == 1:   
        client.simSetTimeOfDay(True, "2019-6-17 18:00:00", move_sun = True)
    else:
        client.simSetTimeOfDay(False)

def start_button_pressed(button):
    global client
    client.simSetWeatherParameter(airsim.WeatherParameter.Rain, 0.99)
    client.simSetWeatherParameter(airsim.WeatherParameter.Snow, 0.99)

def select_button_pressed(button):
    global client
    client.simSetWeatherParameter(airsim.WeatherParameter.Rain, 0)
    client.simSetWeatherParameter(airsim.WeatherParameter.Snow, 0) 
    client.simSetWeatherParameter(airsim.WeatherParameter.MapleLeaf, 0)

def mode_button_pressed(button):
    global Recording
    Recording += 1
    Recording = Recording%2

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




def compute_reward(car_state):
    pd = car_state.kinematics_estimated.position
    quaternion = car_state.kinematics_estimated.orientation
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
        dist1 = 10000

    # if dist1 < 0.2:
    #     dist1 = 0
    dist_min = min(dist, dist1)

    return road, dist_min

        
def get_image(client):
    responses = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)])
    response = responses[0]
    img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)
    img = img1d.reshape(response.height, response.width, 4)
    img = img[:, :, 0:3]
    return img

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

def trainNetwork():
    global Recording
    client = airsim.CarClient()
    client.confirmConnection()
    print('Connect succcefully！')
    # client.enableApiControl(True)
    # client.reset()
    print('Environment initialized!')

    controller = Xbox360Controller()

    # Initialize the buffer
    replay_experiences = deque()
    replay_experiences = store_transition(replay_experiences,'read')

    # get the first state
    kinematics  = client.getCarState().kinematics_estimated
    img = get_image(client)
    velocity_x = kinematics.linear_velocity.x_val+1.6
    velocity_y = kinematics.linear_velocity.y_val-15
    velocity = np.sqrt(velocity_x**2+velocity_y**2)
    acc_x = kinematics.linear_acceleration.x_val
    acc_y = kinematics.linear_acceleration.y_val
    acc = np.sqrt(acc_x**2+acc_y**2)    
    kinematics_state  = np.array([velocity,acc]) 
    
    state_previous = {}
    state_previous['image']=img
    state_previous['kinematics_state'] = kinematics_state

    reward_previous = 10


    pbar = tqdm(total = MEMORY_SIZE-len(replay_experiences))
    
    controller.button_mode.when_pressed = mode_button_pressed
    client.simSetTimeOfDay(False)
    start_size = int(len(replay_experiences))
    difference = int(MEMORY_SIZE-len(replay_experiences))
    # for i in tqdm(range(MEMORY_SIZE+10)):
    while len(replay_experiences) < MEMORY_SIZE:       
        if len(replay_experiences)+1 == start_size + difference//2:   
            client.simSetTimeOfDay(True, "2019-6-17 18:00:00", move_sun = True)
        if (len(replay_experiences)+1-start_size)%(difference//2) == 0:     
            client.simSetWeatherParameter(airsim.WeatherParameter.Rain, 0)
            client.simSetWeatherParameter(airsim.WeatherParameter.Snow, 0)
            client.simSetWeatherParameter(airsim.WeatherParameter.MapleLeaf, 0)
        if (len(replay_experiences)+1-start_size)%(difference//2) == (difference//2) * (1/4):     
            client.simSetWeatherParameter(airsim.WeatherParameter.Rain, 0.99)  
        if (len(replay_experiences)+1-start_size)%(difference//2) == (difference//2) * (1/4):     
            client.simSetWeatherParameter(airsim.WeatherParameter.Snow, 0.99) 
        if (len(replay_experiences)+1-start_size)%(difference//2) == (difference//2) * (1/4):     
            client.simSetWeatherParameter(airsim.WeatherParameter.Rain, 0)  
            client.simSetWeatherParameter(airsim.WeatherParameter.MapleLeaf, 0.99)  
        if (len(replay_experiences)+1)%200 == 0:  
            signal_back = store_transition(replay_experiences, 'store')
            if signal_back:
                print('store pkl file succcefully')                 
        car_state = client.getCarState()
        _, reward_step = compute_reward(car_state)
        kinematics  = car_state.kinematics_estimated
        img = get_image(client)
        velocity_x = kinematics.linear_velocity.x_val+1.6
        velocity_y = kinematics.linear_velocity.y_val-15
        velocity = np.sqrt(velocity_x**2+velocity_y**2)
        acc_x = kinematics.linear_acceleration.x_val
        acc_y = kinematics.linear_acceleration.y_val
        acc = np.sqrt(acc_x**2+acc_y**2)    
        kinematics_state  = np.array([velocity,acc]) 

        state = {}
        state['image']=img
        state['kinematics_state'] = kinematics_state

        steer = controller.axis_l.x
        brake = controller.trigger_l.value
        acc = controller.trigger_r.value
        if steer >= -0.25 and steer <= 0.25:
            steer = 0
        elif steer > 0.25:
            steer = 4/3.0 * steer - 1/3.0
        elif steer < -0.25:
            steer = 4/3.0 * steer + 1/3.0

        print('acc: %f brake: %f steer: %f' %(acc,brake,steer))
        print('reward:',reward_previous)
        print('Recording: %d'%Recording)
        print('memory: %d'%len(replay_experiences))

        print('weather:',times_a,times_b,times_x,times_y)
  

        action = np.array([acc,brake,steer])

        if Recording:
            replay_experiences.append((state_previous, action, reward_previous))
            pbar.update(1)

        state_previous = copy.deepcopy(state)
        reward_previous = reward_step
        time.sleep(0.1)
    client.simSetTimeOfDay(False)
    print(len(replay_experiences))

    signal_back = store_transition(replay_experiences, 'store')
    if signal_back:
        print('store pkl file succcefully')
    print(len(replay_experiences))




def main():
    trainNetwork()

if __name__ == "__main__":
    main()


