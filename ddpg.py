import sys

import torch
import torch.nn as nn
from torch.optim import Adam
# from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import os 

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device2 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


class BottleNeck(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.downsample = downsample

        if downsample:
            self.stride = 2
        else:
            self.stride = 1

        self.relu = nn.ReLU()

        self.conv_main = nn.Conv2d(in_channels = self.in_channels,
                                  out_channels = self.out_channels,
                                  kernel_size = 1,
                                  stride = self.stride,
                                  padding = 0,
                                  dilation = 1,
                                  bias = False)
        self.batch_norm_main  = nn.BatchNorm2d(self.out_channels)

        self.conv1 = nn.Conv2d(in_channels = self.in_channels,
                               out_channels = self.out_channels,
                               kernel_size = 3,
                               stride = self.stride,
                               padding = 1,
                               dilation = 1,
                               bias = False)
        self.batch_norm1  = nn.BatchNorm2d(self.out_channels)
        self.conv2 = nn.Conv2d(in_channels = self.out_channels,
                              out_channels = self.out_channels,
                              kernel_size = 3,
                              stride = 1,
                              padding = 1,
                              dilation = 1,
                              bias = False)
        self.batch_norm2  = nn.BatchNorm2d(self.out_channels) 

    def forward(self, x):
        #main branch
        if self.downsample:
            shortcut = self.conv_main(x)
            shortcut = self.batch_norm_main(shortcut)
        else:
            shortcut = x

        #side branch
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.batch_norm2(x)

        out = x + shortcut
        out = self.relu(out)

        return out



class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.hard_tanh_1 = torch.nn.Hardtanh(-1,1)
        self.hard_tanh_2 = torch.nn.Hardtanh(0,1)

        self.b21 = BottleNeck(64,64)
        self.b22 = BottleNeck(64,64)
        self.b23 = BottleNeck(64,64)

        self.b31 = BottleNeck(64,128,downsample=True)
        self.b32 = BottleNeck(128,128)
        self.b33 = BottleNeck(128,128)      
        self.b34 = BottleNeck(128,128)

        self.b41 = BottleNeck(128,256,downsample=True)
        self.b42 = BottleNeck(256,256)
        self.b43 = BottleNeck(256,256)      
        self.b44 = BottleNeck(256,256)  
        self.b45 = BottleNeck(256,256)
        self.b46 = BottleNeck(256,256)

        self.b51 = BottleNeck(256,512,downsample=True)
        self.b52 = BottleNeck(512,512)
        self.b53 = BottleNeck(512,512)              

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # self.fc1 = nn.Linear(512, 20)
        # self.fc2 = nn.Linear(22, 10)
        # self.fc3 = nn.Linear(10, 3)
        self.fc1 = nn.Linear(513, 250)
        self.fc1_bn = nn.BatchNorm1d(250)
        self.fc2 = nn.Linear(250, 100)
        self.fc2_bn = nn.BatchNorm1d(100)
        self.fc3_actor = nn.Linear(100, 3)

    def forward(self, x, x2):
        # print(x.shape)
        # print(x2.shape)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.b21(x)
        x = self.b22(x)
        x = self.b23(x)

        x = self.b31(x)
        x = self.b32(x)
        x = self.b33(x)
        x = self.b34(x)

        x = self.b41(x)
        x = self.b42(x)
        x = self.b43(x)
        x = self.b44(x)
        x = self.b45(x)
        x = self.b46(x)

        x = self.b51(x)
        x = self.b52(x)
        x = self.b53(x)

        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = torch.cat((x,x2), dim = 1)
        x = self.fc1(x)
        x = self.fc1_bn(x)
        x = self.relu(x) 
        x = self.fc2(x)
        x = self.fc2_bn(x)
        x = self.relu(x)
        x = self.fc3_actor(x)

        acc = self.hard_tanh_2(x[::,0]).unsqueeze(1)
        brake = self.hard_tanh_2(x[::,1]).unsqueeze(1)
        steering = self.hard_tanh_1(x[::,2]).unsqueeze(1)

        x = torch.cat((acc,brake,steering), dim = 1)    

        return x

class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.hard_tanh_1 = torch.nn.Hardtanh(-1,1)
        self.hard_tanh_2 = torch.nn.Hardtanh(0,1)

        self.b21 = BottleNeck(64,64)
        self.b22 = BottleNeck(64,64)
        self.b23 = BottleNeck(64,64)

        self.b31 = BottleNeck(64,128,downsample=True)
        self.b32 = BottleNeck(128,128)
        self.b33 = BottleNeck(128,128)      
        self.b34 = BottleNeck(128,128)

        self.b41 = BottleNeck(128,256,downsample=True)
        self.b42 = BottleNeck(256,256)
        self.b43 = BottleNeck(256,256)      
        self.b44 = BottleNeck(256,256)  
        self.b45 = BottleNeck(256,256)
        self.b46 = BottleNeck(256,256)

        self.b51 = BottleNeck(256,512,downsample=True)
        self.b52 = BottleNeck(512,512)
        self.b53 = BottleNeck(512,512)              

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # self.fc1 = nn.Linear(512, 20)
        # self.fc2 = nn.Linear(22, 10)
        # self.fc3 = nn.Linear(10, 3)
        self.fc1 = nn.Linear(513, 250)
        self.fc1_bn = nn.BatchNorm1d(250)
        self.fc2 = nn.Linear(250, 100)
        self.fc2_bn = nn.BatchNorm1d(100)
        self.fc3_critic = nn.Linear(103, 1)

    def forward(self, x, x2, a):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.b21(x)
        x = self.b22(x)
        x = self.b23(x)

        x = self.b31(x)
        x = self.b32(x)
        x = self.b33(x)
        x = self.b34(x)

        x = self.b41(x)
        x = self.b42(x)
        x = self.b43(x)
        x = self.b44(x)
        x = self.b45(x)
        x = self.b46(x)

        x = self.b51(x)
        x = self.b52(x)
        x = self.b53(x)

        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = torch.cat((x,x2), dim = 1)
        x = self.fc1(x)
        x = self.fc1_bn(x)
        x = self.relu(x) 
        x = self.fc2(x)
        x = self.fc2_bn(x)
        x = self.relu(x)
        x = torch.cat((x,a), dim = 1)
        x = self.fc3_critic(x)

        return x

lr = 0.01
wd = 1e-4

class DDPG(object):
    def __init__(self, gamma, tau):

        def freeze(layer):
            for param in layer.parameters():
                param.requires_grad = False     
        def unfreeze(layer):
            for param in layer.parameters():
                param.requires_grad = True        

        self.actor = Actor().to(device)
        self.actor_target = Actor().to(device)

        freeze(self.actor)
        unfreeze(self.actor.fc1)
        unfreeze(self.actor.fc1_bn)
        unfreeze(self.actor.fc2)
        unfreeze(self.actor.fc2_bn)
        unfreeze(self.actor.fc3_actor)


        # self.actor_optim = Adam(self.actor.parameters(), lr=1e-4)
        # self.actor_optim = Adam(filter(lambda p: p.requires_grad, self.actor.parameters()), lr=1e-4)
        self.actor_optim = torch.optim.SGD(filter(lambda p: p.requires_grad, self.actor.parameters()), lr=0.1*lr, momentum=0.9, weight_decay=wd)
        self.actor.train()
        self.actor_target.train()

        self.critic = Critic().to(device2)
        self.critic_target = Critic().to(device2)

        freeze(self.critic)
        unfreeze(self.critic.fc1)
        unfreeze(self.critic.fc1_bn)
        unfreeze(self.critic.fc2)
        unfreeze(self.critic.fc2_bn)
        unfreeze(self.critic.fc3_critic)

        # self.critic_optim = Adam(self.critic.parameters(), lr=1e-3)
        # self.critic_optim = Adam(filter(lambda p: p.requires_grad, self.critic.parameters()), lr=1e-3)
        self.critic_optim = torch.optim.SGD(filter(lambda p: p.requires_grad, self.critic.parameters()), lr=lr, momentum=0.9, weight_decay=wd)
        self.critic.train()
        self.critic_target.train()


        self.gamma = gamma
        self.tau = tau

        hard_update(self.actor_target, self.actor)  # Make sure target is with the same weight
        hard_update(self.critic_target, self.critic)


    def select_action(self, img, velocity, action_noise=None):
        self.actor.eval()
        with torch.no_grad(): 
            out = self.actor(img, velocity)
            control = out.squeeze(0).cpu().numpy()
            # print('control output:', control)
            control_acc = control[0]
            control_brake = control[1]
            control_steer = control[2]

            # if control_acc < 0.1:
            #     control_acc = 0
            # elif control_acc > 1:
            #     control_acc = 1
            # if control_brake < 0.1:
            #     control_brake = 0
            # elif control_brake > 1:
            #     control_brake = 1
            # if control_steer < -1:
            #     control_steer = -1
            # elif control_steer > 1:
            #     control_steer = 1     
            # if control_acc > control_brake:
            #     control_brake = 0
            # else:
            #     control_acc = 0

            # self.actor.train()
            # mu = mu.data

            control_actual = np.array([control_acc, control_brake, control_steer])

            # if action_noise is not None:
            #     control_actual += torch.Tensor(action_noise.noise())
        self.actor.train()

        return control_actual


    # def update_parameters(self, states_batch,actions_batch,rewards_batch,dones_batch,states_next_batch):
    def update_parameters(self, imgs, velocitys, controls, rewards, dones, imgs_next, velocitys_next):

        # imgs_batch = [d['image'] for d in states_batch]
        # vels_batch = [d['velocity'] for d in states_batch]
        # imgs_next_batch = [d['image'] for d in states_next_batch]
        # vels_next_batch = [d['velocity'] for d in states_next_batch]

        # imgs_batch = torch.cat(imgs_batch).to(device).float()
        # vels_batch = torch.cat(vels_batch).to(device).float()
        # imgs_next_batch = torch.cat(imgs_next_batch).to(device).float()
        # vels_next_batch = torch.cat(vels_next_batch).to(device).float()
        # actions_batch = torch.cat(actions_batch).to(device).float()
        # rewards_batch = torch.cat(rewards_batch).to(device2).float()
        # dones_batch = torch.cat(dones_batch).to(device2).float()

        imgs_batch = imgs.to(device).float()
        vels_batch = velocitys.to(device).float()
        imgs_next_batch = imgs_next.to(device).float()
        vels_next_batch = velocitys_next.to(device).float()
        actions_batch = controls.to(device2).float()
        rewards_batch = rewards.to(device2).float()
        dones_batch = dones.to(device2).float()


        actions_next_batch = self.actor_target(imgs_next_batch, vels_next_batch)
        state_action_values_next_batch = self.critic_target(imgs_next_batch.to(device2), vels_next_batch.to(device2), actions_next_batch.to(device2))

        # rewards_batch = rewards_batch.unsqueeze(1)
        # dones_batch = dones_batch.unsqueeze(1)
        expected_state_action_values_batch = rewards_batch + (self.gamma * (1-dones_batch) * state_action_values_next_batch)

        self.critic_optim.zero_grad()

        state_action_values_batch = self.critic(imgs_batch.to(device2), vels_batch.to(device2), actions_batch.to(device2))

        value_loss = F.mse_loss(state_action_values_batch, expected_state_action_values_batch)
        # value_loss = F.smooth_l1_loss(state_action_values_batch, expected_state_action_values_batch)
        # smooth_l1_loss
        value_loss.backward()
        self.critic_optim.step()


        self.actor_optim.zero_grad()

        policy_loss = -self.critic(imgs_batch.to(device2), vels_batch.to(device2), self.actor(imgs_batch, vels_batch).to(device2))

        policy_loss = policy_loss.mean()
        policy_loss.backward()
        self.actor_optim.step()


        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)

        return value_loss.item(), policy_loss.item()

    def save_model(self, actor_path=None, critic_path=None):
        if not os.path.exists('models/'):
            os.makedirs('models/')
        actor_path = "models/ddpg_actor_pretrained" + actor_path + ".pth" 
        critic_path = "models/ddpg_critic_pretrained" + critic_path + ".pth" 
        print('Saving models to {} and {}'.format(actor_path, critic_path))
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)

    def save_target_model(self, actor_path=None, critic_path=None):
        if not os.path.exists('models/'):
            os.makedirs('models/')
        actor_path = "models/ddpg_actor_target_pretrained" + actor_path + ".pth" 
        critic_path = "models/ddpg_critic_target_pretrained" + critic_path + ".pth" 
        print('Saving models to {} and {}'.format(actor_path, critic_path))
        torch.save(self.actor_target.state_dict(), actor_path)
        torch.save(self.critic_target.state_dict(), critic_path)

    def load_model(self, actor_path, critic_path):
        print('Loading models from {} and {}'.format(actor_path, critic_path))
        if actor_path is not None:
            # self.actor.load_state_dict(torch.load(actor_path))
            pretrained_dict = torch.load(actor_path)
            # print(pretrained_dict.items())
            model_dict = self.actor.state_dict()

            # 1. filter out unnecessary keys
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict) 
            # 3. load the new state dict
            self.actor.load_state_dict(model_dict)
        # if critic_path is not None: 
        #     self.critic.load_state_dict(torch.load(critic_path))

            pretrained_dict = torch.load(actor_path)
            # print(pretrained_dict.items())
            model_dict = self.critic.state_dict()

            # 1. filter out unnecessary keys
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict) 
            # 3. load the new state dict
            self.critic.load_state_dict(model_dict)


class OUNoise:

    def __init__(self, action_dimension, scale=0.1, mu=0, theta=0.15, sigma=0.2):
        self.action_dimension = action_dimension
        self.scale = scale
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()
        self.steer = 0.2

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        x = self.state
        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        # dx1 = self.theta * (self.mu - x1) + self.sigma * np.random.normal(0,1,1)[0]
        dx1 = self.theta * (self.mu - x1) + self.sigma * np.random.normal(0.5,0.5,1)[0]
        dx2 = self.theta * (self.mu - x2) + self.sigma * np.random.normal(0.2,0.2,1)[0]
        # dx3 = self.theta * (self.mu - x3) + self.sigma * np.random.normal(0,0.05,1)[0]
        x1 += dx1
        x2 += dx2
        # x3 += dx3
        x3 = np.random.normal(0,self.steer,1)[0]
        self.state = np.array([x1, x2, x3])
        return self.state * self.scale
