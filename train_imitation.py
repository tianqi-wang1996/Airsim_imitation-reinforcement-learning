import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch import nn
import torch.nn.functional as F
from torch.optim import SGD
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
from tqdm import tqdm
from torch.utils.data.dataset import Subset
import numpy as np
from torch.autograd import Variable
import pickle
from torchsummary import summary

from model.baseline import ResNet34, ResNet50, Critic
from se_resnet import SE_ResNet34, SE_ResNet50
from bam_resnet import BAM_ResNet34, BAM_ResNet50
from cbam_resnet import CBAM_ResNet34, CBAM_ResNet50


device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

store_path = 'training_augmented.pkl'
store_file = open(store_path,'rb')
training_data = pickle.load(store_file)
store_file.close()
memory_len = len(training_data)
print('Successfully load the training.pkl, %05d memory'%memory_len)

store_path = 'validation_augmented.pkl'
store_file = open(store_path,'rb')
val_data = pickle.load(store_file)
store_file.close()
memory_len = len(val_data)
print('Successfully load the validation.pkl, %05d memory'%memory_len)


class TrainingDataset(Dataset):
    global training_data
    def __init__(self, transform=None):
        self.transform = transform

    def __len__(self):
        return len(training_data)

    def __getitem__(self, index):
        state = training_data[index][0]
        image = state['image']
        velocity = state['kinematics_state'][0]
        if self.transform is not None:
            image = self.transform(image)
            velocity = torch.tensor(velocity).unsqueeze(0)
        control = training_data[index][1]
        control = torch.tensor(control)

        return image, velocity, control

class ValDataset(Dataset):
    global val_data
    def __init__(self, transform=None):
        self.transform = transform

    def __len__(self):
        return len(val_data)

    def __getitem__(self, index):
        state = val_data[index][0]
        image = state['image']
        velocity = state['kinematics_state'][0]
        if self.transform is not None:
            image = self.transform(image)
            velocity = torch.tensor(velocity).unsqueeze(0)
        control = val_data[index][1]
        control = torch.tensor(control)

        return image, velocity, control


## Arguments
train_batch_size = 50
val_batch_size = 50
epochs = 120
lr = 0.01 * 10
wd = 1e-4
print_every = 1
eval_every = 1
save_every = 10

net = ResNet34()
print("Model Instantiated")

# uncomment this part to restore weights
checkpoint = torch.load('NEW_ResNet34_augmented-80-0.0034.pth')
net.load_state_dict(checkpoint)
print('Restoring parameters')

net = net.to(device)


def freeze(layer):
    for param in layer.parameters():
        param.requires_grad = False     
def unfreeze(layer):
    for param in layer.parameters():
        param.requires_grad = True        

freeze(net)
unfreeze(net.fc1)
unfreeze(net.fc1_bn)
unfreeze(net.fc2)
unfreeze(net.fc2_bn)
unfreeze(net.fc3)

criterion = nn.SmoothL1Loss()
optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=0.9, weight_decay=wd)

print ('[INFO]Defined the loss function and the optimizer')

# Data Preparation
print('Data Preparation:')


TRAIN_MEAN = (0.1840, 0.1659, 0.1613)
TRAIN_STD = (0.2540, 0.2386, 0.2599)

transform_train = transforms.Compose([
    # transforms.RandomCrop(32, padding=4),
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(TRAIN_MEAN, TRAIN_STD)
])

transform_val = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(TRAIN_MEAN, TRAIN_STD)
])

train_set = TrainingDataset(transform=transform_train)
val_set = ValDataset(transform=transform_val)

train_loader = DataLoader(train_set, batch_size=train_batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=val_batch_size, shuffle=False)



train_losses = []
eval_losses = []

# using Tensorboard-x for recording logs
writer = SummaryWriter(log_dir='RL_ResNet34_unfreeze2')
global_step = 0

# multiply by 0.3 for learning rate every 30 epochs
lr_opt = lambda lr, epoch: lr * (0.3 ** ((epoch) // 30))

for e in range(1, epochs+1):
        
    train_loss = 0
    epoch_miou = 0
    val_miou = 0
    count = 0
    val_count = 0

    #calculate the learning rate
    lr_cur = lr_opt(lr, e)
    #change optimizer's learning rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_cur
    print ('-'*15,'Epoch %d' % e, '-'*15)
    for param_group in optimizer.param_groups:
        lr_read = param_group['lr']
    print('Current learning rate: %f' %lr_read)
    writer.add_scalar("training/learning rate",lr_read, global_step)

    #set net in train() mode
    net.train()
    
    for batch_idx, data in tqdm(enumerate(train_loader)):
        imgs, kinematics_states, controls = data         
        imgs, kinematics_states, controls = imgs.to(device), kinematics_states.to(device), controls.to(device)

        optimizer.zero_grad()

        out = net(imgs.float(), kinematics_states.float())
        # loss = criterion(out, controls.long())
        loss = criterion(out*2, controls.float()*2)

        # measure TOP1 and TOP5 prediction error rate and record loss
        count += 1
        global_step += 1
        writer.add_scalar("training/huber loss", loss.item(), global_step)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        print('huber Loss {:.6f} '.format(loss.item()))       

    train_losses.append(train_loss/count)
    
    if e % print_every == 0:
        print ('Epoch {}/{}...'.format(e, epochs),
                'Average huber Loss {:.6f}'.format(train_loss/count))                       
    
    # test the trained network
    if e % eval_every == 0:
        with torch.no_grad():
            print ('-'*15,'Validation after Epoch %d' % e, '-'*15)
            net.eval()
            eval_loss = 0
            for batch_idx, data in tqdm(enumerate(val_loader)):
                imgs, kinematics_states, controls = data         
                imgs, kinematics_states, controls = imgs.to(device), kinematics_states.to(device), controls.to(device)
                out = net(imgs.float(), kinematics_states.float())

                # measure accuracy and record loss             
                loss_val = criterion(out*2, controls.float()*2)
                val_count += 1                    
                eval_loss += loss_val.item()

            # record the loss and TOP1, TOP5 error after test on the whole test dataset
            writer.add_scalar("val/huber loss", eval_loss/val_count, global_step)
            print ('huber_Loss {:.6f}'.format(eval_loss/val_count))
    
    # save checkpoint file    
    if e % save_every == 0:
        checkpoint = net.state_dict() 
        torch.save(checkpoint, 'RL_ResNet34_freeze2-{}-{:.4f}.pth'.format(e, train_loss/count))
        print ('Model saved!')

    #print the average train_loss till now
    print ('Epoch {}/{}...'.format(e, epochs),
           'Total Average huber Loss: {:.6f}'.format(sum(train_losses) / e))

# writer.close()
print ('[INFO]Training Process complete!')