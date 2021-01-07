# Airsim_imitation-reinforcement-learning
This is the code for my published paper: Improved Reinforcement Learning through Imitation Learning Pretraining Towards Image-based Autonomous Driving, ICCAS 2019
[![video](gif/final_camera.gif)](https://www.youtube.com/watch?v=yjmM70alCSQ&feature=youtu.be)

## my collected data for imitation learning stage
[Google drive link for training data](https://drive.google.com/file/d/1WVL1wdGnKJSbsLFM9pflKOJxj7JhCR4G/view?usp=sharing)
[Google drive link for validation data](https://drive.google.com/file/d/1NohcpousVPlm5DmoUfZjTlnmSLzhMOnO/view?usp=sharing)
After download these files, put them under ./Airsim_imitation-reinforcement-learning folder.

## Pretrained weights
After imitaion learning stage:
`checkpoint_imitation.pth`

After reinforcement learning stage for performance improvement:
`checkpoint_reinforcement.pth`

## Imitation Learning for Pretraining
`train_imitation.py`

## Reinforcement Learning for performance improvement
`train_reinforcement.py`



