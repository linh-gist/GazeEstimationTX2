#!/usr/bin/env python3

# --------------------------------------------------------
# Copyright (C) 2020 NVIDIA Corporation. All rights reserved.
# NVIDIA Source Code License (1-Way Commercial)
# Code written by Shalini De Mello.
# --------------------------------------------------------

import time
import cv2
import numpy as np
from os import path
from subprocess import call
import pickle
import sys
import torch
import os

import warnings
warnings.filterwarnings("ignore")

from frame_processor_finetune import frame_processer, fine_tune
from collect_dataset import board

#################################
# Start camera
#################################
cam_calib = {'mtx': np.array([[925.2764282226562,0,                653.18701171875],
                              [0,                925.1394653320312,350.1488342285156],
                              [0,                0,                1]]),
                              'dist': np.zeros((1, 5))}
# 1280x720  fx: 925.2764282226562 fy: 925.1394653320312 ppx: 653.18701171875 ppy: 350.1488342285156

#################################
# Load gaze network
#################################
ted_parameters_path = 'pretrained_weights/weights_ted.pth.tar'
maml_parameters_path = 'pretrained_weights/weights_maml'
k = 32

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Create network

from models import DTED
gaze_network = DTED(
    growth_rate=32,
    z_dim_app=64,
    z_dim_gaze=2,
    z_dim_head=16,
    decoder_input_c=32,
    normalize_3d_codes=True,
    normalize_3d_codes_axis=1,
    backprop_gaze_to_encoder=False,
).to(device)

#################################

# Load T-ED weights if available
assert os.path.isfile(ted_parameters_path)
print('> Loading: %s' % ted_parameters_path)
ted_weights = torch.load(ted_parameters_path)
if torch.cuda.device_count() == 1:
    if next(iter(ted_weights.keys())).startswith('module.'):
        ted_weights = dict([(k[7:], v) for k, v in ted_weights.items()])

#####################################

# Load MAML MLP weights if available
full_maml_parameters_path = maml_parameters_path +'/%02d.pth.tar' % k
assert os.path.isfile(full_maml_parameters_path)
print('> Loading: %s' % full_maml_parameters_path)
maml_weights = torch.load(full_maml_parameters_path)
ted_weights.update({  # rename to fit
    'gaze1.weight': maml_weights['layer01.weights'],
    'gaze1.bias':   maml_weights['layer01.bias'],
    'gaze2.weight': maml_weights['layer02.weights'],
    'gaze2.bias':   maml_weights['layer02.bias'],
})
gaze_network.load_state_dict(ted_weights)

#################################
# Personalize gaze network
#################################

# Initialize monitor and frame processor
mon = board()
frame_processor = frame_processer(cam_calib)

# collect person calibration data and fine-tune gaze network
subject = '../dataset/tin_2020-11-25 16.17.17_calib'
# adjust steps and lr for best results, To debug calibration, set show=True
gaze_network = fine_tune(subject, frame_processor, mon, device, gaze_network, k, steps=1000, lr=1e-5, show=True)
print('Finetuned => Done')
#################################
# Run on live webcam feed and show point of regard on screen
#################################
#subject = './dataset/linh_2020-10-26 17.55.37_calib'
#vid_cap = cv2.VideoCapture('%s.avi' % subject)
#data = frame_processor.process(subject, vid_cap, mon, device, gaze_network, show=True)
