#!/usr/bin/env python3

# --------------------------------------------------------
# Copyright (C) 2020 NVIDIA Corporation. All rights reserved.
# NVIDIA Source Code License (1-Way Commercial)
# Code written by Shalini De Mello.
# --------------------------------------------------------

import cv2
import numpy as np
from subprocess import call
import torch
from multiprocessing import Process, Value, Queue

import warnings
warnings.filterwarnings("ignore")

from frame_processor_trt import frame_processer
from collect_dataset import board
#import pyrealsense2 as rs

#################################
# Start camera
#################################
print('Adjust brightness=100, contrast=50, sharpness=100 for camera to get the best accuracy')
cam_idx = 3
# adjust these for your camera to get the best accuracy
call('v4l2-ctl -d /dev/video%d -c brightness=100' % cam_idx, shell=True)
call('v4l2-ctl -d /dev/video%d -c contrast=50' % cam_idx, shell=True)
call('v4l2-ctl -d /dev/video%d -c sharpness=100' % cam_idx, shell=True)

print('Start realsense pipeline to obtain intrinsic camera parameters')
#pipe = rs.pipeline()
#config = rs.config()
#config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
#profile = pipe.start(config)
#stream = profile.get_stream(rs.stream.color) # Fetch stream profile for color
#intr = stream.as_video_stream_profile().get_intrinsics() # Downcast to video
#fx, fy, ppx, ppy, coeffs = intr.fx, intr.fy, intr.ppx, intr.ppy, intr.coeffs
#cam_calib = {'mtx': np.array([[fx,0, ppx],
#                              [0 ,fy,ppy],
#                              [0 ,0 ,1]]),
#                              'dist': np.array([coeffs])}
#pipe.stop()
print('Grabbed! intrinsic camera parameters ==> Stop pipeline')
cam_calib = {'mtx': np.array([[925.2764282226562,0,                653.18701171875],
                              [0,                925.1394653320312,350.1488342285156],
                              [0,                0,                1]]),
                              'dist': np.zeros((1, 5))}
# 1280x720  fx: 925.2764282226562 fy: 925.1394653320312 ppx: 653.18701171875 ppy: 350.1488342285156
print('Camera Intrinsic Parameters: ', cam_calib)

#################################
# Load gaze network
#################################
# Set device
print('\nLoad gaze network')
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

# Load pre-trained weights
gaze_network.load_state_dict(torch.load('./pretrained_weights/finetuned32_gaze_network.pth.tar'))

# Initialize monitor and frame processor
mon = board()
frame_processor = frame_processer(cam_calib)

#################################
# Run on live webcam feed and show point of regard on screen
#################################
import time
def get_frames(q, read_frames):
    vid_cap = cv2.VideoCapture('../dataset/tin_2020-11-25 16.17.17_calib.avi')
    ret = True
    while ret:
        ret, img = vid_cap.read()
        time.sleep(1/10)

        if q.full():
            q.get()
        q.put_nowait(img)
    print('\nPipeline of camera process stopped')
    # END

queue = Queue(maxsize = 30)
read_frames = Value('i', True)
p1 = Process(target=get_frames, args=(queue, read_frames, ))
p1.start()
print('\nCamera process started')

data = frame_processor.process(queue, read_frames, mon, device, gaze_network)
p1.terminate()
print('\nCamera process stopped')
