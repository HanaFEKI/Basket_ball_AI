#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 14:26:53 2024

@author: fenaux
"""

import sys
import matplotlib.pyplot as plt
import cv2

video_in = "../../../TacTic/src/data/basket_game.mp4",
video_out = 'cut0.mp4'
image_name = 'img_0.png'


init_frame = 104700+75+35
video_capture = cv2.VideoCapture()
if video_capture.open( video_in ):
    w, h = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    frame_count = video_capture.get(cv2.CAP_PROP_FRAME_COUNT)
    video_capture.set(cv2.CAP_PROP_POS_FRAMES, init_frame) 
    ret, frame = video_capture.read()
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #cv2.imwrite(f"img_{init_frame}.png", frame)
    plt.axis('off')
    plt.imshow(img)
    
