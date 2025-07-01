#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 17:16:16 2025

@author: fenaux
"""

#import os, sys

import numpy as np
from matplotlib import pyplot as plt

import cv2
import sys
import time
"""
from typing import Iterator, List
import supervision as sv
from tqdm import tqdm
from ultralytics import YOLO
"""

import torch
import torchvision.transforms.functional as TF

import supervision as sv
#from yolov6.utils.events import LOGGER, load_yaml
project_root = Path(__file__).resolve().parent.parent  # Goes up from src to Tracker
yolo_path = str(project_root / "Tracker" / "src" / "YOLOv6")
sys.path.insert(0, yolo_path)
import yolov6 as bid # trick to avoid Key Eroor on second runtime
from yolov6.layers.common import DetectBackend
from yolov6.utils.nms import non_max_suppression
from yolov6.core.inferer import Inferer
from pathlib import Path
from typing import List, Optional

project_root = Path(__file__).resolve().parent.parent.parent
player_detection_path = str(project_root / "Boxes_Detection" / "src" / "utils")
sys.path.insert(0, player_detection_path)
from yolov6_utils import check_img_size, process_image_array

video_in = "basket_short.mp4"

supervis= False
if supervis:
    COLORS = ['#FF1493', '#00BFFF', '#FF6347', '#FFD700']
    
    BOX_ANNOTATOR = sv.BoxAnnotator(
        color=sv.ColorPalette.from_hex(COLORS),
        thickness=2
    )
    ELLIPSE_ANNOTATOR = sv.EllipseAnnotator(
        color=sv.ColorPalette.from_hex(COLORS),
        thickness=2
    )
    BOX_LABEL_ANNOTATOR = sv.LabelAnnotator(
        color=sv.ColorPalette.from_hex(COLORS),
        text_color=sv.Color.from_hex('#FFFFFF'),
        text_padding=5,
        text_thickness=1,
    )
    
    PLAYER_DETECTION_MODEL_PATH = 'YOLOv6/football-player-detection.pt'
    #sys.path.append(r"/home/fenaux/Documents/foot/sports-main/examples")
    
    #from soccer.main import run_player_detection
    def run_player_detection(source_video_path: str, device: str, start: int, end: int) -> Iterator[np.ndarray]:
        """
        Run player detection on a video and yield annotated frames.
    
        Args:
            source_video_path (str): Path to the source video.
            device (str): Device to run the model on (e.g., 'cpu', 'cuda').
    
        Yields:
            Iterator[np.ndarray]: Iterator over annotated frames.
        """
        player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
        frame_generator = sv.get_video_frames_generator(source_path=source_video_path, 
                                                        start=start, end=end)
        for frame in frame_generator:
            result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
            detections = sv.Detections.from_ultralytics(result)
    
            annotated_frame = frame.copy()
            annotated_frame = BOX_ANNOTATOR.annotate(annotated_frame, detections)
            annotated_frame = BOX_LABEL_ANNOTATOR.annotate(annotated_frame, detections)
            yield annotated_frame
        
    frame_generator = run_player_detection(source_video_path=video_in, device=device,
                                           start = 0, end = 0 + 2000)
    t0 = time.time()
    video_info = sv.VideoInfo.from_video_path(video_in)
    with sv.VideoSink(target_video_path, video_info) as sink:
        for i_frame, frame in enumerate(frame_generator):
            sink.write_frame(frame)
            if i_frame % 100 == 0: print(i_frame, time.time() - t0)
            
    #source_video_path = 'data/2e57b9_0.mp4'
    
    
    target_video_path = 'palyers.mp4'
    device = 'cuda'
    
    frame_generator = run_player_detection(source_video_path=video_in, device=device,
                                           start = 0, end = 0 + 2000)
    t0 = time.time()
    video_info = sv.VideoInfo.from_video_path(video_in)
    with sv.VideoSink(target_video_path, video_info) as sink:
        for i_frame, frame in enumerate(frame_generator):
            sink.write_frame(frame)
            if i_frame % 100 == 0: print(i_frame, time.time() - t0)
            
            
def func_box(video_name: str, save_box_name: str, start_frame: int, end_frame: int):

    plot_box = True
    
    checkpoint:str ="yolov6l6"  #@param ["yolov6l6", "yolov6s", "yolov6n", "yolov6t"]
    device:str = "gpu"#@param ["gpu", "cpu"]
    half:bool = False #@param {type:"boolean"}
    img_size:int = [1280, 1280]
    
    #Set-up hardware options
    cuda = device != 'cpu' and torch.cuda.is_available()
    device = torch.device('cuda:0' if cuda else 'cpu')
    
    local_checkpoint = f"YOLOv6/{checkpoint}.pt"
    
    model = DetectBackend(local_checkpoint, device=device)
    stride = model.stride
    #class_names = load_yaml("./data/coco.yaml")['names']
    
    if half & (device.type != 'cpu'):
      model.model.half()
    else:
      model.model.float()
      half = False
    
    if device.type != 'cpu':
      model(torch.zeros(1, 3, *img_size).to(device).type_as(next(model.model.parameters())))  # warmup 
    
    conf_thres: float =.25 #@param {type:"number"}
    iou_thres: float =.45 #@param {type:"number"}
    max_det:int =  1000#@param {type:"integer"}
    agnostic_nms: bool = False #@param {type:"boolean"}
    
    model.eval()
    
    classes:Optional[List[int]] = [0] #None # the classes to keep
    conf_thres: float =.25 #@param {type:"number"}
    iou_thres: float =.45 #@param {type:"number"}
    max_det:int =  1000#@param {type:"integer"}
    agnostic_nms: bool = False #@param {type:"boolean"}
    
    img_size:int = 1280#@param {type:"integer"}
    
    img_size = check_img_size(img_size, s=stride)

    
    batch_size = 8 # mieux que 8
    
    bboxes = []
    video_capture = cv2.VideoCapture()
    
    if video_capture.open(video_name):

      video_capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
      
      length = end_frame - start_frame
    
    
    with torch.no_grad():
        start = time.time()
        i_frame = 0
        while i_frame < length:
            imgs = []
            for i in range(batch_size):
                ret, frame = video_capture.read()
                if not ret:
                    break
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if (i_frame + i) % 100 == 0: 
                    annotated_img = img.copy()
                img, img_src = process_image_array(img, img_size, stride, half)
                img = img.to(device)
                img = img[None]
                
                if len(imgs)==0: imgs = img
                else: imgs = torch.cat((imgs, img))
              
            pred_results = model(imgs)
              
            dets = non_max_suppression(pred_results, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
                
            for det in dets:
                if len(det):
                  det[:, :4] = Inferer.rescale(img.shape[2:], det[:, :4], img_src.shape).round()
                  boxes = det.cpu().numpy()
                  boxes = np.hstack((np.ones((len(det),1)) * i_frame, boxes[:,:5]))

                  #bboxes.append(boxes)
                  if len(bboxes)==0: bboxes = boxes.copy()
                  else: bboxes = np.vstack((bboxes, boxes))
          
                
                if (i_frame % 100) == 0: 
                    print(f" yolo  current frame {i_frame} average time per frame {(time.time() - start) / (i_frame + 1)} s")
                    if plot_box:
                        for x1, y1, x2, y2 in boxes[:,1:5].astype(np.int16):
                            cv2.rectangle(annotated_img, (x1,y1), (x2,y2), (0,255,0), 2)
                        plt.imshow(annotated_img)
                        plt.axis('off')
                        plt.title(f"frame {i_frame + start_frame}")
                        plt.show()
                    
                i_frame += 1

    np.save(save_box_name, bboxes)
    
    return
