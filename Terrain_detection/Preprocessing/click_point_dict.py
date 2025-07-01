2#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 09:21:50 2024

@author: fenaux
"""

import os, sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector, Button
from matplotlib.backend_bases import MouseButton

import cv2


def select_roi(eclick, erelease):
    """
    Callback for line selection.

    *eclick* and *erelease* are the press and release events.
    """
    
    global roi
    
    x1, y1 = eclick.xdata, eclick.ydata
    x2, y2 = erelease.xdata, erelease.ydata
    #print(f"({x1:3.2f}, {y1:3.2f}) --> ({x2:3.2f}, {y2:3.2f})")
    #print(f"The buttons you used were: {eclick.button} {erelease.button}")
    roi = np.int16( [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)] )

def button_press_callback(event):
    'whenever a mouse button is pressed'
    global pts_local, image
    
    image_pt = image.copy()
    if event.inaxes is None:
        return
    
    h, w = image.shape[:2]
    ix, iy = event.xdata, event.ydata
    if ix < 1 or ix > w or iy < 1 or iy > h: # > 1 is a trick to avoid considering click on button
        return
    new_pt = np.array([event.xdata, event.ydata]).reshape(1,2)
    
    if event.button is MouseButton.LEFT:
        pts_local = np.append(pts_local, new_pt, axis=0)
        #new_idx = int(input())
        
    if event.button is MouseButton.RIGHT:
        if len(pts_local) > 0:
            to_delete = np.argmin(np.linalg.norm(pts_local - new_pt, axis=1))
            pts_local = np.delete(pts_local, to_delete, axis=0)
    
    if event.button == 2: return

    global imgplot, fig
    for pt in pts_local:
        cv2.circle(image_pt, pt.astype(np.int16), 3, (0,255,0), -1)
    imgplot.set_data(image_pt)
    fig.canvas.draw_idle()
    
    
def on_close(event):
    truc = 0
    
    
def terminate(event):
    global finished
    finished = True
    plt.close('all')
    

i_frame = [1, 2, 3]  
i=2
img_name = f"../data/input_imgs/img_{i_frame[i]}.png"
pts_name = f"../data/annotations/pts_dict_{i_frame[i]}.npy"

img = cv2.imread(img_name)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_ori = img.copy()

global pts, finished

if os.path.exists(pts_name):
    pts_dict = np.load(pts_name, allow_pickle=True).item()
    pts = pts_dict['pts']
else: 
    pts_dict = {}
    pts = np.array([]).reshape(-1,2)

finished = False

while not finished:
    #####################
    # selection of zone to keep
    #####################
    rgb = img.copy()
    for pt in pts:
        cv2.circle(rgb, pt.astype(np.int16), 3, (0,255,0), -1)
    fig_all, ax = plt.subplots(figsize=(13, 7))
    ax.imshow(rgb)
    
    selector = RectangleSelector(
    ax, select_roi,
    useblit=True,
    button=[1, 3],  # disable middle button
    minspanx=5, minspany=5,
    spancoords='pixels',
    interactive=True)
    
    axcolor = 'lightgoldenrodyellow'
    lineax1 = plt.axes([0.8, 0.025, 0.1, 0.04])
    button = Button(lineax1, 'terminate', color=axcolor, hovercolor='0.975')
    button.on_clicked(terminate)
    
    plt.axis("off")
    plt.title ('close window when selection of zone is correct')
    plt.show()
    fig_all.canvas.mpl_connect('close_event', on_close)
    
    if not finished:

        global roi
        xt, yt, xb, yb = roi
        
        global pts_local
        pts_local = np.array([]).reshape(-1,2)
        
        global fig
        fig, ax = plt.subplots(figsize=(16,9))
        global imgplot
        global image
        #image = image_in.copy()
        #ax.axis=("off")
        image = rgb[yt:yb,xt:xb]
        imgplot = plt.imshow(image)
        
        # to close the window when line is correctly selected
        fig.canvas.mpl_connect('close_event', on_close)
        
        
        
        # Call click func
        global cid
       
        cid = fig.canvas.mpl_connect('button_press_event', button_press_callback)
        button.disconnect(cid)
        
        plt.axis("off")
        #plt.tight_layout()
        plt.title('to correct a point right click next to it when finish close window')
        plt.show()
        
        if len(pts)==0: pts = pts_local + np.array([xt, yt]).reshape(1,2)
        else: pts = np.vstack((pts, pts_local + np.array([xt, yt]).reshape(1,2)))

pts_dict['pts'] = pts.copy()      
np.save(pts_name, pts)

#pts = np.load(pts_name_b)
try: idents = pts_dict['ident'].astype(np.int16)
except: idents = []
annot = True

font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 2
fontColor = (0,0,0)
thickness = 2
lineType = 2

if len(idents) == len(pts):
    annot = False
    
    for ident, pt in zip(idents, pts.astype(np.int16)):
        cv2.circle(img, pt, 5, (255,0,0), -1)
        cv2.putText(img,f"{ident}", pt, font, fontScale, fontColor, thickness, lineType)
    plt.imshow(img)
    plt.axis('off')
    plt.show()
    
    print('Nouvelle annotation y / n')
    if input() == 'y': annot=True
    
if annot:
    #idents = np.array([])
    new_pts = pts[len(idents):]
    for pt in new_pts:
        im_plot  = img_ori.copy()
        cv2.circle(im_plot, pt.astype(np.int16), 5, (0,255,0), -1)
        plt.imshow(im_plot)
        plt.title('close window and enter point identity on keyboard')
        plt.axis('off')
        plt.show()
        
        new_in = int(input())
        idents = np.append(idents, new_in)

    pts_dict['ident'] = idents.astype(np.int16)
    np.save(pts_name, pts_dict)
    
    for ident, pt in zip(idents, pts.astype(np.int16)):
        cv2.circle(img, pt, 5, (255,0,0), -1)
        cv2.putText(img,f"{ident}", pt, font, fontScale, fontColor, thickness, lineType)
    plt.imshow(img)
    plt.axis('off')
    plt.title('current annotation')
    plt.show()