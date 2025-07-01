#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 08:52:30 2025

@author: fenaux
"""

import numpy as np
import matplotlib.pyplot as plt

import cv2


h = 1080
w = 1920
"""
# comparaision par écart des points à l'écran moins adaptée pour un tracking des joueurs
homogs_ref = np.load('Hs_supt1.npy')[:,1]
homogs = np.load('Hs_kaze1.npy')[:,1]

pitch = np.load('pitch.npy')
pitch_reshaped = pitch.reshape(-1,1,2)
errors = []
for homog, homog_ref in zip(homogs, homogs_ref):
    screen_pts = cv2.perspectiveTransform(pitch_reshaped, homog).squeeze()
    screen_pts_ref = cv2.perspectiveTransform(pitch_reshaped, homog_ref).squeeze()
    
    in_screen_x = (screen_pts[:,0] > 0) * (screen_pts[:,0] < w)
    in_screen_y = (screen_pts[:,1] > 0) * (screen_pts[:,1] < h)
    in_screen = in_screen_x * in_screen_y
    
    deltas = (screen_pts - screen_pts_ref)[in_screen]
    deltas_norm = np.linalg.norm(deltas, axis=1)
    errors.append( [deltas_norm.mean(), np.median(deltas_norm), deltas_norm.max()])
errors = np.array(errors)

# illustration des erreurs
ul = 10
i = 1
err_ul = errors[:,i] < ul
#err2_ul = errors_2[:,i] < ul
plt.hist(errors[err_ul,i], bins=100)
#plt.hist(errors_2[err2_ul,i], bins=100)
#plt.hist(np.column_stack((errors[err2_ul,i], errors_2[err2_ul,i])), bins=100)
plt.xlim(0,ul), 
#plt.legend(['resize','2'])
"""

homogs_ref = np.load('Hs_supt1.npy')[:,2]
homogs = np.load('Hs_supt1.npy')[:,2]

#5 Fictional players
players = np.array([[w / 4, h / 4], [3 * w / 4 , h / 4],
                    [w / 4, 3 * h / 4], [3 * w / 4 , 3 * h / 4],
                    [w / 2, h / 2]])

players_reshaped = players.reshape(-1,1,2)

# correction d'une erreur isolée facilement détectable par 
# différence au filtrage type mad med
homogs[629] = homogs[[628,630]].mean(axis=0)


#H = np.reshape(homogs, (-1,9))

n_frame = len(homogs)
idxs = np.arange(n_frame)
from scipy.interpolate import make_interp_spline
#Do an interpolation between homographies-->Smoother set of transformations
spl = make_interp_spline(idxs[::2], homogs[::2], k=1, axis=0)  # k=1: linear
# for k=1: bc_type can only be None (otherwise bc_type="natural")
homogs = spl(idxs)

errors = []
traj = []
traj_ref = []
for homog, homog_ref in zip(homogs, homogs_ref):
    grd_pts = cv2.perspectiveTransform(players_reshaped, homog).squeeze()
    grd_pts_ref = cv2.perspectiveTransform(players_reshaped, homog_ref).squeeze()
        
    deltas = (grd_pts - grd_pts_ref)
    deltas_norm = np.linalg.norm(deltas, axis=1)
    errors.append(deltas_norm)
    traj.append(grd_pts)
    traj_ref.append(grd_pts_ref)
    
errors = np.array(errors)
err_max = errors.max(axis=1)
traj = np.array(traj)
traj_ref = np.array(traj_ref)

_ = plt.hist(err_max, bins=100)
#plt.xlim(0,0.2)
plt.xlabel('écart en m')
plt.show()