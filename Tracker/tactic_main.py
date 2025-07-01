#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 14:27:18 2025

@author: fenaux
"""

import numpy as np
from matplotlib import pyplot as plt

import os, sys
import time
import gc

import cv2
import supervision as sv

from typing import List

from func_in_pitch import on_pitch, on_pitch_adaptive
from pitch_utils import draw_points_on_pitch, run_radar
from track_utils import run_sv_tracker
from track_utils import track_in_pitch, box_and_track, ChainTrack, GraphTrack
from track_utils import ShowTrackHmm, crop_track
from render_track import plot_tracks
from team import TeamClassifier, get_crops, create_batches
from team import HMMarkov
from team import HMM_missings

# Activer/désactiver les différentes étapes du pipeline
store_to_keep = True
do_team_classif = True # take long time
do_HMMmissings = False
do_chain_track = True
do_graph_track = True
use_adaptive_homography = False  # Nouveau flag pour utiliser l'homographie adaptative

device = 'cuda'

path_to_here = os.path.split(os.path.abspath(os.path.realpath(sys.argv[0])))[0]
video_in = os.getcwd()+"/basket_short.mp4"
homog_file = path_to_here + '/pitch/Hs_supt1.npy'
pitch_file = path_to_here + '/pitch.npy'
video_track = path_to_here + '/video_clip.mp4'


boxes_file = path_to_here + '/boxes.npy'
track_file = path_to_here + '/tracks_clip.npy'
dict_file = path_to_here + '/clip_dict_4.npy'
# Nouveau fichier pour stocker les résultats avec homographie adaptative
adaptive_dict_file = path_to_here + '/clip_dict_adaptive.npy'

#pitch = np.load(pitch_file)
#corners = pitch[[0,1,4,5]].copy()
#lines = [[0,1], [0,2],[1,2],[2,3]]

if not os.path.exists(boxes_file):
    from func_players_batch import func_box
    func_box(video_in, boxes_file, start_frame=0, end_frame=0+2000)

track_array = []
byte_dict = run_sv_tracker(boxes_file)
for index, bbox in enumerate(byte_dict['bboxes']):
    track_array.append(np.hstack([bbox, byte_dict['track_ids'][index]]))
np.save(track_file, track_array)

box_and_track(boxes_file, track_file, dict_file)

# Utiliser l'homographie standard ou adaptative selon le flag
if os.path.exists(dict_file):
    if use_adaptive_homography:
        # Copier le fichier dict_file vers adaptive_dict_file pour préserver l'original
        if not os.path.exists(adaptive_dict_file):
            data_dict = np.load(dict_file, allow_pickle=True).item()
            np.save(adaptive_dict_file, data_dict)
        # Utiliser la méthode adaptative
        on_pitch_adaptive(adaptive_dict_file, homog_file)
    else:
        # Utiliser la méthode standard
        on_pitch(dict_file, homog_file)

# Utiliser le fichier approprié pour la suite du traitement
working_dict_file = adaptive_dict_file if use_adaptive_homography else dict_file

track_in_pitch(dict_file)
#starts_ends = StartsEnds(dict_file, pitch_only=True)

        

#run_player_tracking(boxes_file) # remplacé par boTSort (colab)

#run_radar(video_track, dict_file,
#              'radar_bot.mp4')#,
              #start=0, end=0 + 2000)
              


track_dict = np.load(working_dict_file, allow_pickle=True).item()

bboxes_ = track_dict['bboxes']
inframe_ = bboxes_[:,0].astype(np.int16)
boxes_ = bboxes_[:,1:5]
# Utiliser les coordonnées adaptatives si disponibles
xy_ = track_dict['xy_adaptive'] if 'xy_adaptive' in track_dict else track_dict['xy']
track_ids_ = track_dict['track_ids']

not_in_pitch = np.logical_and( np.logical_not(track_dict['in_pitch']),
                              track_ids_ > 0)
track_ids_[not_in_pitch] *= -1
track_ids_[not_in_pitch] -= 1

idx_tracks_valid = np.where(track_ids_ > -1)[0]
track_ids = track_ids_[idx_tracks_valid]

vits = []
unique_track_ids = np.unique(track_ids)
print(unique_track_ids)
print(xy_)
print(inframe_)
for i_track in unique_track_ids:
    in_track = np.where(track_ids_ == i_track)[0]
    if len(in_track) > 1:
        dxdy = np.gradient(xy_[in_track], inframe_[in_track], axis=0)
        ds = np.linalg.norm(dxdy, axis=1)
        vits.append(np.quantile(ds, 0.9)) # / inframe_[in_track].ptp())
    else:
        vits.append(np.quantile(1, 0.9)) # / inframe_[in_track].ptp())
vits = np.array(vits)

slow_threshold = 0.14 # with 30 fps
where_slow = np.where(vits < slow_threshold)[0]
# vits_slow = vits[vits < slow_threshold]
# sort_idx_slow = idx_tracks[where_slow[np.argsort(vits_slow)]]
unique_track_ids = np.delete(unique_track_ids, where_slow)
vits = np.delete(vits, where_slow)

track_to_keep = np.zeros(track_ids_.size)
for i_track in unique_track_ids:
    in_track = np.where(track_ids_ == i_track)
    track_to_keep[in_track] = 1
    
if store_to_keep: 
    track_dict['to_keep'] = track_to_keep.astype('bool_')
    np.save(working_dict_file, track_dict)


if do_team_classif:
    # to initialze classifier we take a strict definition of pitch
    # in order to exclude coaches or public
    is_in_pitch_x = (xy_[:,0] > 0) * (xy_[:,0] < 28)
    is_in_pitch_y = (xy_[:,1] > 0) * (xy_[:,1] < 15)
    is_in_pitch = is_in_pitch_x * is_in_pitch_y
    
    inframe = inframe_[is_in_pitch]
    boxes = boxes_[is_in_pitch]
    
    stride = 50#100
    start = 0
    source_video_path = video_in
    source = sv.get_video_frames_generator(
        source_path=source_video_path, start=0, end=0 + 2000, stride=stride)
    
    crops = []
    for i_frame, frame in enumerate(source):
        
        in_this_frame = np.where(inframe == i_frame * stride)[0]
        detections = sv.Detections(xyxy = boxes[in_this_frame])
        crops += get_crops(frame, detections)
    
    t0 = time.time()
    team_classifier = TeamClassifier(device=device)
    team_classifier.fit(crops)
    
    
    move_idx = np.hstack([np.where(track_ids_ == track_id)[0] for track_id in unique_track_ids])
    inframe_move = inframe_[move_idx]
    boxes_move = boxes_[move_idx]
    track_ids_move = track_ids_[move_idx]
    source = sv.get_video_frames_generator(
        source_path=source_video_path, start=0, end=0 + 2000)
    
    crops = []
    idx_crops = np.array([]).astype(np.int16)
    boxes_ = np.clip(boxes_, 0, None)
    player_team_id = np.array([])
    
    for i_frame, frame in enumerate(source):
        in_this_frame = np.where(inframe_ == i_frame)[0]
        in_this_frame_move = in_this_frame[np.isin(track_ids_[in_this_frame], unique_track_ids)]
        detections = sv.Detections(xyxy = boxes_[in_this_frame_move])
        crops += get_crops(frame, detections)
        idx_crops = np.append(idx_crops, in_this_frame_move)
        
        if (i_frame >= 2000) or (len(crops) > 256):
            new_team_id = team_classifier.predict(crops)
            player_team_id = np.append(player_team_id, new_team_id)
            print(i_frame, len(crops))
            crops = []
    
    if len(crops) > 0:
        new_team_id = team_classifier.predict(crops)
        player_team_id = np.append(player_team_id, new_team_id)
    
    team_id = -np.ones(len(boxes_))
    team_id[idx_crops] = player_team_id
    print(time.time() - t0)
    track_dict['team_id'] = team_id.astype(np.int16)
    np.save(working_dict_file, track_dict)

##### first version of hmm (clip_dict_3.npy)

track_ids_hmm, team_id_hmm = HMMarkov(unique_track_ids, 
                                      track_ids_.copy(), track_dict['team_id'].copy())

track_dict['track_ids_hmm']= track_ids_hmm.copy()
track_dict['team_id_hmm'] = team_id_hmm.copy()
np.save(working_dict_file,track_dict)

######### second version of hmm with missings (clip_dict_4.npy)
if do_HMMmissings:
    track_dict = np.load(working_dict_file, allow_pickle=True).item()
    
    to_keep = np.where(track_dict['to_keep'])[0]
    bboxes_ = track_dict['bboxes'][to_keep]
    boxes_ = bboxes_[:,1:5]
    inframe = bboxes_[:,0].astype(np.int16)
    xy = track_dict['xy_adaptive'] if 'xy_adaptive' in track_dict else track_dict['xy'][to_keep]
    team_id_ = track_dict['team_id'][to_keep]
    track_ids = track_dict['track_ids'][to_keep]
    
    track_ids_hmm, team_id_hmm= HMM_missings(track_ids, inframe, team_id_, show_plot=True)
    
    track_dict['track_ids_hmm']= track_dict['track_ids'].copy()
    track_dict['team_id_hmm'] = track_dict['team_id'].copy()
    
    track_dict['track_ids_hmm'][to_keep] = track_ids_hmm.copy()
    track_dict['team_id_hmm'][to_keep] = team_id_hmm.copy()
    np.save(working_dict_file, track_dict)



if do_chain_track:
    track_ids_chain = ChainTrack(working_dict_file)
    # track_ids_chain is stored in ChainTrack


#ShowTrackHmm(dict_file, video_in, unique_track_ids, stride = 4)
"""
in_team = np.where( team_id_ == 0)[0]
track_ids = track_dict['track_ids_chain'][to_keep]
track_ids_team = track_ids[in_team]
unique_track_ids = np.unique(track_ids_team)

for track_id in unique_track_ids:
    in_track = np.where(track_ids == track_id)[0]
    plt.scatter(inframe[in_track], np.ones(in_track.size) * track_id, s=0.5)
plt.ylabel('track id'), plt.xlabel('frame number')
"""

start_chain = 100
end_chain = 1300
if do_graph_track:
    GraphTrack(working_dict_file, start_chain, end_chain, fps=30, show_tracks=False)
"""
plot_tracks(source_video_path='basket_short.mp4', dict_file=dict_file,
              target_video_path='graph_track.mp4', track_kind='graph',
              start=0, end=-1)
"""

data_dict = np.load(working_dict_file, allow_pickle=True).item()
bboxes_ = data_dict['bboxes']
inframe = bboxes_[:,0].astype(np.int16)
#frames = np.unique(inframe)
bboxes = bboxes_[:,1:5]

track_kind = 'graph'
if track_kind == 'hmm': track_ids = data_dict['track_ids_hmm'].astype(np.int16)
elif track_kind == 'chain': track_ids = data_dict['track_ids_chain'].astype(np.int16)
elif track_kind == 'graph': track_ids = data_dict['track_ids_graph'].astype(np.int16)
else: track_ids = data_dict['track_ids'].astype(np.int16)

to_keep = data_dict['to_keep'] if 'to_keep' in data_dict else np.ones_like(track_ids, dtype=bool)
inframe = inframe[to_keep]
bboxes = bboxes[to_keep]
track_ids = track_ids[to_keep]
# Utiliser les coordonnées adaptatives si disponibles
xy = data_dict['xy_adaptive'] if 'xy_adaptive' in data_dict else data_dict['xy']
xy = xy[to_keep]
team_id_ = track_dict['team_id_hmm'][to_keep]

# search for complete tracks or longer ones if less tha 5 in team
for i_team in range(2):
    start_end = []
    in_team = np.where( team_id_ == i_team)[0]
    track_ids_team = track_ids[in_team]
    unique_track_ids = np.unique(track_ids_team)
    for track_id in unique_track_ids:
        in_track = np.where(track_ids == track_id)[0]
        inframe_track = inframe[in_track].astype(np.int16)
        start_end.append(inframe_track[[0,-1]])
    start_end = np.array(start_end)
    complete_idx = np.where((start_end[:,0] <= start_chain) * (start_end[:,1] >= end_chain))[0]
    complete_traj_team = unique_track_ids[complete_idx]
    remain = 5 - len(complete_traj_team)
    if remain > 0:
        start_end = np.delete(start_end, complete_idx, axis=0)
        short_track_ids = np.delete(unique_track_ids, complete_idx)
        if len(short_track_ids) > remain:
            traj_length = start_end.ptp(axis=1)
            short_traj = short_track_ids[np.argsort(traj_length[-remain:])]
        else:
            short_traj = short_track_ids.copy()
        complete_traj_team = np.append(complete_traj_team, short_traj)
        
    if i_team == 0: complete_traj = complete_traj_team.copy()
    else: complete_traj = np.append(complete_traj, complete_traj_team)

# smooting and interpolation
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from scipy.signal import sosfiltfilt, butter

ls = 50
ls_bds = (20, 1e2) 

xy_preds = []
for track_id in complete_traj:
    in_track = np.where(track_ids == track_id)[0]
    inframe_track = inframe[in_track].astype(np.int16)

    boxes_in_frame = bboxes[in_track]
    hboxes = np.abs(np.diff(boxes_in_frame[:,1::2], axis=1)).squeeze()
    xy_in_track = xy[in_track]

    # Modification de la logique de filtrage pour conserver plus de détections
    # Calcul des statistiques de hauteur de boîte
    med = np.median(hboxes)
    mad = np.median(np.abs(hboxes - med))  # Utiliser l'écart absolu médian standard
    
    # Paramètre moins strict pour la sélection des boîtes trop petites (2.5 au lieu de 2)
    too_small = np.where(hboxes < med - 3.0 * mad)
    
    in_track_valids = np.delete(in_track, too_small)
    valids = np.delete(inframe_track, too_small)
    hvalids = np.delete(hboxes, too_small)
    range_track = np.arange(inframe_track.min(), inframe_track.max() + 1)
    Xpred = range_track.reshape(-1,1)
    
    hbInterp = np.interp(range_track, valids, hvalids)
    sos = butter(4, 2, 'low', fs=30, output='sos')
    hbLow = sosfiltfilt(sos, hbInterp)
            
    dhLow = hvalids - hbLow[(valids - valids[0])]
    medLow = np.median(dhLow)
    madLow = np.median(np.abs(dhLow - medLow))
    
    # Paramètre moins strict pour les anomalies (2.5 au lieu de 2)
    anoms = np.where(np.abs(dhLow - medLow) > 3.0 * madLow)
    valids = np.delete(valids, anoms)
    in_track_valids = np.delete(in_track_valids, anoms)
    hb_valids = np.abs(np.diff(bboxes[in_track_valids,1::2], axis=1)).squeeze()
    xy_valids = xy[in_track_valids]
    
    X = inframe_track.reshape(-1,1)
    kernel = 1 * RBF(length_scale=ls, length_scale_bounds=ls_bds)
    gaussian_process = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
    gaussian_process.fit(X,  xy_in_track[:,0])
    x_prediction = gaussian_process.predict(Xpred)
    
    X = valids.reshape(-1,1)
    kernel = 1 * RBF(length_scale=ls, length_scale_bounds=ls_bds)
    gaussian_process = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
    gaussian_process.fit(X,  xy_valids[:,1])
    y_prediction = gaussian_process.predict(Xpred)
    
    xy_preds.append(np.column_stack((x_prediction, y_prediction)))

complete_track_dict = {}
complete_track_dict['ids'] = complete_traj
complete_track_dict['xy'] = xy_preds

# Sauvegarder les trajectoires complètes avec le préfixe correspondant à l'homographie utilisée
print("saved")
np.save(f"complete_tracks_{'adaptive' if use_adaptive_homography else 'standard'}.npy", complete_track_dict)




