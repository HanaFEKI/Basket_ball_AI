#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 17:13:41 2025

@author: fenaux
"""

import numpy as np
import cv2
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent.parent
player_detection_path = str(project_root / "Tracker" / "src" / "utils")
sys.path.insert(0, player_detection_path)
from partial_detection_utils import get_adaptive_reference_points


def in_pitch(boxes_file, homog_file, boxes_pitch_file):
    Hs = np.load(homog_file)
    # Hs[i,0] src_pts reference image dst_pts image i
    # Hs[i,1] src_pts grd dst_pts image i
    # Hs[i,2] src_pts image i dst_pts ground
    
    bboxes = np.load(boxes_file)
    inframe = bboxes[:,0].astype(np.int16)
    frames = np.unique(inframe)
    bboxes = bboxes[:,1:5]
    players = np.column_stack((bboxes[:,[0,2]].mean(axis=1), bboxes[:,3])) # middle of bottom
    
    homogs = Hs[:,2]
    
    all_in_pitch = np.array([])
    xy_players = []
    for i_frame in frames:
        
        in_i_frame = np.where(inframe== i_frame)[0]
        players_in_frame = players[in_i_frame]
        M = homogs[i_frame]
        players_grdframe = cv2.perspectiveTransform(players_in_frame.reshape(-1,1,2), M).squeeze()
        in_pitch_x = (players_grdframe[:,0] > -0.5) * (players_grdframe[:,0] < 28.5)
        in_pitch_y = (players_grdframe[:,1] > -0.5) * (players_grdframe[:,1] < 15.5)
        in_pitch = in_pitch_x * in_pitch_y
        
        """
        if i_frame%50 == 0:
            plt.scatter(corners[:,0], corners[:,1], s=1)
            plt.scatter(players_grdframe[in_pitch,0], players_grdframe[in_pitch,1], s=0.5)
            plt.title(f"{i_frame}")
            plt.show()
        
        if i_frame > 500: break"""
        
        all_in_pitch = np.append(all_in_pitch, in_pitch.astype(np.int16))
        if len(xy_players) == 0: xy_players = players_grdframe.copy()
        else: xy_players = np.vstack((xy_players, players_grdframe))
    
    
    bboxes_pitch = np.column_stack((inframe, bboxes, all_in_pitch, xy_players))
    np.save(boxes_pitch_file ,bboxes_pitch)
    
def on_pitch(dict_file:str, homog_file:str):
    """
    Projette les positions des joueurs sur le terrain à l'aide de l'homographie.
    Version originale utilisant le centre bas des boîtes englobantes.
    
    Args:
        dict_file: Fichier contenant les données de tracking
        homog_file: Fichier contenant les matrices d'homographie
    """
    Hs = np.load(homog_file)
    # Hs[i,0] src_pts reference image dst_pts image i
    # Hs[i,1] src_pts grd dst_pts image i
    # Hs[i,2] src_pts image i dst_pts ground
    
    data_dict = np.load(dict_file, allow_pickle=True).item()
    bboxes = data_dict['bboxes']
    inframe = bboxes[:,0].astype(np.int16)
    frames = np.unique(inframe)
    bboxes = bboxes[:,1:5]
    players = np.column_stack((bboxes[:,[0,2]].mean(axis=1), bboxes[:,3])) # middle of bottom
    
    homogs = Hs[:,2]
    
    xy_players = []
    for i_frame in frames:
        
        in_i_frame = np.where(inframe== i_frame)[0]
        players_in_frame = players[in_i_frame]
        M = homogs[i_frame]
        players_grdframe = cv2.perspectiveTransform(players_in_frame.reshape(-1,1,2), M).squeeze()

        if len(xy_players) == 0: xy_players = players_grdframe.copy()
        else: xy_players = np.vstack((xy_players, players_grdframe))
    
    data_dict['xy'] = xy_players.copy()
    np.save(dict_file, data_dict)

def on_pitch_adaptive(dict_file:str, homog_file:str):
    """
    Projette les positions des joueurs sur le terrain à l'aide de l'homographie
    en utilisant des points de référence adaptatifs pour gérer les occlusions partielles.
    
    Args:
        dict_file: Fichier contenant les données de tracking
        homog_file: Fichier contenant les matrices d'homographie
    """
    Hs = np.load(homog_file)
    # Hs[i,0] src_pts reference image dst_pts image i
    # Hs[i,1] src_pts grd dst_pts image i
    # Hs[i,2] src_pts image i dst_pts ground
    
    data_dict = np.load(dict_file, allow_pickle=True).item()
    bboxes = data_dict['bboxes']
    inframe = bboxes[:,0].astype(np.int16)
    frames = np.unique(inframe)
    boxes = bboxes[:,1:5]
    track_ids = data_dict['track_ids']
    
    # Calculer les points de référence adaptatifs
    print("Calcul des points de référence adaptatifs...")
    
    # Utiliser les paramètres de la configuration
    reference_points, partial_detections = get_adaptive_reference_points(
        boxes, track_ids, inframe
    )
    
    # Compter les détections partielles
    num_partial = np.sum(partial_detections)
    total_detections = len(boxes)
    print(f"Détections partielles: {num_partial}/{total_detections} ({num_partial/total_detections*100:.1f}%)")
    
    # Stocker les informations détaillées sur les détections partielles
    data_dict['partial_detection_info'] = {
        'partial_detections': partial_detections,
        'reference_points': reference_points
    }
    
    homogs = Hs[:,2]
    
    xy_players = []
    for i_frame in frames:
        in_i_frame = np.where(inframe == i_frame)[0]
        players_in_frame = reference_points[in_i_frame]
        M = homogs[i_frame]
        players_grdframe = cv2.perspectiveTransform(players_in_frame.reshape(-1,1,2), M).squeeze()

        if len(xy_players) == 0: 
            xy_players = players_grdframe.copy()
        else: 
            xy_players = np.vstack((xy_players, players_grdframe))
    
    # Sauvegarder les résultats avec le préfixe 'adaptive'
    data_dict['xy_adaptive'] = xy_players.copy()
    np.save(dict_file, data_dict)
    
    # On garde également les coordonnées originales pour comparaison
    if 'xy' not in data_dict:
        original_players = np.column_stack((boxes[:,[0,2]].mean(axis=1), boxes[:,3]))
        
        xy_original = []
        for i_frame in frames:
            in_i_frame = np.where(inframe == i_frame)[0]
            players_in_frame = original_players[in_i_frame]
            M = homogs[i_frame]
            players_grdframe = cv2.perspectiveTransform(players_in_frame.reshape(-1,1,2), M).squeeze()

            if len(xy_original) == 0: 
                xy_original = players_grdframe.copy()
            else: 
                xy_original = np.vstack((xy_original, players_grdframe))
                
        data_dict['xy'] = xy_original.copy()
        np.save(dict_file, data_dict)