#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration centralisée pour les paramètres du projet.
Permet d'ajuster facilement les paramètres sans avoir à modifier plusieurs fichiers.
"""

# Paramètres pour la détection des occlusions partielles
PARTIAL_DETECTION_CONFIG = {
    # Seuil de chute de hauteur pour détecter une occlusion partielle (0.0-1.0)
    # Plus la valeur est petite, plus les occlusions seront détectées facilement
    "height_drop_threshold": 0.3,
    
    # Seuil de changement du ratio hauteur/largeur (0.0-1.0)
    # Plus la valeur est petite, plus les changements mineurs seront détectés
    "aspect_ratio_change_threshold": 0.2,
    
    # Taille de la fenêtre pour le calcul des statistiques temporelles
    "window_size": 20,
    
    # Stabilité de la largeur (0.0-1.0)
    # Plus la valeur est grande, plus on tolère des variations de largeur
    "width_stability_threshold": 0.6
}

# Configuration des fichiers (peut être modifiée en fonction des besoins)
FILE_CONFIG = {
    "homog_file": "/Terrain_Detection/src/data/annotations/Hs_supt1.5.npy",
    "boxes_file": "/Boxes_Detection/src/data/annotations/boxes.npy",
    "video_in": "/basket_short.mp4",
    "folder_video_out": "/Tracker/src/data/videos",
    "track_file": "/Tracker/src/data/annotations/tracks_clip.npy",
    "dict_file": "/Tracker/src/data/annotations/clip_dict_4.npy",
    "adaptive_dict_file": "/Tracker/src/data/annotations/clip_dict_adaptive.npy"
}

# Configuration de la visualisation
VISUALIZATION_CONFIG = {
    "start_frame": 0,
    "end_frame": 2000,
    "trace_length": 30  # Nombre de frames pour conserver la trace dans le radar
} 
