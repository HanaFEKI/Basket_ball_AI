#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script principal pour exécuter notre solution d'homographie adaptative
et comparer les résultats avec la méthode standard.
"""

import os
import numpy as np
import argparse
import sys
from pathlib import Path
from func_in_pitch import on_pitch, on_pitch_adaptive
project_root = Path(__file__).resolve().parent.parent.parent
player_detection_path = str(project_root / "Tracker" / "src" / "utils")
sys.path.insert(0, player_detection_path)
from track_utils import run_sv_tracker, track_in_pitch, box_and_track, ChainTrack, GraphTrack
from pitch_utils import run_radar, run_radar_adaptive
from team import HMM_missings
from render_track import plot_tracks
from config import FILE_CONFIG, VISUALIZATION_CONFIG
from compare_trajectories import add_trajectory_comparison_to_pipeline

store_to_keep = True
do_team_classif = True # take long time
do_HMMmissings = True
do_chain_track = False
do_graph_track = False
use_adaptive_homography = True  # Nouveau flag pour utiliser l'homographie adaptative

def run_pipeline(use_adaptive=True, start_from_scratch=False):
    """
    Exécute le pipeline complet avec ou sans homographie adaptative.
    
    Args:
        use_adaptive: Utiliser l'homographie adaptative
        start_from_scratch: Recalculer les étapes depuis le début
    """
    # start_from_scratch=True
    # Configuration des fichiers
    path_to_here = os.path.split(os.path.abspath(os.path.realpath(sys.argv[0])))[0]
    video_in = os.getcwd() + FILE_CONFIG["video_in"]
    folder_video_out = Path(os.getcwd() + FILE_CONFIG["folder_video_out"])
    #homog_file = os.getcwd() + FILE_CONFIG["homog_file"]
    homog_file = path_to_here + '/pitch/Hs_supt1.npy'
    boxes_file = os.getcwd() + FILE_CONFIG["boxes_file"]
    track_file = os.getcwd() + FILE_CONFIG["track_file"]
    dict_file = os.getcwd() + FILE_CONFIG["dict_file"]
    adaptive_dict_file = os.getcwd() + FILE_CONFIG["adaptive_dict_file"]
    
    # Fichier de travail (standard ou adaptatif)
    working_dict_file = adaptive_dict_file if use_adaptive else dict_file
    
    # Étape 1: Détection des boîtes (si nécessaire)
    if start_from_scratch and not os.path.exists(boxes_file):
        from func_players_batch import func_box
        print("Détection des joueurs avec YOLOv6...")
        func_box(video_in, boxes_file, start_frame=0, end_frame=0+2000)
    
    # Étape 2: Tracking (si nécessaire)
    if start_from_scratch and (not os.path.exists(dict_file) or not os.path.exists(track_file)):
        print("Tracking des joueurs avec ByteTrack...")
        track_array = []
        byte_dict = run_sv_tracker(boxes_file)
        for index, bbox in enumerate(byte_dict['bboxes']):
            track_array.append(np.hstack([bbox, byte_dict['track_ids'][index]]))
        np.save(track_file, track_array)
        box_and_track(boxes_file, track_file, dict_file)
    
    # Étape 3: Homographie (standard ou adaptative)
    if use_adaptive:
        # Copier le fichier dict_file vers adaptive_dict_file pour préserver l'original
        if not os.path.exists(adaptive_dict_file) and os.path.exists(dict_file):
            print("Préparation du fichier pour homographie adaptative...")
            data_dict = np.load(dict_file, allow_pickle=True).item()
            np.save(adaptive_dict_file, data_dict)
        
        # Utiliser la méthode adaptative
        print("Application de l'homographie adaptative...")
        on_pitch_adaptive(adaptive_dict_file, homog_file)
    else:
        # Utiliser la méthode standard
        print("Application de l'homographie standard...")
        on_pitch(dict_file, homog_file)
    
    # Étape 4: Filtrage des joueurs hors terrain (si besoin)
    if start_from_scratch:
        print("Filtrage des joueurs hors terrain...")
        track_in_pitch(working_dict_file)
    
    # Étape 5: Classification des équipes (si activée)
    if do_team_classif and start_from_scratch:
        print("Classification des équipes...")
        # Importer ici pour éviter les conflits
        from team import TeamClassifier, get_crops, create_batches
        import supervision as sv
        import time
        # Code pour la classification des équipes...
        # to initialze classifier we take a strict definition of pitch
        # in order to exclude coaches or public
        device = 'cuda'

        track_dict = np.load(working_dict_file, allow_pickle=True).item()

        bboxes_ = track_dict['bboxes']
        inframe_ = bboxes_[:,0].astype(np.int16)
        boxes_ = bboxes_[:,1:5]
        track_ids_ = track_dict['track_ids']
        unique_track_ids = np.unique(track_ids_)
        # Utiliser les coordonnées adaptatives si disponibles
        xy_ = track_dict['xy_adaptive'] if 'xy_adaptive' in track_dict else track_dict['xy']
        track_ids_ = track_dict['track_ids']



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
    
    # Étape 6: Correction HMM (si activée)
    if do_HMMmissings and start_from_scratch:
        print("Application de la correction HMM...")
        track_dict = np.load(working_dict_file, allow_pickle=True).item()
        
        if 'to_keep' in track_dict:
            to_keep = np.where(track_dict['to_keep'])[0]
            bboxes_ = track_dict['bboxes'][to_keep]
            boxes_ = bboxes_[:,1:5]
            inframe = bboxes_[:,0].astype(np.int16)
            xy = track_dict['xy_adaptive'] if 'xy_adaptive' in track_dict else track_dict['xy'][to_keep]
            team_id_ = track_dict['team_id'][to_keep]
            track_ids = track_dict['track_ids'][to_keep]
            
            track_ids_hmm, team_id_hmm = HMM_missings(track_ids, inframe, team_id_, show_plot=True)
            
            track_dict['track_ids_hmm'] = track_dict['track_ids'].copy()
            track_dict['team_id_hmm'] = track_dict['team_id'].copy()
            
            track_dict['track_ids_hmm'][to_keep] = track_ids_hmm.copy()
            track_dict['team_id_hmm'][to_keep] = team_id_hmm.copy()
            np.save(working_dict_file, track_dict)
    
    # Étape 7: Chaînage des trajectoires (si activé)
    if do_chain_track and start_from_scratch:
        print("Application du chaînage des trajectoires...")
        ChainTrack(working_dict_file)
    
    # Étape 8: Optimisation par graphe (si activée)
    if do_graph_track and start_from_scratch:
        print("Application de l'optimisation par graphe...")
        start_chain = 100
        end_chain = 1300
        GraphTrack(working_dict_file, start_chain, end_chain, fps=30, show_tracks=False)
    
    # Étape 9: Visualisation
    print("Génération des visualisations...")
    method_name = "adaptive" if use_adaptive else "standard"
    
    # Visualisation du tracking
    track_kind = 'graph' if 'track_ids_graph' in np.load(working_dict_file, allow_pickle=True).item() else None
    file_name = f"visualisation_{method_name}_{track_kind}.mp4"
    file_path = folder_video_out / file_name
    plot_tracks(source_video_path=video_in, dict_file=working_dict_file,
               target_video_path= file_path, 
               track_kind=track_kind, start=0, end=-1)
    
    # Configuration de visualisation
    start_frame = VISUALIZATION_CONFIG["start_frame"] 
    end_frame = VISUALIZATION_CONFIG["end_frame"]
    
    # Visualisation radar standard
    file_name = f"radar_{method_name}.mp4"
    file_path = folder_video_out / file_name
    run_radar(video_in, working_dict_file, file_path, start=start_frame, end=end_frame)
    
    # Visualisation radar avec traces (uniquement pour les méthodes adaptatives)
    if use_adaptive:
        print("Génération du radar avec traces...")
        file_name = f"radar_{method_name}_traces.mp4"
        file_path = folder_video_out / file_name
        run_radar_adaptive(video_in, working_dict_file, file_path, start=start_frame, end=end_frame)
    
    # Visualisation des détections partielles
    from visualize_partial_detections import visualize_partial_detections
    file_name = f"partial_detections_{method_name}.mp4"
    file_path = folder_video_out / file_name
    visualize_partial_detections(
        video_in, 
        working_dict_file,
        file_path,
        start_frame=start_frame,
        end_frame=end_frame
    )
    
    # Analyse comparative des trajectoires (uniquement pour les méthodes adaptatives)
    if use_adaptive:
        print("Génération des visualisations de trajectoires comparatives...")
        add_trajectory_comparison_to_pipeline(working_dict_file)
    
    print(f"Pipeline exécuté avec succès, fichier résultant: {working_dict_file}")
    print(f"Visualisations générées: ")
    print(f"  - visualisation_{method_name}_{track_kind}.mp4")
    print(f"  - radar_{method_name}.mp4")
    if use_adaptive:
        print(f"  - radar_{method_name}_traces.mp4")
        print(f"  - trajectoires_comparaison/ (comparaisons des trajectoires)")
    print(f"  - partial_detections_{method_name}.mp4")
    
    return working_dict_file

def compare_methods():
    """
    Compare les méthodes standard et adaptative et génère des visualisations.
    """
    print("Comparaison des méthodes standard et adaptative...")
    
    # 1. Exécution avec méthode standard
    print("\n=== Exécution avec méthode standard ===")
    std_file = run_pipeline(use_adaptive=False, start_from_scratch=False)
    
    # 2. Exécution avec méthode adaptative
    print("\n=== Exécution avec méthode adaptative ===")
    adp_file = run_pipeline(use_adaptive=True, start_from_scratch=False)
    
    # 3. Visualiser les deux méthodes séparément
    print("\n=== Visualisations générées ===")
    print("  - partial_detections_standard.mp4")
    print("  - partial_detections_adaptive.mp4")
    print("  - radar_standard.mp4")
    print("  - radar_adaptive.mp4")
    print("  - radar_adaptive_traces.mp4")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Exécution du pipeline avec homographie adaptative')
    parser.add_argument('--adaptive', action='store_true', default=True,
                        help='Utiliser l\'homographie adaptative (défaut: True)')
    parser.add_argument('--standard', action='store_false', dest='adaptive',
                        help='Utiliser l\'homographie standard')
    parser.add_argument('--scratch', action='store_true',
                        help='Recalculer les étapes depuis le début')
    parser.add_argument('--compare', action='store_true',
                        help='Comparer les méthodes standard et adaptative')
    
    args = parser.parse_args()
    
    if args.compare:
        compare_methods()
    else:
        run_pipeline(use_adaptive=args.adaptive, start_from_scratch=args.scratch) 
