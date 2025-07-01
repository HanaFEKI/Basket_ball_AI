#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de visualisation des détections partielles.
Permet de vérifier si les détections partielles sont correctement identifiées.
"""

import numpy as np
import supervision as sv
import cv2
import argparse
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent.parent
player_detection_path = str(project_root / "Tracker" / "src" / "utils")
sys.path.insert(0, player_detection_path)
from partial_detection_utils import (
    get_adaptive_reference_points,
    detect_partial_boxes
)
from config import VISUALIZATION_CONFIG, FILE_CONFIG

def visualize_partial_detections(
    video_path: str,
    dict_file: str,
    output_path: str,
    start_frame: int = VISUALIZATION_CONFIG["start_frame"],
    end_frame: int = VISUALIZATION_CONFIG["end_frame"]
):
    """
    Crée une vidéo où les détections partielles sont mises en évidence.
    
    Args:
        video_path: Chemin vers la vidéo source
        dict_file: Fichier contenant les données de tracking
        output_path: Chemin pour la vidéo de sortie
        start_frame: Frame de début
        end_frame: Frame de fin (-1 pour aller jusqu'à la fin)
    """
    # Charger les données de tracking
    data_dict = np.load(dict_file, allow_pickle=True).item()
    bboxes = data_dict['bboxes']
    inframe = bboxes[:, 0].astype(np.int16)
    boxes = bboxes[:, 1:5]
    track_ids = data_dict['track_ids']
    
    # Vérifier si on doit recalculer les données
    if 'partial_detection_info' not in data_dict or 'partial_detections' not in data_dict['partial_detection_info']:
        print("Détection des occlusions partielles...")
        
        # Identifier les détections partielles
        partial_detections = detect_partial_boxes(boxes, track_ids, inframe)
        
        # Calculer les points de référence adaptatifs
        reference_points, partial_detections = get_adaptive_reference_points(
            boxes, track_ids, inframe
        )
        
        # Sauvegarder pour référence future
        if 'partial_detection_info' not in data_dict:
            data_dict['partial_detection_info'] = {}
        
        # Stocker les nouvelles informations
        data_dict['partial_detection_info']['partial_detections'] = partial_detections
        data_dict['partial_detection_info']['reference_points'] = reference_points
        
        np.save(dict_file, data_dict)
    
    # Sinon utiliser les informations existantes
    else:
        print("Utilisation des informations sur les détections partielles existantes")
        partial_detections = data_dict['partial_detection_info']['partial_detections']
        reference_points = data_dict['partial_detection_info']['reference_points']
    
    # Compter les détections partielles
    num_partial = np.sum(partial_detections)
    total_detections = len(boxes)
    print(f"Détections partielles: {num_partial}/{total_detections} ({num_partial/total_detections*100:.1f}%)")
    
    # Compter par ID de tracking
    unique_ids = np.unique(track_ids[track_ids > 0])
    partial_by_id = {}
    
    for track_id in unique_ids:
        mask = track_ids == track_id
        partial_count = np.sum(partial_detections[mask])
        total_count = np.sum(mask)
        partial_by_id[track_id] = (partial_count, total_count, partial_count/total_count*100)
        
    # Afficher les statistiques pour les 5 IDs avec le plus de détections partielles
    sorted_ids = sorted(partial_by_id.items(), key=lambda x: x[1][0], reverse=True)
    print("\nDétections partielles par ID (top 5):")
    for track_id, (partial_count, total_count, percentage) in sorted_ids[:5]:
        print(f"  ID {track_id}: {partial_count}/{total_count} ({percentage:.1f}%)")
    
    # Préparer la visualisation
    video_info = sv.VideoInfo.from_video_path(video_path)
    if end_frame == -1:
        end_frame = video_info.total_frames
        
    # Définir les annotateurs
    box_annotator_normal = sv.BoxAnnotator(
        color=sv.Color.BLUE,
        thickness=2
    )
    
    box_annotator_partial = sv.BoxAnnotator(
        color=sv.Color.RED,
        thickness=2
    )
    
    label_annotator = sv.LabelAnnotator(
        text_color=sv.Color.WHITE,
        text_padding=5,
        text_thickness=1,
        text_position=sv.Position.TOP_LEFT,
    )
    
    # Créer un générateur de frames vidéo
    source = sv.get_video_frames_generator(
        source_path=video_path, start=start_frame, end=end_frame)
    
    # Créer la vidéo de sortie
    with sv.VideoSink(output_path, video_info) as sink:
        for i_frame, frame in enumerate(source):
            i_frame += start_frame  # Ajuster l'index de frame
            
            # Trouver les détections dans cette frame
            in_frame_mask = inframe == i_frame
            current_boxes = boxes[in_frame_mask]
            current_track_ids = track_ids[in_frame_mask]
            current_partial = partial_detections[in_frame_mask]
            current_ref_points = reference_points[in_frame_mask]
            
            if len(current_boxes) == 0:
                sink.write_frame(frame)
                continue
            
            # Séparer les détections normales et partielles
            normal_boxes = current_boxes[~current_partial]
            partial_boxes = current_boxes[current_partial]
            
            normal_track_ids = current_track_ids[~current_partial]
            partial_track_ids = current_track_ids[current_partial]
            
            # Créer les annotations
            annotated_frame = frame.copy()
            
            # Annoter les détections normales
            if len(normal_boxes) > 0:
                normal_detections = sv.Detections(
                    xyxy=normal_boxes,
                    class_id=np.zeros(len(normal_boxes), dtype=int)
                )
                normal_labels = [f"{track_id}" for track_id in normal_track_ids]
                annotated_frame = box_annotator_normal.annotate(annotated_frame, normal_detections)
                annotated_frame = label_annotator.annotate(annotated_frame, normal_detections, labels=normal_labels)
            
            # Annoter les détections partielles
            if len(partial_boxes) > 0:
                partial_detections_obj = sv.Detections(
                    xyxy=partial_boxes,
                    class_id=np.ones(len(partial_boxes), dtype=int)
                )
                
                # Ajouter l'astérisque pour les détections partielles
                partial_labels = [f"{id}*" for id in partial_track_ids]
                
                annotated_frame = box_annotator_partial.annotate(annotated_frame, partial_detections_obj)
                annotated_frame = label_annotator.annotate(annotated_frame, partial_detections_obj, labels=partial_labels)
            
            # Ajouter des informations supplémentaires
            frame_partial_count = np.sum(current_partial)
            frame_total_count = len(current_partial)
            info_text = f"Frame: {i_frame} | Partielles: {frame_partial_count}/{frame_total_count}"
            cv2.putText(annotated_frame, info_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Visualiser les points de référence adaptatifs
            for i, (is_partial, box) in enumerate(zip(current_partial, current_boxes)):
                # Afficher le point de référence 
                x, y = current_ref_points[i]
                
                # Couleur du cercle selon le type de détection
                color = (0, 0, 255) if is_partial else (255, 255, 0)  # Rouge pour partiel, jaune pour normal
                
                # Dessiner un cercle à la position du point de référence
                cv2.circle(annotated_frame, (int(x), int(y)), 5, color, -1)
            
            sink.write_frame(annotated_frame)
    
    print(f"Vidéo de visualisation des détections partielles créée: {output_path}")
    return partial_by_id

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualisation des détections partielles')
    parser.add_argument('--video', type=str, default=FILE_CONFIG["video_in"],
                        help='Chemin vers la vidéo source')
    parser.add_argument('--dict_file', type=str, default=FILE_CONFIG["adaptive_dict_file"],
                        help='Fichier contenant les données de tracking')
    parser.add_argument('--output', type=str, default='partial_detections.mp4',
                        help='Chemin pour la vidéo de sortie')
    parser.add_argument('--start', type=int, default=VISUALIZATION_CONFIG["start_frame"],
                        help='Frame de début')
    parser.add_argument('--end', type=int, default=VISUALIZATION_CONFIG["end_frame"],
                        help='Frame de fin')
    
    args = parser.parse_args()
    
    visualize_partial_detections(
        args.video,
        args.dict_file,
        args.output,
        args.start,
        args.end
    ) 