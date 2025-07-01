#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 15:49:57 2025

@author: fenaux
"""
import os
import numpy as np
import supervision as sv
import cv2


COLORS = ['#FF1493', '#00BFFF', '#FF6347', '#FFD700']

# Annotateurs pour les boîtes normales et partielles
BOX_ANNOTATOR_NORMAL = sv.BoxAnnotator(
    color=sv.Color.BLUE,
    thickness=2
)

BOX_ANNOTATOR_PARTIAL = sv.BoxAnnotator(
    color=sv.Color.RED,
    thickness=2
)

ELLIPSE_ANNOTATOR = sv.EllipseAnnotator(
    thickness=2
)

ELLIPSE_LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=sv.ColorPalette.from_hex(COLORS),
    text_color=sv.Color.from_hex('#FFFFFF'),
    text_padding=5,
    text_thickness=1,
    text_position=sv.Position.BOTTOM_CENTER,
)

def plot_tracks(source_video_path: str, dict_file: str,
              target_video_path: str,
              track_kind: str = 'base',
              start: int = 0, end: int = -1,
              show_partial_detections: bool = True):
    """
    après la détection des joueurs, modification pour ne rechercher les équipes que si
    une nouvelle trace est apparue

    Args:
        source_video_path: Chemin vers la vidéo source
        dict_file: Fichier contenant les données de tracking
        target_video_path: Chemin pour la vidéo de sortie
        track_kind: Type de tracking à utiliser ('base', 'hmm', 'chain', 'graph')
        start: Frame de début
        end: Frame de fin (-1 pour aller jusqu'à la fin)
        show_partial_detections: Si True, affiche les détections partielles en rouge
    """
    data_dict = np.load(dict_file, allow_pickle=True).item()
    bboxes_ = data_dict['bboxes']
    inframe = bboxes_[:,0].astype(np.int16)
    #frames = np.unique(inframe)
    bboxes = bboxes_[:,1:5]

    bh_bboxes_ = np.load(os.getcwd()+"/Boxes_Detection/src/data/annotations/ball_handler.npy")
    bh_inframe = bh_bboxes_[:,0].astype(np.int16)
    bh_bboxes = bh_bboxes_[:,1:5]

    if track_kind == 'hmm': track_id = data_dict['track_ids_hmm'].astype(np.int16)
    elif track_kind == 'chain': track_id = data_dict['track_ids_chain'].astype(np.int16)
    elif track_kind == 'graph': track_id = data_dict['track_ids_graph'].astype(np.int16)
    else: track_id = data_dict['track_ids'].astype(np.int16)
   
    if 'to_keep' in data_dict:
        to_keep = data_dict['to_keep']
        inframe = inframe[to_keep]
        bboxes = bboxes[to_keep]
        track_id = track_id[to_keep]
    
    # Vérifier si nous avons des informations sur les détections partielles
    has_partial_info = 'partial_detection_info' in data_dict and show_partial_detections
    partial_detections = None
    reference_points = None
    
    if has_partial_info:
        print("Affichage des détections partielles en rouge")
        partial_info = data_dict['partial_detection_info']
        if 'partial_detections' in partial_info:
            if 'to_keep' in data_dict:
                partial_detections = partial_info['partial_detections'][to_keep]
            else:
                partial_detections = partial_info['partial_detections']
        elif 'reference_points' in partial_info:
            # Si nous n'avons pas directement le masque mais que nous avons les points de référence
            if 'to_keep' in data_dict:
                reference_points = partial_info['reference_points'][to_keep]
            else:
                reference_points = partial_info['reference_points']
            partial_detections = np.zeros_like(track_id, dtype=bool)
            
            # Reconstruire le masque des détections partielles
            for i, (tid, box) in enumerate(zip(track_id, bboxes)):
                if tid <= 0:
                    continue
                
                center_x = (box[0] + box[2]) / 2
                bottom_y = box[3]
                
                if np.linalg.norm(reference_points[i] - np.array([center_x, bottom_y])) > 1.0:
                    partial_detections[i] = True
        else:
            print("Aucune information sur les détections partielles trouvée")
            has_partial_info = False

    video_info = sv.VideoInfo.from_video_path(source_video_path)
    if end == -1:
        end = video_info.total_frames
    source = sv.get_video_frames_generator(
        source_path=source_video_path, start=start, end=end)
    
    with sv.VideoSink(target_video_path, video_info) as sink:
        for i_frame, frame in enumerate(source):
            i_frame += start  # Ajuster l'index de frame
            
            in_i_frame = np.where(inframe == i_frame)[0]
            bh_in_i_frame = np.where(bh_inframe == i_frame)[0]

            annotated_frame = frame.copy()
            bh_detections = sv.Detections(
                xyxy=bh_bboxes[bh_in_i_frame],
                class_id = np.zeros(len(bh_bboxes[bh_in_i_frame]), dtype=int)
            )
            annotated_frame = BOX_ANNOTATOR_NORMAL.annotate(annotated_frame, bh_detections)
            annotated_frame = ELLIPSE_LABEL_ANNOTATOR.annotate(annotated_frame, bh_detections, labels=np.array(["ball_handler" for _ in range(len(bh_bboxes[bh_in_i_frame]))]))

            if len(in_i_frame) == 0:
                sink.write_frame(frame)
                continue

            boxes_in_frame = bboxes[in_i_frame]
            track_ids_in_frame = track_id[in_i_frame].astype(np.int16)
            
            annotated_frame = frame.copy()
            
            # Si nous avons des informations sur les détections partielles
            if has_partial_info:
                partial_in_frame = partial_detections[in_i_frame]
                
                # Séparer les boîtes normales et partielles
                normal_boxes = boxes_in_frame[~partial_in_frame]
                normal_track_ids = track_ids_in_frame[~partial_in_frame]
                
                partial_boxes = boxes_in_frame[partial_in_frame]
                partial_track_ids = track_ids_in_frame[partial_in_frame]
                
                # Annoter les boîtes normales
                if len(normal_boxes) > 0:
                    normal_detections = sv.Detections(
                        xyxy=normal_boxes,
                        class_id=np.zeros(len(normal_boxes), dtype=int)
                    )
                    normal_labels = [str(tid) for tid in normal_track_ids]
                    annotated_frame = BOX_ANNOTATOR_NORMAL.annotate(annotated_frame, normal_detections)
                    annotated_frame = ELLIPSE_LABEL_ANNOTATOR.annotate(annotated_frame, normal_detections, labels=normal_labels)
                
                # Annoter les boîtes partielles
                if len(partial_boxes) > 0:
                    partial_detections_obj = sv.Detections(
                        xyxy=partial_boxes,
                        class_id=np.ones(len(partial_boxes), dtype=int)
                    )
                    partial_labels = [f"{tid}*" for tid in partial_track_ids]
                    annotated_frame = BOX_ANNOTATOR_PARTIAL.annotate(annotated_frame, partial_detections_obj)
                    annotated_frame = ELLIPSE_LABEL_ANNOTATOR.annotate(annotated_frame, partial_detections_obj, labels=partial_labels)
                
                # Visualiser les points de référence si disponibles
                if reference_points is not None:
                    ref_points_in_frame = reference_points[in_i_frame]
                    for i, is_partial in enumerate(partial_in_frame):
                        if is_partial:
                            x, y = ref_points_in_frame[i]
                            # Dessiner un cercle jaune à la position du point de référence
                            cv2.circle(annotated_frame, (int(x), int(y)), 5, (0, 255, 255), -1)
                
                # Ajouter un compteur de détections partielles
                partial_count = np.sum(partial_in_frame)
                total_count = len(partial_in_frame)
                info_text = f"Partielles: {partial_count}/{total_count}"
                cv2.putText(annotated_frame, info_text, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                # Méthode originale si pas d'info sur les détections partielles
                detections = sv.Detections(
                    xyxy=boxes_in_frame,
                    class_id=track_ids_in_frame
                )
                labels = [str(tid) for tid in track_ids_in_frame]
                annotated_frame = ELLIPSE_ANNOTATOR.annotate(annotated_frame, detections)
                annotated_frame = ELLIPSE_LABEL_ANNOTATOR.annotate(annotated_frame, detections, labels=labels)
            
            # Ajouter le numéro de frame
            cv2.putText(annotated_frame, str(i_frame), (100, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0))

            sink.write_frame(annotated_frame)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualiser les trajectoires des joueurs')
    parser.add_argument('--video', type=str, default='basket_short.mp4',
                        help='Chemin vers la vidéo source')
    parser.add_argument('--dict_file', type=str, default='clip_dict_adaptive.npy',
                        help='Fichier contenant les données de tracking')
    parser.add_argument('--output', type=str, default='tracks_visualization.mp4',
                        help='Chemin pour la vidéo de sortie')
    parser.add_argument('--track_kind', type=str, default='base',
                        choices=['base', 'hmm', 'chain', 'graph'],
                        help='Type de tracking à utiliser')
    parser.add_argument('--start', type=int, default=0,
                        help='Frame de début')
    parser.add_argument('--end', type=int, default=-1,
                        help='Frame de fin')
    parser.add_argument('--show_partial', action='store_true',
                        help='Afficher les détections partielles en rouge')
    
    args = parser.parse_args()
    
    plot_tracks(
        args.video,
        args.dict_file,
        args.output,
        args.track_kind,
        args.start,
        args.end,
        args.show_partial
    )
