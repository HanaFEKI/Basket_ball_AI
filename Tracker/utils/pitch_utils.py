#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 20:17:55 2025

@author: fenaux
"""

from typing import Optional

import os
import cv2
import supervision as sv
import numpy as np

def draw_pitch(
    background_color: sv.Color = sv.Color(34, 139, 34),
    line_color: sv.Color = sv.Color.WHITE,
    padding: int = 50,
    line_thickness: int = 4,
    scale: float = 40
) -> np.ndarray:       

    vertices = [(0,0), (0,15), (14,0), (14,15), (28,0), (28,15),
                (0, 7.5 - 2.45), (0, 7.5 + 2.45),
                (5.8, 7.5 - 2.45), (5.8, 7.5 + 2.45),
                (28, 7.5 - 2.45), (28, 7.5 + 2.45),
                (28 - 5.8, 7.5 - 2.45), (28 - 5.8, 7.5 + 2.45),
                (14, 7.5 - 1.8), (14, 7.5 + 1.8)]
    
    centers = [(5.8, 7.5), (28 - 5.8, 7.5), (14, 7.5)]
                     
    edges =[(0, 1), (0, 4), (1, 5), (4, 5),
        (6, 8), (7, 9), (8, 9),
        (10, 12), (11, 13), (12, 13),
        (2,14), (3, 15)]
    
    angles = [180, 0]
    
    
    width = 15
    length = 28
    radius = 1.8
    
    scaled_width = int(width * scale)
    scaled_length = int(length * scale)
    scaled_radius = int(radius * scale)
    
    pitch_image = np.ones(
        (scaled_width + 2 * padding,
         scaled_length + 2 * padding, 3),
        dtype=np.uint8
    ) * np.array(background_color.as_bgr(), dtype=np.uint8)
    
    for start, end in edges:
        point1 = (int(vertices[start][0] * scale) + padding,
                  int(vertices[start][1] * scale) + padding)
        point2 = (int(vertices[end][0] * scale) + padding,
                  int(vertices[end][1] * scale) + padding)
        cv2.line(
            img=pitch_image,
            pt1=point1,
            pt2=point2,
            color=line_color.as_bgr(),
            thickness=line_thickness
        )
            
    for center, angle in zip(centers[:2], angles):
        pt = (int(center[0] * scale) + padding,
                  int(center[1] * scale) + padding)
        axes = (scaled_radius, scaled_radius)
        cv2.ellipse(img=pitch_image,
                    center=pt, axes=axes,
                    angle=angle, startAngle=90, endAngle=270,
                    color=line_color.as_bgr(), thickness=line_thickness)
        
    center = centers[2]
    pt = (int(center[0] * scale) + padding,
              int(center[1] * scale) + padding)
    
    cv2.circle(
        img=pitch_image,
        center=pt,
        radius=scaled_radius,
        color=line_color.as_bgr(),
        thickness=line_thickness
    )
        
    return pitch_image

def draw_points_on_pitch(
    xy: np.ndarray,
    face_color: sv.Color = sv.Color.RED,
    edge_color: sv.Color = sv.Color.BLACK,
    radius: int = 10,
    thickness: int = 2,
    padding: int = 50,
    scale: float = 40,
    pitch: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Draws points on a soccer pitch.

    Args:
        config (SoccerPitchConfiguration): Configuration object containing the
            dimensions and layout of the pitch.
        xy (np.ndarray): Array of points to be drawn, with each point represented by
            its (x, y) coordinates.
        face_color (sv.Color, optional): Color of the point faces.
            Defaults to sv.Color.RED.
        edge_color (sv.Color, optional): Color of the point edges.
            Defaults to sv.Color.BLACK.
        radius (int, optional): Radius of the points in pixels.
            Defaults to 10.
        thickness (int, optional): Thickness of the point edges in pixels.
            Defaults to 2.
        padding (int, optional): Padding around the pitch in pixels.
            Defaults to 50.
        scale (float, optional): Scaling factor for the pitch dimensions.
            Defaults to 0.1.
        pitch (Optional[np.ndarray], optional): Existing pitch image to draw points on.
            If None, a new pitch will be created. Defaults to None.

    Returns:
        np.ndarray: Image of the soccer pitch with points drawn on it.
    """
    if pitch is None:
        pitch = draw_pitch(
            padding=padding,
            scale=scale
        )
    if len(xy) != 0:
        for point in xy:
            scaled_point = (
                int(point[0] * scale) + padding,
                int(point[1] * scale) + padding
            )
            cv2.circle(
                img=pitch,
                center=scaled_point,
                radius=radius,
                color=face_color.as_bgr(),
                thickness=-1
            )
            cv2.circle(
                img=pitch,
                center=scaled_point,
                radius=radius,
                color=edge_color.as_bgr(),
                thickness=thickness
            )

    return np.flipud(pitch)

BOX_ANNOTATOR = sv.BoxAnnotator(
    color=sv.Color.BLUE,
    thickness=2
)
ELLIPSE_ANNOTATOR = sv.EllipseAnnotator(
    color=sv.Color.YELLOW,
    thickness=2
)
COLORS = ['#FF1493', '#00BFFF', '#FF6347', '#FFD700']

ELLIPSE_LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=sv.ColorPalette.from_hex(COLORS),
    text_color=sv.Color.from_hex('#FFFFFF'),
    text_padding=5,
    text_thickness=1,
    text_position=sv.Position.BOTTOM_CENTER,
)

TOP_ELLIPSE_LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=sv.ColorPalette.from_hex(COLORS),
    text_color=sv.Color.from_hex('#FFFFFF'),
    text_padding=5,
    text_thickness=1,
    text_position=sv.Position.TOP_CENTER,
)

CIRCLE_ANNOTATOR = sv.CircleAnnotator()

def run_radar(source_video_path: str, dict_file: str,
              target_video_path: str,
              start: int = 0, end: int = -1):# -> Iterator[np.ndarray]:
    """
    après la détection des joueurs, modification pour ne rechercher les équipes que si
    une nouvelle trace est apparue

    """
    data_dict = np.load(dict_file, allow_pickle=True).item()
    bboxes_ = data_dict['bboxes']
    inframe = bboxes_[:,0].astype(np.int16)
    #frames = np.unique(inframe)
    bboxes = bboxes_[:,1:5]
    in_pitch = data_dict['in_pitch'].astype(np.bool_)
    track_id = data_dict['track_ids'].astype(np.int16)
    players = data_dict['xy'].astype(np.int16)
    
    inframe = inframe[in_pitch]
    bboxes = bboxes[in_pitch]
    track_id = track_id[in_pitch]
    players = players[in_pitch]
    
    in_track = (track_id > -1)
    inframe = inframe[in_track]
    bboxes = bboxes[in_track]
    track_id = track_id[in_track]
    players = players[in_track]
    
    video_info = sv.VideoInfo.from_video_path(source_video_path)
    if end == -1:
        end = video_info.total_frames
    source = sv.get_video_frames_generator(
        source_path=source_video_path, start=start, end=end)
    
   
    with sv.VideoSink(target_video_path, video_info) as sink:
        for i_frame, frame in enumerate(source):
            in_i_frame = np.where(inframe== i_frame)[0]
            players_in_frame = players[in_i_frame]
            # boxes_in_frame = bboxes[in_i_frame]
            
            """
            detections_in = sv.Detections(xyxy = boxes_in_frame[are_in_pitch_frame],
                                          class_id=np.zeros(are_in_pitch_frame.sum()))
            detections_out = sv.Detections(xyxy = boxes_in_frame[are_out_pitch_frame],
                                           class_id=np.zeros(are_out_pitch_frame.sum()))
            
            annotated_frame = frame.copy()
            annotated_frame = ELLIPSE_ANNOTATOR.annotate(
                annotated_frame, detections_in)
            annotated_frame = BOX_ANNOTATOR.annotate(
                annotated_frame, detections_out)"""
    
            radar = draw_points_on_pitch(players_in_frame)
            
            h, w, _ = frame.shape
            radar = sv.resize_image(radar, (w // 2, h // 2))
            radar_h, radar_w, _ = radar.shape
            rect = sv.Rect(
                x=w // 2 - radar_w // 2,
                y=h - radar_h,
                width=radar_w,
                height=radar_h
            )
            annotated_frame = frame.copy()
            annotated_frame = sv.draw_image(annotated_frame, radar, opacity=0.5, rect=rect)
            sink.write_frame(annotated_frame)
            
def traj_radar(source_video_path: str, traj_file: str,
              target_video_path: str,
              start: int = 0, end: int = -1):# -> Iterator[np.ndarray]:
    
    """data_dict = np.load(dict_file, allow_pickle=True).item()
    bboxes_ = data_dict['bboxes']
    inframe = bboxes_[:,0].astype(np.int16) 
    track_id = data_dict['track_ids'].astype(np.int16)"""
    
    traj_dict = np.load(traj_file, allow_pickle=True).item()
    # traj_ids = traj_dict['ids'] # list of track ids
    fxys = traj_dict['xy'] # list of arrays coordinates
    
    video_info = sv.VideoInfo.from_video_path(source_video_path)
    if end == -1:
        end = video_info.total_frames
    source = sv.get_video_frames_generator(
        source_path=source_video_path, start=start, end=end)
    
   
    with sv.VideoSink(target_video_path, video_info) as sink:
        for i_frame, frame in enumerate(source):
            players_in_frame = []
            for fxy in fxys:
                coords_in_frame = fxy[np.where(fxy[:,0] == i_frame)[0],1:]
                if coords_in_frame.size > 0:  players_in_frame.append(coords_in_frame)
               
            players_in_frame = np.array(players_in_frame).reshape(-1,2)
            
            """
            detections_in = sv.Detections(xyxy = boxes_in_frame[are_in_pitch_frame],
                                          class_id=np.zeros(are_in_pitch_frame.sum()))
            detections_out = sv.Detections(xyxy = boxes_in_frame[are_out_pitch_frame],
                                           class_id=np.zeros(are_out_pitch_frame.sum()))
            
            annotated_frame = frame.copy()
            annotated_frame = ELLIPSE_ANNOTATOR.annotate(
                annotated_frame, detections_in)
            annotated_frame = BOX_ANNOTATOR.annotate(
                annotated_frame, detections_out)"""
    
            radar = draw_points_on_pitch(players_in_frame)
            
            h, w, _ = frame.shape
            radar = sv.resize_image(radar, (w // 2, h // 2))
            radar_h, radar_w, _ = radar.shape
            rect = sv.Rect(
                x=w // 2 - radar_w // 2,
                y=h - radar_h,
                width=radar_w,
                height=radar_h
            )
            annotated_frame = frame.copy()
            annotated_frame = sv.draw_image(annotated_frame, radar, opacity=0.5, rect=rect)
            sink.write_frame(annotated_frame)

def run_radar_adaptive(source_video_path: str, dict_file: str,
              target_video_path: str,
              start: int = 0, end: int = -1):# -> Iterator[np.ndarray]:
    """
    Version du radar utilisant les coordonnées adaptatives pour une meilleure
    continuité des trajectoires pendant les occlusions partielles.
    """
    data_dict = np.load(dict_file, allow_pickle=True).item()
    bboxes_ = data_dict['bboxes']
    inframe = bboxes_[:,0].astype(np.int16)
    bboxes = bboxes_[:,1:5]
    in_pitch = data_dict['in_pitch'].astype(np.bool_)
    track_id = data_dict['track_ids'].astype(np.int16)
    
    # Utiliser les coordonnées adaptatives si disponibles
    if 'xy_adaptive' in data_dict:
        print("Utilisation des coordonnées adaptatives pour le radar")
        players = data_dict['xy_adaptive'].astype(np.int16)
    else:
        print("Coordonnées adaptatives non disponibles, utilisation des coordonnées standard")
        players = data_dict['xy'].astype(np.int16)
    
    inframe = inframe[in_pitch]
    bboxes = bboxes[in_pitch]
    track_id = track_id[in_pitch]
    players = players[in_pitch]
    
    in_track = (track_id > -1)
    inframe = inframe[in_track]
    bboxes = bboxes[in_track]
    track_id = track_id[in_track]
    players = players[in_track]
    
    bh_bboxes_ = np.load(os.getcwd()+"/Boxes_Detection/src/data/annotations/ball_handler.npy")
    bh_inframe = bh_bboxes_[:,0].astype(np.int16)
    bh_bboxes = bh_bboxes_[:,1:5]

    ball_bboxes_ = np.load(os.getcwd()+"/Boxes_Detection/src/data/annotations/ball.npy")
    ball_inframe = ball_bboxes_[:,0].astype(np.int16)
    ball_bboxes = ball_bboxes_[:,1:5]


    video_info = sv.VideoInfo.from_video_path(source_video_path)
    if end == -1:
        end = video_info.total_frames
    source = sv.get_video_frames_generator(
        source_path=source_video_path, start=start, end=end)
    
    # Dictionnaire pour stocker les positions précédentes par ID
    # prev_positions = {}
    
    # Paramètres pour les traces
    trace_length = 30  # Nombre de frames pour conserver la trace
    team_colors = {
        0: sv.Color.BLUE,  # Équipe 1
        1: sv.Color.RED,   # Équipe 2
        2: sv.Color.GREEN  # Arbitre ou autre
    }
    
    # Historique des positions
    position_history = {track_id: [] for track_id in np.unique(track_id)}
    
    with sv.VideoSink(target_video_path, video_info) as sink:
        for i_frame, frame in enumerate(source):
            in_i_frame = np.where(inframe == i_frame)[0]
            bh_in_i_frame = np.where(bh_inframe == i_frame)[0]
            ball_in_i_frame = np.where(ball_inframe == i_frame)[0]
            players_in_frame = players[in_i_frame]
            track_ids_in_frame = track_id[in_i_frame]
            boxes_in_frame = bboxes[in_i_frame]
            
            # Associer chaque position à son ID et mettre à jour l'historique
            for pos, tid in zip(players_in_frame, track_ids_in_frame):
                if tid > 0:  # Ignorer les IDs négatifs
                    position_history[tid].append((i_frame, pos))
                    # Garder seulement les N dernières positions
                    if len(position_history[tid]) > trace_length:
                        position_history[tid].pop(0)
            
            # Radar de base avec les positions actuelles
            radar = draw_pitch()
            
            # Dessiner les traces pour chaque joueur
            for tid in position_history:
                history = position_history[tid]
                if not history:
                    continue
                
                # Déterminer la couleur (utiliser l'ID modulo 3 pour simplifier)
                color = team_colors.get(tid % 3, sv.Color.YELLOW)
                
                # Dessiner les points de l'historique avec opacité croissante
                points = [pos for _, pos in history if _ <= i_frame]
                if len(points) > 1:
                    # Dessiner les lignes entre les points
                    points_array = np.array(points)
                    for j in range(1, len(points_array)):
                        alpha = 0.3 + 0.7 * (j / len(points_array))  # Opacité croissante
                        pt1 = (int(points_array[j-1][0] * 40) + 50, int(points_array[j-1][1] * 40) + 50)
                        pt2 = (int(points_array[j][0] * 40) + 50, int(points_array[j][1] * 40) + 50)
                        cv2.line(radar, pt1, pt2, (*color.as_bgr(), int(255 * alpha)), 2)
                
                # Dessiner le point actuel
                if history[-1][0] <= i_frame:
                    current_pos = history[-1][1]
                    scaled_point = (
                        int(current_pos[0] * 40) + 50,
                        int(current_pos[1] * 40) + 50
                    )
                    cv2.circle(
                        img=radar,
                        center=scaled_point,
                        radius=8,
                        color=color.as_bgr(),
                        thickness=-1
                    )
                    cv2.circle(
                        img=radar,
                        center=scaled_point,
                        radius=8,
                        color=sv.Color.BLACK.as_bgr(),
                        thickness=2
                    )
                    
                    # Ajouter le numéro du joueur
                    cv2.putText(
                        img=radar,
                        text=str(tid),
                        org=(scaled_point[0] - 4, scaled_point[1] + 4),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.4,
                        color=sv.Color.WHITE.as_bgr(),
                        thickness=2
                    )
            
            # Appliquer le flip vertical pour avoir la bonne orientation
            radar = np.flipud(radar)
            
            # Redimensionner et intégrer le radar dans la frame
            h, w, _ = frame.shape
            radar = sv.resize_image(radar, (w // 2, h // 2))
            radar_h, radar_w, _ = radar.shape
            rect = sv.Rect(
                x=w // 2 - radar_w // 2,
                y=h - radar_h,
                width=radar_w,
                height=radar_h
            )
            annotated_frame = frame.copy()

            detections = sv.Detections(
                xyxy=boxes_in_frame,
                class_id=track_ids_in_frame
            )
            labels = [str(tid) for tid in track_ids_in_frame]
            annotated_frame = ELLIPSE_ANNOTATOR.annotate(annotated_frame, detections)
            annotated_frame = ELLIPSE_LABEL_ANNOTATOR.annotate(annotated_frame, detections, labels=labels)

            bh_detections = sv.Detections(
                xyxy=bh_bboxes[bh_in_i_frame],
                class_id = np.zeros(len(bh_bboxes[bh_in_i_frame]), dtype=int)
            )
            annotated_frame = BOX_ANNOTATOR.annotate(annotated_frame, bh_detections)
            annotated_frame = TOP_ELLIPSE_LABEL_ANNOTATOR.annotate(annotated_frame, bh_detections, labels=np.array(["ball_handler" for _ in range(len(bh_bboxes[bh_in_i_frame]))]))

            ball_detections = sv.Detections(
                xyxy=ball_bboxes[ball_in_i_frame],
                class_id = np.zeros(len(ball_bboxes[ball_in_i_frame]), dtype=int)
            )
            annotated_frame = CIRCLE_ANNOTATOR.annotate(annotated_frame, ball_detections)


            annotated_frame = sv.draw_image(annotated_frame, radar, opacity=0.7, rect=rect)
            
            # Ajouter le numéro de frame
            cv2.putText(
                img=annotated_frame,
                text=f"Frame: {i_frame}",
                org=(20, 30),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1.0,
                color=(0, 255, 0),
                thickness=2
            )
            
            sink.write_frame(annotated_frame)
    
    print(f"Radar adaptatif généré: {target_video_path}")
