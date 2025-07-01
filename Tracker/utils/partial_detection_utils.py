#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilitaires pour la détection et la gestion des détections partielles lors de tracking.
Ces fonctions visent à améliorer la continuité des trajectoires pendant les occlusions.

@author: Cedric
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Tuple, Dict
from config import PARTIAL_DETECTION_CONFIG

# Extraire les paramètres de configuration
height_drop_threshold = PARTIAL_DETECTION_CONFIG["height_drop_threshold"]
window_size = PARTIAL_DETECTION_CONFIG["window_size"]
aspect_ratio_change_threshold = PARTIAL_DETECTION_CONFIG["aspect_ratio_change_threshold"]
width_stability_threshold = PARTIAL_DETECTION_CONFIG["width_stability_threshold"]

def estimate_full_height(
    boxes: np.ndarray,
    track_ids: np.ndarray,
    inframe: np.ndarray,
    partial_detections: np.ndarray,
    window_size: int = window_size
) -> Dict[int, float]:
    """
    Estime la hauteur complète attendue pour chaque ID de tracking.
    
    Args:
        boxes: Tableau de boîtes englobantes [x1, y1, x2, y2]
        track_ids: Identifiants des trajectoires correspondant aux boîtes
        inframe: Numéros de frame pour chaque boîte
        partial_detections: Masque indiquant les détections partielles
        window_size: Taille de la fenêtre pour le calcul des statistiques
    
    Returns:
        Un dictionnaire associant chaque ID de trajectoire à sa hauteur estimée
    """
    heights = boxes[:, 3] - boxes[:, 1]
    estimated_heights = {}
    
    unique_ids = np.unique(track_ids[track_ids > 0])
    for track_id in unique_ids:
        indices = np.where(track_ids == track_id)[0]
        
        # Exclure les détections partielles pour le calcul de la hauteur de référence
        valid_indices = indices[~partial_detections[indices]]
        
        if len(valid_indices) < 3:
            # Pas assez de détections valides, utiliser toutes les détections
            valid_heights = heights[indices]
        else:
            valid_heights = heights[valid_indices]
        
        # Utiliser une hauteur robuste (75e percentile) pour éviter les valeurs aberrantes
        estimated_heights[track_id] = np.percentile(valid_heights, 75)
    
    return estimated_heights

def detect_partial_boxes(
    boxes: np.ndarray, 
    track_ids: np.ndarray,
    inframe: np.ndarray,
    window_size: int = window_size,
    height_drop_threshold: float = height_drop_threshold,
    aspect_ratio_change_threshold: float = aspect_ratio_change_threshold,
    visualize: bool = False
) -> np.ndarray:
    """
    Détecte les boîtes englobantes correspondant à des détections partielles
    en se basant uniquement sur l'historique des détections.
    
    Args:
        boxes: Tableau de boîtes englobantes [x1, y1, x2, y2]
        track_ids: Identifiants des trajectoires correspondant aux boîtes
        inframe: Numéros de frame pour chaque boîte
        window_size: Taille de la fenêtre pour le calcul des statistiques
        height_drop_threshold: Seuil de chute de hauteur pour considérer une détection comme partielle
        aspect_ratio_change_threshold: Seuil de changement du ratio hauteur/largeur
        visualize: Si True, génère des graphiques d'analyse pour chaque track_id
    
    Returns:
        Un masque booléen indiquant quelles boîtes sont des détections partielles
    """
    # Calculer la hauteur, largeur et ratio hauteur/largeur de chaque boîte
    heights = boxes[:, 3] - boxes[:, 1]
    widths = boxes[:, 2] - boxes[:, 0]
    aspect_ratios = heights / widths
    
    # Initialiser le masque de détections partielles
    partial_detections = np.zeros_like(track_ids, dtype=bool)
    
    # Créer un répertoire pour les visualisations si nécessaire
    vis_dir = "height_analysis"
    if visualize and not os.path.exists(vis_dir):
        os.makedirs(vis_dir)
    
    # Pour chaque ID de trajectoire unique
    unique_ids = np.unique(track_ids[track_ids > 0])
    for track_id in unique_ids:
        # Indices où cet ID apparaît
        indices = np.where(track_ids == track_id)[0]
        
        if len(indices) < 5:  # Ignorer les trajectoires trop courtes (besoin d'historique suffisant)
            continue
            
        # Trier les indices par numéro de frame
        sort_idx = np.argsort(inframe[indices])
        sorted_indices = indices[sort_idx]
        frames = inframe[sorted_indices]
        
        # Structures pour stocker les données d'analyse
        if visualize:
            track_frames = []
            track_heights = []
            medians = []
            p75_values = []
            is_partial = []
        
        # Calculer la médiane mobile des hauteurs précédentes pour chaque point
        for i in range(len(sorted_indices)):
            # Utiliser uniquement une fenêtre sur le passé
            past_window_size = min(i, window_size)
            
            if past_window_size < 3:  # Besoin d'un historique minimum
                continue
                
            # Indices pour le passé
            past_indices = sorted_indices[i-past_window_size:i]
            current_idx = sorted_indices[i]
            current_frame = inframe[current_idx]
            
            # Statistiques des hauteurs passées
            past_heights = heights[past_indices]
            past_height_p75 = np.percentile(past_heights, 75)  # 75e percentile
            past_height_median = np.median(past_heights)  # Médiane
            past_aspect_median = np.median(aspect_ratios[past_indices])
            
            # Hauteur et ratio actuels
            current_height = heights[current_idx]
            current_aspect = aspect_ratios[current_idx]
            current_width = widths[current_idx]
            
            # Conditions pour une détection partielle
            # 1. Chute significative de la hauteur par rapport à l'historique
            relative_height_drop = (past_height_p75 - current_height) / past_height_p75
            
            # 2. Changement significatif du ratio hauteur/largeur
            aspect_change = abs(current_aspect - past_aspect_median) / past_aspect_median
            
            # 3. Largeur relativement stable (pour éliminer les cas de rétrécissement global)
            past_widths = widths[past_indices]
            past_width_median = np.median(past_widths)
            width_stability = abs(current_width - past_width_median) / past_width_median
            
            # 4. Vérifier si la hauteur est trop petite comparée à l'historique
            is_significantly_shorter = relative_height_drop > height_drop_threshold
            
            # 5. Vérifier que ce n'est pas juste une boîte globalement plus petite
            has_abnormal_aspect = aspect_change > aspect_ratio_change_threshold
            has_stable_width = width_stability < width_stability_threshold
            
            # Une détection est partielle si elle est significativement plus courte et
            # a soit un ratio anormal, soit une largeur stable
            is_partial_detection = is_significantly_shorter and (has_abnormal_aspect or has_stable_width)
            if is_partial_detection:
                partial_detections[current_idx] = True
            
            # Stocker les données pour visualisation
            if visualize:
                track_frames.append(current_frame)
                track_heights.append(current_height)
                medians.append(past_height_median)
                p75_values.append(past_height_p75)
                is_partial.append(is_partial_detection)
        
        # Créer la visualisation pour ce track_id
        if visualize and len(track_frames) > 0:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})
            
            # Graphique principal: hauteurs, médiane et 75e percentile
            ax1.plot(track_frames, track_heights, 'k-', label='Hauteur actuelle')
            ax1.plot(track_frames, medians, 'b--', label='Médiane mobile')
            ax1.plot(track_frames, p75_values, 'r--', label='75e percentile')
            
            # Marquer les détections partielles
            partial_indices = [i for i, partial in enumerate(is_partial) if partial]
            if partial_indices:
                partial_frames = [track_frames[i] for i in partial_indices]
                partial_heights = [track_heights[i] for i in partial_indices]
                ax1.scatter(partial_frames, partial_heights, color='red', s=50, marker='o', label='Détection partielle')
            
            # Ajouter les légendes et titres
            ax1.set_title(f'Analyse des hauteurs pour le joueur {track_id}')
            ax1.set_xlabel('Numéro de frame')
            ax1.set_ylabel('Hauteur (pixels)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Graphique secondaire: écart entre 75e percentile et médiane
            differences = [p - m for p, m in zip(p75_values, medians)]
            ax2.plot(track_frames, differences, 'g-', label='Écart 75e percentile - médiane')
            ax2.axhline(y=np.mean(differences), color='r', linestyle='--', label=f'Écart moyen: {np.mean(differences):.2f} px')
            
            # Marquer les détections partielles
            if partial_indices:
                partial_diffs = [differences[i] for i in partial_indices]
                ax2.scatter(partial_frames, partial_diffs, color='red', s=30, marker='x')
            
            ax2.set_xlabel('Numéro de frame')
            ax2.set_ylabel('Écart (pixels)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Statistiques
            stats_text = (f"Statistiques:\n"
                          f"- Nombre de détections: {len(track_frames)}\n"
                          f"- Détections partielles: {sum(is_partial)} ({sum(is_partial)/len(track_frames)*100:.1f}%)\n"
                          f"- Écart moyen entre 75e percentile et médiane: {np.mean(differences):.2f} pixels\n"
                          f"- Écart max: {max(differences):.2f} pixels\n"
                          f"- Hauteur moyenne: {np.mean(track_heights):.2f} pixels")
            
            # Ajouter un encadré pour les statistiques
            ax1.text(0.02, 0.02, stats_text, transform=ax1.transAxes, verticalalignment='bottom',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), fontsize=10)
            
            plt.tight_layout()
            plt.savefig(f"{vis_dir}/track_id_{track_id}_height_analysis.png", dpi=150)
            plt.close(fig)
    
    return partial_detections

def calculate_reference_point(
    box: np.ndarray, 
    is_partial: bool,
    full_height: float = None
) -> Tuple[float, float]:
    """
    Calcule un point de référence adapté pour les boîtes partielles en utilisant
    la hauteur complète estimée.
    
    Args:
        box: Boîte englobante [x1, y1, x2, y2]
        is_partial: Indique si c'est une détection partielle
        full_height: Hauteur complète estimée
    
    Returns:
        Coordonnées (x, y) du point de référence pour l'homographie
    """
    x1, y1, x2, y2 = box
    center_x = (x1 + x2) / 2
    
    if not is_partial or full_height is None:
        # Pour les détections normales, utiliser le point bas standard
        return center_x, y2
    else:
        # Pour les détections partielles, calculer le point bas ajusté
        # en utilisant la hauteur complète estimée (point milieu)
        current_height = y2 - y1
        offset = (full_height - current_height) / 2
        return center_x, y2 + offset

def get_adaptive_reference_points(boxes, track_ids, frame_ids, height_drop_threshold=None, visualize=False):
    """
    Calcule les points de référence adaptatifs pour les boîtes englobantes en
    identifiant les occlusions partielles et en ajustant les points en conséquence.
    
    Args:
        boxes: Tableau de boîtes englobantes au format [x1, y1, x2, y2]
        track_ids: ID des pistes correspondant aux boîtes
        frame_ids: ID des images correspondant aux boîtes
        height_drop_threshold: Seuil pour la chute de hauteur considérée comme une occlusion partielle
        visualize: Si True, génère des graphiques d'analyse pour chaque track_id
    
    Returns:
        reference_points: Points de référence pour chaque boîte
        partial_detections: Tableau booléen indiquant si la détection est partielle
    """
    # Utiliser la valeur par défaut de la configuration si non spécifiée
    if height_drop_threshold is None:
        height_drop_threshold = PARTIAL_DETECTION_CONFIG["height_drop_threshold"]
    
    # Identifier les détections partielles
    partial_detections = detect_partial_boxes(
        boxes, track_ids, frame_ids, 
        height_drop_threshold=height_drop_threshold,
        visualize=visualize
    )
    
    # Estimer les hauteurs complètes pour les pistes avec des détections partielles
    full_heights = estimate_full_height(boxes, track_ids, frame_ids, partial_detections)
    
    # Calculer les points de référence pour chaque boîte
    reference_points = np.zeros((len(boxes), 2))
    
    for i, (box, is_partial) in enumerate(zip(boxes, partial_detections)):
        if track_ids[i] <= 0:
            # Pour les détections sans ID de piste, utiliser le point bas standard
            x1, y1, x2, y2 = box
            reference_points[i] = [(x1 + x2) / 2, y2]
            continue
            
        # Récupérer la hauteur complète estimée pour cette piste
        full_height = full_heights.get(track_ids[i])
        
        # Calculer le point de référence adapté
        reference_points[i] = calculate_reference_point(
            box, is_partial, full_height
        )
    
    return reference_points, partial_detections

# Fonction de compatibilité avec l'ancienne API
def get_adaptive_reference_points_original(
    boxes: np.ndarray,
    track_ids: np.ndarray,
    inframe: np.ndarray
) -> np.ndarray:
    """
    Version de compatibilité pour l'interface originale.
    Utilise la méthode adaptative principale sans renvoyer les informations supplémentaires.
    """
    reference_points, _ = get_adaptive_reference_points(boxes, track_ids, inframe)
    return reference_points

# Fonction de compatibilité avec l'ancienne API améliorée
def get_improved_adaptive_reference_points(
    boxes: np.ndarray,
    track_ids: np.ndarray,
    inframe: np.ndarray,
    max_variation_factor: float = 0.3
) -> np.ndarray:
    """
    Version de compatibilité pour l'interface améliorée.
    Utilise la méthode adaptative principale sans renvoyer les informations supplémentaires.
    """
    reference_points, _ = get_adaptive_reference_points(
        boxes, track_ids, inframe,
        height_drop_threshold=max_variation_factor
    )
    return reference_points

