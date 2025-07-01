#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comparaison des trajectoires de joueurs avec et sans points de référence adaptatifs
pour évaluer la réduction des trous dus aux occlusions
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import argparse
import os
from tqdm import tqdm
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent.parent
player_detection_path = str(project_root / "Tracker" / "src" / "utils")
sys.path.insert(0, player_detection_path)
from pitch_utils import draw_pitch
from config import FILE_CONFIG

def compare_player_trajectories(dict_file, player_id=1, output_file=None, show_plot=True):
    """
    Compare les trajectoires d'un joueur spécifique avec et sans points de référence adaptatifs.
    
    Args:
        dict_file: Fichier contenant les données de tracking (.npy)
        player_id: ID du joueur à analyser
        output_file: Fichier de sortie pour sauvegarder la visualisation (optionnel)
        show_plot: Afficher le graphique à l'écran
    """
    # Charger les données
    data_dict = np.load(dict_file, allow_pickle=True).item()
    bboxes = data_dict['bboxes']
    inframe = bboxes[:,0].astype(np.int16)
    
    # Utiliser track_ids_graph si disponible, sinon track_ids standard
    if 'track_ids_graph' in data_dict:
        track_ids = data_dict['track_ids_graph'].astype(np.int16)
    else:
        track_ids = data_dict['track_ids'].astype(np.int16)
    
    # Récupérer les positions standard et adaptatives
    xy_standard = data_dict['xy']
    xy_adaptive = data_dict['xy_adaptive'] if 'xy_adaptive' in data_dict else None
    
    if xy_adaptive is None:
        print(f"Erreur: Coordonnées adaptatives non trouvées dans {dict_file}")
        return
    
    # Filtrer pour le joueur spécifié
    player_mask = track_ids == player_id
    player_frames = inframe[player_mask]
    player_positions_std = xy_standard[player_mask]
    player_positions_adp = xy_adaptive[player_mask]
    
    # Trier par numéro de frame
    sort_idx = np.argsort(player_frames)
    player_frames = player_frames[sort_idx]
    player_positions_std = player_positions_std[sort_idx]
    player_positions_adp = player_positions_adp[sort_idx]
    
    # Créer la visualisation
    fig = plt.figure(figsize=(18, 12))
    
    # 1. Trajectoires sur le terrain (standard vs adaptatif)
    ax1 = plt.subplot2grid((2, 3), (0, 0), colspan=2)
    pitch = draw_pitch()
    pitch_rgb = cv2.cvtColor(pitch, cv2.COLOR_BGR2RGB)
    ax1.imshow(pitch_rgb)
    ax1.set_title(f"Comparaison des trajectoires du joueur {player_id}")
    
    # Nombre de points
    n_points = len(player_positions_std)
    
    # Tracer les deux trajectoires
    ax1.plot(player_positions_std[:, 0] * 40 + 50, player_positions_std[:, 1] * 40 + 50, 
             'b-', alpha=0.5, linewidth=1.5, label='Standard')
    ax1.plot(player_positions_adp[:, 0] * 40 + 50, player_positions_adp[:, 1] * 40 + 50, 
             'r-', alpha=0.5, linewidth=1.5, label='Adaptatif')
    
    # Marqueurs pour début/fin
    ax1.scatter(player_positions_std[0, 0] * 40 + 50, player_positions_std[0, 1] * 40 + 50, 
               color='blue', s=100, marker='o')
    ax1.scatter(player_positions_std[-1, 0] * 40 + 50, player_positions_std[-1, 1] * 40 + 50, 
               color='blue', s=100, marker='x')
    ax1.scatter(player_positions_adp[0, 0] * 40 + 50, player_positions_adp[0, 1] * 40 + 50, 
               color='red', s=100, marker='o')
    ax1.scatter(player_positions_adp[-1, 0] * 40 + 50, player_positions_adp[-1, 1] * 40 + 50, 
               color='red', s=100, marker='x')
    
    ax1.legend()
    
    # 2. Zone d'intérêt zoomée (où les trajectoires diffèrent significativement)
    ax2 = plt.subplot2grid((2, 3), (0, 2))
    ax2.imshow(pitch_rgb)
    ax2.set_title("Zone avec occlusions (zoom)")
    
    # Calculer les différences entre les trajectoires pour trouver la zone d'intérêt
    diffs = np.linalg.norm(player_positions_std - player_positions_adp, axis=1)
    max_diff_idx = np.argmax(diffs)
    
    # Sélectionner une plage autour de la différence maximale
    window = 10
    start_idx = max(0, max_diff_idx - window)
    end_idx = min(len(diffs), max_diff_idx + window)
    
    # Tracer les segments de trajectoire dans la zone d'intérêt
    ax2.plot(player_positions_std[start_idx:end_idx, 0] * 40 + 50, 
             player_positions_std[start_idx:end_idx, 1] * 40 + 50, 
             'b-', alpha=0.7, linewidth=2, label='Standard')
    ax2.plot(player_positions_adp[start_idx:end_idx, 0] * 40 + 50, 
             player_positions_adp[start_idx:end_idx, 1] * 40 + 50, 
             'r-', alpha=0.7, linewidth=2, label='Adaptatif')
    
    # Ajouter des marqueurs pour montrer la progression temporelle
    for i in range(start_idx, end_idx):
        alpha = 0.3 + 0.7 * ((i - start_idx) / (end_idx - start_idx))
        ax2.scatter(player_positions_std[i, 0] * 40 + 50, player_positions_std[i, 1] * 40 + 50, 
                   color='blue', s=30, alpha=alpha)
        ax2.scatter(player_positions_adp[i, 0] * 40 + 50, player_positions_adp[i, 1] * 40 + 50, 
                   color='red', s=30, alpha=alpha)
    
    # Ajuster le zoom pour se concentrer sur la zone d'intérêt
    center_x = (player_positions_std[max_diff_idx, 0] + player_positions_adp[max_diff_idx, 0]) / 2 * 40 + 50
    center_y = (player_positions_std[max_diff_idx, 1] + player_positions_adp[max_diff_idx, 1]) / 2 * 40 + 50
    zoom_radius = 150
    ax2.set_xlim(center_x - zoom_radius, center_x + zoom_radius)
    ax2.set_ylim(center_y - zoom_radius, center_y + zoom_radius)
    ax2.legend()
    
    # 3. Continuité des frames (pour identifier les trous) - Standard
    ax3 = plt.subplot2grid((2, 3), (1, 0))
    ax3.set_title("Continuité - Standard")
    
    # Calculer les différences entre frames consécutives
    frame_diffs_std = np.diff(player_frames)
    
    # Identifier les trous (gaps) importants
    gaps_std = np.where(frame_diffs_std > 1)[0]
    
    ax3.plot(np.arange(len(player_frames)), player_frames, 'b-', label='Frames avec détection')
    ax3.set_xlabel("Indice de détection")
    ax3.set_ylabel("Numéro de frame")
    
    # Mettre en évidence les trous
    for gap_idx in gaps_std:
        gap_start = player_frames[gap_idx]
        gap_end = player_frames[gap_idx + 1]
        gap_size = gap_end - gap_start - 1
        
        # Ne mettre en évidence que les trous significatifs (> 1 frame)
        if gap_size > 1:
            ax3.axvspan(gap_idx, gap_idx + 1, color='red', alpha=0.3)
            ax3.annotate(f"{gap_size}", 
                        xy=((gap_idx + gap_idx + 1) / 2, (player_frames[gap_idx] + player_frames[gap_idx + 1]) / 2),
                        ha='center', va='bottom', color='red', fontsize=8)
    
    # Ajouter des statistiques
    total_gaps_std = len(gaps_std)
    large_gaps_std = len(np.where(frame_diffs_std > 5)[0])
    max_gap_std = frame_diffs_std.max() if len(frame_diffs_std) > 0 else 0
    
    stats_std = (f"Statistiques (Standard):\n"
                f"- Nombre de trous: {total_gaps_std}\n"
                f"- Trous > 5 frames: {large_gaps_std}\n"
                f"- Plus grand trou: {max_gap_std} frames")
    
    ax3.text(0.02, 0.97, stats_std, transform=ax3.transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 4. Continuité des frames (version adaptative)
    ax4 = plt.subplot2grid((2, 3), (1, 1))
    ax4.set_title("Continuité - Adaptatif")
    
    # Les trajectoires adaptatives ont les mêmes frames que les standards
    # (même détections, seulement les positions changent)
    ax4.plot(np.arange(len(player_frames)), player_frames, 'r-', label='Frames avec détection')
    ax4.set_xlabel("Indice de détection")
    ax4.set_ylabel("Numéro de frame")
    
    # Mettre en évidence les mêmes trous (pour comparaison)
    for gap_idx in gaps_std:
        gap_start = player_frames[gap_idx]
        gap_end = player_frames[gap_idx + 1]
        gap_size = gap_end - gap_start - 1
        
        # Ne mettre en évidence que les trous significatifs (> 1 frame)
        if gap_size > 1:
            ax4.axvspan(gap_idx, gap_idx + 1, color='red', alpha=0.3)
            ax4.annotate(f"{gap_size}", 
                        xy=((gap_idx + gap_idx + 1) / 2, (player_frames[gap_idx] + player_frames[gap_idx + 1]) / 2),
                        ha='center', va='bottom', color='red', fontsize=8)
    
    # 5. Analyse de la différence entre les deux trajectoires
    ax5 = plt.subplot2grid((2, 3), (1, 2))
    ax5.set_title("Différence entre trajectoires")
    
    # Tracer la différence Euclidienne entre les positions standard et adaptatives
    ax5.plot(np.arange(len(diffs)), diffs, 'g-', linewidth=1.5, label='Distance entre trajectoires')
    ax5.set_xlabel("Indice de détection")
    ax5.set_ylabel("Distance (unités de terrain)")
    
    # Mettre en évidence les zones où la différence est significative
    threshold = np.percentile(diffs, 75)
    significant_diffs = diffs > threshold
    
    for i in range(len(diffs)):
        if significant_diffs[i]:
            ax5.axvspan(i-0.5, i+0.5, color='yellow', alpha=0.3)
    
    # Ajouter des statistiques sur les différences
    stats_diff = (f"Différences Trajectoires:\n"
                 f"- Différence max: {diffs.max():.2f}\n"
                 f"- Différence moyenne: {diffs.mean():.2f}\n"
                 f"- Écart-type: {diffs.std():.2f}\n"
                 f"- % Détections avec diff. significative: {100*np.sum(significant_diffs)/len(diffs):.1f}%")
    
    ax5.text(0.02, 0.97, stats_diff, transform=ax5.transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Mettre en évidence les frames avec des occlusions partielles
    if 'partial_detection_info' in data_dict and 'partial_detections' in data_dict['partial_detection_info']:
        partial_detections = data_dict['partial_detection_info']['partial_detections']
        player_partial = partial_detections[player_mask]
        player_partial = player_partial[sort_idx]
        
        for i in range(len(player_partial)):
            if player_partial[i]:
                ax5.axvspan(i-0.5, i+0.5, color='red', alpha=0.2)
                ax5.plot(i, diffs[i], 'ro', markersize=4)
    
    ax5.legend()
    
    # Titre principal
    plt.suptitle(f"Analyse comparative des trajectoires du joueur {player_id} - Standard vs Adaptatif", fontsize=16)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Visualisation comparative sauvegardée dans {output_file}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()

def compare_all_player_trajectories(dict_file, output_dir='.'):
    """
    Génère des visualisations comparatives pour tous les joueurs significatifs
    """
    # Créer le répertoire de sortie s'il n'existe pas
    os.makedirs(output_dir, exist_ok=True)
    
    data_dict = np.load(dict_file, allow_pickle=True).item()
    
    # Utiliser track_ids_graph si disponible, sinon track_ids standard
    if 'track_ids_graph' in data_dict:
        track_ids = data_dict['track_ids_graph'].astype(np.int16)
    else:
        track_ids = data_dict['track_ids'].astype(np.int16)
    
    unique_ids = np.unique(track_ids[track_ids > 0])
    
    # Compter le nombre de détections par joueur
    id_counts = {}
    for id in unique_ids:
        id_counts[id] = np.sum(track_ids == id)
    
    # Trier les joueurs par nombre de détections (décroissant)
    sorted_ids = sorted(id_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Générer des visualisations pour les N joueurs les plus détectés
    top_n = min(10, len(sorted_ids))
    print(f"Génération des visualisations comparatives pour les {top_n} joueurs les plus suivis...")
    
    for i, (player_id, count) in enumerate(tqdm(sorted_ids[:top_n])):
        if count > 50:  # Ne garder que les joueurs avec suffisamment de détections
            output_file = os.path.join(output_dir, f"joueur_{player_id}_trajectoire_comparaison.png")
            compare_player_trajectories(dict_file, player_id, output_file, show_plot=False)

def add_trajectory_comparison_to_pipeline(dict_file):
    """
    Ajoute l'analyse des trajectoires au pipeline existant
    """
    print("Analyse des trajectoires avec et sans points de référence adaptatifs...")
    
    # Créer un répertoire pour les résultats
    output_dir = "trajectoires_comparaison"
    os.makedirs(output_dir, exist_ok=True)
    
    # Comparer les trajectoires de tous les joueurs significatifs
    compare_all_player_trajectories(dict_file, output_dir)
    
    print(f"Analyse terminée. Les visualisations sont disponibles dans le dossier '{output_dir}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Comparaison des trajectoires avec et sans points de référence adaptatifs')
    parser.add_argument('--dict_file', type=str, default=FILE_CONFIG["adaptive_dict_file"],
                        help='Fichier de données de tracking avec points de référence adaptatifs (.npy)')
    parser.add_argument('--player_id', type=int, default=2,
                        help='ID du joueur à analyser')
    parser.add_argument('--output', type=str, default=None,
                        help='Fichier de sortie pour sauvegarder la visualisation')
    parser.add_argument('--all', action='store_true',
                        help='Générer des visualisations pour tous les principaux joueurs')
    
    args = parser.parse_args()
    
    if args.all:
        compare_all_player_trajectories(args.dict_file)
    else:
        compare_player_trajectories(args.dict_file, args.player_id, args.output) 