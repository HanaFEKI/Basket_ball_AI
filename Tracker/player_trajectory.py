#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyse des trajectoires d'un joueur spécifique pour identifier les trous dus aux occlusions
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import argparse
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent.parent
player_detection_path = str(project_root / "Tracker" / "src" / "utils")
sys.path.insert(0, player_detection_path)
from pitch_utils import draw_pitch

def analyze_player_trajectory(dict_file, player_id=1, output_file=None, show_plot=True):
    """
    Analyse et visualise la trajectoire d'un joueur spécifique.
    
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
    track_ids = data_dict['track_ids_graph'].astype(np.int16)
    xy = data_dict['xy']
    
    # Filtrer pour le joueur spécifié
    player_mask = track_ids == player_id
    player_frames = inframe[player_mask]
    player_positions = xy[player_mask]
    
    # Trier par numéro de frame
    sort_idx = np.argsort(player_frames)
    player_frames = player_frames[sort_idx]
    player_positions = player_positions[sort_idx]
    
    # Créer la visualisation
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # Plot 1: Trajectoire sur le terrain
    pitch = draw_pitch()
    pitch_rgb = cv2.cvtColor(pitch, cv2.COLOR_BGR2RGB)
    ax1.imshow(pitch_rgb)
    ax1.set_title(f"Trajectoire du joueur {player_id}")
    
    # Utiliser un dégradé de couleurs pour montrer la progression temporelle
    n_points = len(player_positions)
    #colors = plt.cm.viridis(np.linspace(0, 1, n_points))
    
    # Tracer la trajectoire avec le dégradé de couleurs
    ax1.scatter(player_positions[:, 0] * 40 + 50, player_positions[:, 1] * 40 + 50, 
               c=np.arange(n_points), cmap='viridis', s=30, alpha=0.7)
    ax1.plot(player_positions[:, 0] * 40 + 50, player_positions[:, 1] * 40 + 50, 
             'r-', alpha=0.3, linewidth=1)
    
    # Marquer le début et la fin
    ax1.scatter(player_positions[0, 0] * 40 + 50, player_positions[0, 1] * 40 + 50, 
               color='blue', s=100, marker='o', label='Début')
    ax1.scatter(player_positions[-1, 0] * 40 + 50, player_positions[-1, 1] * 40 + 50, 
               color='red', s=100, marker='x', label='Fin')
    
    ax1.legend()
    
    # Plot 2: Continuité des frames (pour identifier les trous)
    ax2.set_title(f"Continuité des détections pour le joueur {player_id}")
    
    # Calculer les différences entre frames consécutives
    frame_diffs = np.diff(player_frames)
    
    # Identifier les trous (gaps) importants
    gaps = np.where(frame_diffs > 1)[0]
    
    ax2.plot(np.arange(len(player_frames)), player_frames, 'b-', label='Frames avec détection')
    ax2.set_xlabel("Indice de détection")
    ax2.set_ylabel("Numéro de frame")
    
    # Mettre en évidence les trous
    for gap_idx in gaps:
        gap_start = player_frames[gap_idx]
        gap_end = player_frames[gap_idx + 1]
        gap_size = gap_end - gap_start - 1
        
        # Ne mettre en évidence que les trous significatifs (> 1 frame)
        if gap_size > 1:
            ax2.axvspan(gap_idx, gap_idx + 1, color='red', alpha=0.3)
            # Annoter avec la taille du trou
            ax2.annotate(f"{gap_size} frames", 
                        xy=((gap_idx + gap_idx + 1) / 2, (player_frames[gap_idx] + player_frames[gap_idx + 1]) / 2),
                        xytext=(0, 30), textcoords='offset points',
                        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2", color='black'),
                        ha='center', va='bottom', color='red', fontsize=10)
    
    # Ajouter des statistiques
    total_gaps = len(gaps)
    large_gaps = len(np.where(frame_diffs > 5)[0])
    max_gap = frame_diffs.max() if len(frame_diffs) > 0 else 0
    
    stats = (f"Statistiques:\n"
             f"- Nombre total de détections: {len(player_frames)}\n"
             f"- Nombre de trous: {total_gaps}\n"
             f"- Trous > 5 frames: {large_gaps}\n"
             f"- Plus grand trou: {max_gap} frames\n"
             f"- Frame début: {player_frames[0]}\n"
             f"- Frame fin: {player_frames[-1]}")
    
    ax2.text(0.02, 0.97, stats, transform=ax2.transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Visualisation sauvegardée dans {output_file}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()

def generate_all_player_trajectories(dict_file, output_dir='.'):
    """
    Génère des visualisations pour tous les joueurs significatifs
    (ceux ayant plus d'un certain nombre de détections)
    """
    import os
    
    data_dict = np.load(dict_file, allow_pickle=True).item()
    track_ids = data_dict['track_ids_graph'].astype(np.int16)
    unique_ids = np.unique(track_ids[track_ids > 0])
    
    # Compter le nombre de détections par joueur
    id_counts = {}
    for id in unique_ids:
        id_counts[id] = np.sum(track_ids == id)
    
    # Trier les joueurs par nombre de détections (décroissant)
    sorted_ids = sorted(id_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Générer des visualisations pour les N joueurs les plus détectés
    top_n = 5
    print(f"Génération des visualisations pour les {top_n} joueurs les plus suivis...")
    
    for i, (player_id, count) in enumerate(sorted_ids[:top_n]):
        print(f"Joueur {player_id}: {count} détections")
        output_file = os.path.join(output_dir, f"joueur_{player_id}_trajectoire.png")
        analyze_player_trajectory(dict_file, player_id, output_file, show_plot=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyse de trajectoire de joueur')
    parser.add_argument('--dict_file', type=str, default='clip_dict_4.npy',
                        help='Fichier de données de tracking (.npy)')
    parser.add_argument('--player_id', type=int, default=1,
                        help='ID du joueur à analyser')
    parser.add_argument('--output', type=str, default=None,
                        help='Fichier de sortie pour sauvegarder la visualisation')
    parser.add_argument('--all', action='store_true',
                        help='Générer des visualisations pour tous les principaux joueurs')
    
    args = parser.parse_args()
    
    if args.all:
        generate_all_player_trajectories(args.dict_file)
    else:
        analyze_player_trajectory(args.dict_file, args.player_id, args.output) 