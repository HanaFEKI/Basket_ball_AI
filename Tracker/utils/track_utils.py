#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 14:27:47 2025

@author: fenaux
"""

from matplotlib import pyplot as plt
from scipy.spatial import distance
from itertools import product
from team import get_crops
import supervision as sv
import cv2
import networkx as nx
import numpy as np


def run_sv_tracker(boxes_file: str) :
    """
    Run tracking on boxes with bytetrack

    Args:
         (str): Path to detections 
        

    """
    
    bboxes_ = np.load(boxes_file)
    inframe = bboxes_[:,0].astype(np.int16)
    frames = np.unique(inframe)
    confs = bboxes_[:,5]
    bboxes = bboxes_[:,1:5]
    
    tracker = sv.ByteTrack(minimum_consecutive_frames=3)
    tracker.reset() # needed to have tracks from 1 if video was processed before
    keep_track = np.array([])
    for i_frame in frames:
        in_i_frame = np.where(inframe== i_frame)[0]
        n_in_i_frame = len(in_i_frame)
        boxes_in_i_frame = bboxes[in_i_frame]
        conf_in_i_frame = confs[in_i_frame]
        
        detections = sv.Detections(xyxy = boxes_in_i_frame,
                                      class_id=np.zeros(n_in_i_frame),
                                      confidence=conf_in_i_frame,
                                      data={'0':np.arange(n_in_i_frame)})
        
        detections = tracker.update_with_detections(detections)

        tracks = - np.ones(n_in_i_frame)
        tracks[detections.data['0']] = detections.tracker_id.copy()
        keep_track = np.append(keep_track, tracks)
        #labels = [str(tracker_id) for tracker_id in detections.tracker_id]
    
    return {'bboxes':bboxes_,'track_ids':keep_track}
 

def box_and_track(boxes_file:str, track_file:str, dict_file:str, ConfOnly:bool = True):
    """
    
    Parameters
    ----------
    boxes_file:str name of file where detections are stored numpy array each row
        [i_frame, xtop, ytop, xbottom, ybottom, confidence]
    track_file:str name of file where tracker output are stored numpy array each row
            [i_frame, xtop, ytop, xbottom, ybottom, confidence, track_id]
    boxes may be different in boxes_file an track_file due to kalmann filtering
    ConfOnly : bool, optional
        DESCRIPTION. The default is True. If True pairing of boxes only relies on confidence
        should be set to true for BotSort False for DeepOcSort
        
    dict_file : file where to save results
    ['bboxes'] like boxes_file with kalmann improvement
    ['track_ids'] ids from track_file, id is -1 if not in a track
    Returns
    -------
    None.

    """

    tracks = np.load(track_file)
    if len(tracks.shape) == 1:
        tracks = tracks.reshape(-1,8)
    if np.array_equal(tracks[:,5], tracks[:,6]):
        tracks = np.delete(tracks, 6, axis=1)
        
    tracks_ids = tracks[:,-1]
    
    bboxes_ = np.load(boxes_file)
    inframe = bboxes_[:,0].astype(np.int16)
    frames = np.unique(inframe)
    
    in_which_track = -np.ones(len(bboxes_))
    for i_frame in frames:
        
        in_i_frame = np.where(inframe== i_frame)[0]
        n_in_i_frame = len(in_i_frame)
        boxes_in_i_frame = bboxes_[in_i_frame,1:]
        
        in_i_frame_track = np.where(tracks[:,0] == i_frame)[0]
        boxes_in_i_frame_track = tracks[in_i_frame_track,1:-1]
        
        if ConfOnly:
            conf_in_i_frame = boxes_in_i_frame[:,-1].reshape(-1,1)
            conf_in_i_frame_track = boxes_in_i_frame_track[:,-1].reshape(-1,1)
            pairdist = distance.cdist(conf_in_i_frame, conf_in_i_frame_track, 'euclidean')
            
        else:       
            pairdist = distance.cdist(boxes_in_i_frame, boxes_in_i_frame_track, 'euclidean')
                
        pairs = np.where(pairdist == 0)
      
        in_which_track[in_i_frame[pairs[0]]] = tracks_ids[in_i_frame_track[pairs[1]]]
        
        if i_frame % 100 == 0 and i_frame > 1:
            print( np.linalg.norm(
                bboxes_[in_i_frame[pairs[0]],1:-1] - tracks[in_i_frame_track[pairs[1]],1:-2],
                axis=1).max()
                )

        if ConfOnly: # to keep improvement due to kalmann filtering
            bboxes_[in_i_frame[pairs[0]],1:] = tracks[in_i_frame_track[pairs[1]],1:-1]

    
    deep_dict = {'bboxes':bboxes_,'track_ids':in_which_track.astype(np.int16)}
    np.save(dict_file, deep_dict)
     

def track_in_pitch(dict_file):
    
    data_dict = np.load(dict_file, allow_pickle=True).item()
    track_ids = data_dict['track_ids']
    xy = data_dict['xy']
    
    is_on_pitch = np.zeros(len(xy))
    i_tracks = np.unique(track_ids)
    i_tracks = i_tracks[i_tracks > - 1]
    
    in_pitch_x = (xy[:,0] > -2) * (xy[:,0] < 30)
    in_pitch_y = (xy[:,1] > -0.5) * (xy[:,1] < 15.5)
    in_pitch_flag = in_pitch_x * in_pitch_y
    for i_track in i_tracks:
        
        in_i_track = np.where(track_ids== i_track)[0]
        in_pitch_ratio = in_pitch_flag[in_i_track].mean()
        if in_pitch_ratio > 0.5:
            is_on_pitch[in_i_track] = 1
    
    data_dict['in_pitch'] = is_on_pitch.astype('bool_')
    np.save(dict_file, data_dict)
       
def StartsEnds(dict_file:str, pitch_only:bool = False):
    track_dict = np.load(dict_file, allow_pickle=True).item()
    
    bboxes_ = track_dict['bboxes']
    inframe = bboxes_[:,0].astype(np.int16)
    xy = track_dict['xy']
    track_ids = track_dict['track_ids']
    if pitch_only: 
        is_on_pitch = track_dict['in_pitch']
        track_ids = track_ids[is_on_pitch]
        inframe = inframe[is_on_pitch]
        xy = xy[is_on_pitch]
        
    i_tracks = np.unique(track_ids)
    i_tracks = i_tracks[i_tracks > -1]
    starts_ends = []
    for i_track in i_tracks:
        in_i_track = np.where(track_ids == i_track)[0]
        starts_ends.append([inframe[in_i_track].min(), inframe[in_i_track].max(), i_track])
        
    from matplotlib import pyplot as plt
    plt.scatter(xy[:,0], xy[:,1], c=track_ids, 
                norm = plt.Normalize(vmin=0, vmax=track_ids.max()), cmap="nipy_spectral" ,s=0.2)
    plt.show()
    return np.array(starts_ends)
    

def ChainTrack(dict_file:str):
    """
    associates tracks for tracks concerning the same team_id.
    Association is done with minimal distance first
    distance between two tracks is norm of ([end1, x1, y1] - [start2, x2, y2])
    start and end are the frame where tracks ends and starts
    xi, yi are corrdaintes of players on ground when tracks ends and starts
    times are normalized by fps
    dsitance normalized by a scale (2m)
    
    possible improvement add a distance depending on semantic contents

    Parameters
    ----------
    dict_file (str) : path to directoy describing detections and tracks
    ['to_keep'] (array) : boolean true for tracks mainly in pitch and moving
    inframe (array) : frame in wich detection is maide
    xy (array): pitch coordinates af detection
    track_ids (array) : track_id the detection belongs to after hmm correction
    team_id_ (array): team_id the detection belongs to after hmm correction

    Returns
    -------
    track_ids_chain (array): track_id after tracked have been merged

    """
    
    track_dict = np.load(dict_file, allow_pickle=True).item()
    
    to_keep = np.where(track_dict['to_keep'])[0]
    bboxes_ = track_dict['bboxes'][to_keep]
    inframe = bboxes_[:,0].astype(np.int16)
    xy = track_dict['xy'][to_keep]
    track_ids = track_dict['track_ids_hmm'][to_keep]
    team_id_ = track_dict['team_id_hmm'][to_keep]
    
    track_ids_chain = track_dict['track_ids_hmm'].copy()
    track_ids_hmm = track_dict['track_ids_hmm'].copy()
    for i_team in range(2):
        in_team = np.where( team_id_ == i_team)[0]
        track_ids_team = track_ids[in_team]
        
        unique_track_ids = np.unique(track_ids_team)
        xys, debs, fins = [], [], []
        for track_id in unique_track_ids:
            in_track = np.where(track_ids == track_id)[0]
            xy_in_track = xy[in_track]
            xys.append(np.append(xy_in_track[0], xy_in_track[-1]))
            debs.append(inframe[in_track[0]])
            fins.append(inframe[in_track[-1]])
        xys = np.array(xys) / 2 # 5 # 15
        debs = np.array(debs) / 30
        fins = np.array(fins) / 30
        debs = np.column_stack((debs, xys[:,:2]))
        fins = np.column_stack((fins, xys[:,2:]))
        
        StartEndDiff= debs[:,0].reshape(-1,1) - fins[:,0].reshape(1,-1)
        StartBeforeEnd = StartEndDiff <= 0
        StartTooLate = StartEndDiff > 4 # 2
        BadStart = StartBeforeEnd + StartTooLate
        pairdist = distance.cdist(debs, fins, 'euclidean')
        
        pairdist[BadStart] = 1e6
        mat = pairdist.copy()
        chains = []
        dists = []
        while True:
            min_idx = np.unravel_index(np.argmin(mat), mat.shape)
            # first index is the track that begins, second the track that ends
            max_dist = 4 # 2 #10
            if mat[min_idx] > max_dist: break
            chains.append(unique_track_ids[list(min_idx)])
            dists.append(mat[min_idx])
            mat[min_idx[0]] = 1e6
            mat[:,min_idx[1]] = 1e6
            
        chains = np.array(chains)
        chains = chains[np.argsort(chains[:,1])]
        for i1, i2 in chains:
            in1 = np.where(track_ids_hmm == i1)[0]
            in2 = np.where(track_ids_hmm == i2)[0]
            
            if track_ids_chain[in1[0]] < track_ids_chain[in2[0]]:
                track_ids_chain[in2] = track_ids_chain[in1[0]]
            else:
                track_ids_chain[in1] = track_ids_chain[in2[0]]
    
    track_dict['track_ids_chain'] = track_ids_chain.copy()
    np.save(dict_file, track_dict)
    
    return track_ids_chain

def GraphTrack(dict_file, start_chain, end_chain, fps=30, show_tracks=False):
    """
    associates tracks for tracks concerning the same team_id.
    Association is done to get a complete track i.e. from start_chain to end_chain.
    Association is done until we get 5 tracks by team or no more complete tracks are 
    possible to obtain. Time gap is increased to 10 seconds. No more condition 
    on maximum distance as if time gap increase distance will probably be increased too.
    distance between two tracks is norm of ([end1, x1, y1] - [start2, x2, y2])
    start and end are the frame where tracks ends and starts
    xi, yi are corrdaintes of players on ground when tracks ends and starts
    times are normalized by fps
    dsitance normalized by a scale (2m)
    
    possible improvement add a distance depending on semantic contents
    remaining problem how to determine start_chain and end_chain

    Parameters
    ----------
    dict_file (str) : path to directoy describing detections and tracks
    ['to_keep'] (array) : boolean true for tracks mainly in pitch and moving
    inframe (array) : frame in wich detection is maide
    xy (array): pitch coordinates af detection
    track_ids (array) : track_id the detection belongs to after hmm correction
    team_id_ (array): team_id the detection belongs to after hmm correction

    Returns
    -------
    track_ids_graph (array): track_id after tracked have been merged
    
    """
    
    track_dict = np.load(dict_file, allow_pickle=True).item()
    
    to_keep = np.where(track_dict['to_keep'])[0]
    bboxes_ = track_dict['bboxes'][to_keep]
    inframe = bboxes_[:,0].astype(np.int16)
    xy = track_dict['xy'][to_keep]
    track_ids = track_dict['track_ids_chain'][to_keep]
    team_id_ = track_dict['team_id_hmm'][to_keep]
    
    track_ids_chain = track_dict['track_ids_chain'].copy()
    track_ids_graph = track_ids_chain.copy()
    
    for i_team in range(2):
    
        in_team = np.where( team_id_ == i_team)[0]
        track_ids_team = track_ids[in_team]
        unique_track_ids = np.unique(track_ids_team)
        
        complete_tracks = 0
        track_to_chain, xys, debs, fins = [], [], [], []
        for track_id in unique_track_ids:
            in_track = np.where(track_ids == track_id)[0]
            if inframe[in_track[0]] <= start_chain and inframe[in_track[-1]] >= end_chain:
                complete_tracks += 1
            else:
                track_to_chain.append(track_id)
                xy_in_track = xy[in_track]
                xys.append(np.append(xy_in_track[0], xy_in_track[-1]))
                debs.append(inframe[in_track[0]])
                fins.append(inframe[in_track[-1]])
                
        track_to_chain = np.array(track_to_chain)        
        xys = np.array(xys) / 2 # 5 # 15
        debs = np.array(debs) / fps
        fins = np.array(fins) / fps
        debs = np.column_stack((debs, xys[:,:2]))
        fins = np.column_stack((fins, xys[:,2:]))
        
        if show_tracks:
            for track_id in track_to_chain:
                in_track = np.where(track_ids == track_id)[0]
                plt.scatter(inframe[in_track], np.ones(in_track.size) * track_id, s=0.5)
            plt.ylabel('track id'), plt.xlabel('frame number')
            plt.title(f" team {i_team}")
            plt.show()
        
        while( complete_tracks < 5):
        
            StartEndDiff= debs[:,0].reshape(-1,1) - fins[:,0].reshape(1,-1)
            StartBeforeEnd = StartEndDiff <= 0
            StartTooLate = StartEndDiff > 10#4 # 2
            BadStart = StartBeforeEnd + StartTooLate
            pairdist = distance.cdist(debs, fins, 'euclidean')
            
            pairdist[BadStart] = 1e6
            
            ntracks = len(track_to_chain)
            subA = np.zeros((ntracks,ntracks))
            
            # on each line we have the distance of the ending track to the
            #tracks that begin after
            for i,j in product(np.arange(ntracks), np.arange(ntracks)):
                if fins[i,0] < debs[j,0]: subA[i,j] = pairdist[j,i]
            subA = np.where(subA == 1e6, 0, subA) # no connestion is 0
            
            # addition of two extra nodes one for starting frame and one for 
            # ending frame. distance is minimal foar tracks including one of this frame
            # no edge for others
            # node 0 is for beginning
            # node ntracks+2 is for ending
            A = np.zeros((ntracks+2,ntracks+2))
            A[1:-1,1:-1] = subA.copy()
            startings = np.where(debs[:,0] <= start_chain / fps)[0]
            A[0, startings + 1] = 1e-2
            endings = np.where(fins[:,0] >= end_chain / fps)[0]
            A[endings + 1,-1] = 1e-2
            
            G = nx.from_numpy_array(A, create_using=nx.DiGraph)
            
            i_target = A.shape[1] - 1
            try:
                p = nx.bellman_ford_path(G, source=0, target=i_target, weight="weight")
            except:
                print(f"{complete_tracks} complete tracks and no more path")
                break
            
            path_weight = nx.path_weight(G, path=p, weight='weight')
            if path_weight >= 1e6:
                print(f"{complete_tracks} complete tracks and no more path wih valid gap")
                break
                
            
            new_chain = np.delete(p,[0,-1]) - 1 #removal of extra nodes
            
            # all tracks_ids to the id of the tracks with minimal id
            new_track_id = track_to_chain[new_chain].min()
            for i_track in track_to_chain[new_chain]:
                in_track = np.where(track_ids_chain == i_track)[0]
                track_ids_graph[in_track] = new_track_id
                
            # the tracks that have been used are no mre available
            track_to_chain = np.delete(track_to_chain, new_chain)
            debs = np.delete(debs, new_chain, axis=0)
            fins = np.delete(fins, new_chain, axis=0)
            complete_tracks += 1
            
        if complete_tracks == 5:
            print(f"{complete_tracks} complete tracks are available")
        
    track_dict['track_ids_graph'] = track_ids_graph.copy()
    np.save(dict_file, track_dict)


def crop_track(track_id, video_in, tracks, bboxes, inframe, stride, init_frame, 
               for_features=False):
    """
    Extract crops from the frame based on detected bounding boxes for a given track.

    Args :
    ----------
    track_id (int) : the track number where crops are collected
    video_in (string) : path to the video where observations and tracks have been obtaines
    tracks (nd.array) : tracks ids of the different detections
    bboxes (nd.array) : coordinates of the box of the different detections
    inframe T(nd.array) : frame number of the box of the different detections
    stride (int) : stride of the crops in the track
    init_frame (int) : first frame in which detections have been searched
    for_features (bool) : if True only beginings and endings crops are processed 

    Returns
    -------
    crops List[np.ndarray]: List of cropped images

    """
    bboxes = np.clip(bboxes,0,None)
    in_track = np.where(tracks == track_id)[0]
    in_track_frames = inframe[in_track]
    in_track_boxes = bboxes[in_track]
        
    cap = cv2.VideoCapture()
    cap.open(video_in )
    
    crops = []
    
    if for_features: to_get = np.hstack((in_track[:5], in_track[-5:]))
    else: to_get = in_track[::stride]
    
    in_track_frames = inframe[to_get]
    in_track_boxes = bboxes[to_get]
        
    for i_frame, box in zip(in_track_frames, in_track_boxes):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i_frame + init_frame)
        ret, frame = cap.read()
        #if i + first_in_track in in_track_frames:
        
        detections = sv.Detections(xyxy = box.reshape(1,4))
        crops += get_crops(frame, detections)
        
    return crops


def ShowTrackHmm(dict_file:str, video_in:str, unique_track_ids, stride:int = 4):
    
    track_dict = np.load(dict_file, allow_pickle=True).item()
        
    team_id_ = track_dict['team_id']
    team_id_hmm = track_dict['team_id_hmm']
    track_ids_ = track_dict['track_ids']
    
    bboxes_ = track_dict['bboxes']
    inframe_ = bboxes_[:,0].astype(np.int16)
    boxes_ = bboxes_[:,1:5]
    
    show = True
    if show:

        for track_id in unique_track_ids:
            in_track = np.where(track_ids_ == track_id)[0]
            id_in_track, cnt_in_track = np.unique(team_id_[in_track], return_counts=True)
            cnt0 = cnt_in_track[id_in_track==0] if (id_in_track==0).any() else 0
            cnt1 = cnt_in_track[id_in_track==1] if (id_in_track==1).any() else 0
            cnt2 = cnt_in_track[id_in_track==2] if (id_in_track==2).any() else 0
            
            do_hmm_1 = (max(cnt0, cnt1) > in_track.size / 5)
            do_hmm_2 = (cnt2 > in_track.size * 0.5) or (min(cnt0, cnt1) > in_track.size / 7)
            do_hmm = do_hmm_1 and do_hmm_2
            
            print('hmm', do_hmm)
            if not do_hmm: continue
            crops = crop_track(track_id, video_in, 
                               track_ids_, boxes_, inframe_,
                               stride=stride, init_frame=0)
            old_team = team_id_[in_track][::stride]
            new_team = team_id_hmm[in_track][::stride]
            show_frames = inframe_[in_track][::stride]
            for crop, old, new, i_frame in zip(crops, old_team, new_team, show_frames):
                
                plt.imshow(crop[...,::-1])
                plt.axis('off')
                if do_hmm: plt.title(f"{track_id}, {new}, {i_frame}, {old}")
                else: plt.title(f"{track_id}, {new}, {i_frame}")
                plt.show()
            print(track_id, np.unique(team_id_hmm[in_track], return_counts=True))

            print( 'continue y / n')
            stopIt = True  if str(input()) == 'n' else False
            if stopIt: break