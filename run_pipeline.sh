#!/bin/bash

echo "Getting ball_handler.npy"
python3 Boxes_Detection/src/func_ball_handler.py

echo "Getting ball.npy"
python3 Boxes_Detection/src/func_ball.py

echo "Boxes_Detection complete"

echo "Running collinear.py..."
python3 Terrain_Detection/src/pos_processing/collinear.py \
  --config Terrain_Detection/src/prediction/superpoint_config.json

echo "Running superpointREF.py with config..."
python3 Terrain_Detection/src/prediction/superpointREF.py \
  --config Terrain_Detection/src/prediction/superpoint_config.json

echo "Terrain Dection complete."

echo "Running Tracker"
python Tracker/src/run_adaptive_homography.py
