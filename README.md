# Basketball Player Tracking and Tactical Analysis System

This project implements a robust tracking system for basketball games that detects and follows players and referees in real-time, maintains their identities, classifies teams, and visualizes their movements on a 2D radar—even under partial occlusions.

## Project Overview

The system integrates several state-of-the-art techniques:
- **Detection:** YOLOv6 is used for accurate player and referee detection.
- **Tracking:** ByteTrack associates detections across frames to maintain consistent tracking.
- **Team Classification:** Visual features are extracted with SIGLIP, followed by dimensionality reduction using UMAP and clustering with K-means to differentiate teams.
- **Projection:** Player positions are projected onto a 2D radar using homography matrices.

### Handling Partial Occlusions

- Occlusions are detected by monitoring abrupt changes in bounding box height and aspect ratio.
- When lower body parts are occluded, player positions are estimated from their recent trajectory history instead of relying solely on bounding boxes.
- Trajectory fragments are linked using a directed graph model with edges weighted by similarity probabilities, connected through the Bellman-Ford algorithm to reconstruct full paths.
- Team consistency is enforced using a Hidden Markov Model to reduce classification errors.

## Results and Limitations

- Successful tracking and team classification with effective 2D radar visualization.
- Some limitations remain, including occasional 2D position jumps during prolonged occlusions and sensitivity to camera angle and lighting conditions.

## Future Work

- Integrate advanced tactical data analysis to identify key players, preferred playing zones, and overall strategies.
- Implement predictive models for key events such as shots, passes, and interceptions.
- Automate statistical extraction using graph theory and machine learning techniques (e.g., with scikit-learn).

## Technologies Used

- YOLOv6 (object detection)
- ByteTrack (tracking)
- SIGLIP (feature extraction)
- UMAP (dimensionality reduction)
- K-means (clustering)
- Bellman-Ford algorithm (trajectory reconstruction)
- Hidden Markov Models (team classification consistency)
- Homography matrix for 2D projection

## Getting Started

1. Clone this repository:
```
git clone https://github.com/HanaFEKI/Basket_ball_AI.git
```
2. Install dependencies (see `requirements.txt`).
3. Run the detection and tracking pipeline on your video data.
4. Visualize the player trajectories on the 2D radar interface.

## Contact

For questions or collaboration opportunities, feel free to open an issue or contact me directly.

---

*This project was developed as part of a team effort focusing on computer vision and sports analytics.*
