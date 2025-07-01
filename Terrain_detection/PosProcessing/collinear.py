from tqdm import tqdm
from pathlib import Path
import argparse
import json
import cv2
import numpy as np

def load_data(img_name, pts_name):
    """Load image and points data."""
    data = np.load(pts_name, allow_pickle=True).item()
    img = cv2.imread(img_name)
    pts = data["pts"]
    idents = data["ident"]
    return img, pts, idents

def draw_points(img, pts, color=(255, 255, 0), radius=3):
    """Draw points on the image."""
    for x, y in pts:
        cv2.circle(img, (int(round(x)), int(round(y))), radius, color, -1)

def fit_and_project_line(points):
    """Fit a line using PCA and project points onto the best fit line."""
    points = np.array(points)
    if len(points) < 2:
        return points, None  # Not enough points to fit a line

    mean = np.mean(points, axis=0)
    cov_matrix = np.cov(points.T)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    direction = eigenvectors[:, np.argmax(eigenvalues)]
    projected_points = np.array([mean + np.dot(pt - mean, direction) * direction for pt in points])
    return projected_points, direction

def fit_collinear_points(pts, idents, points_ident):
    """Fit collinear points and update the original points."""
    collinear_points = []
    original_idx = []
    for idx, i in enumerate(idents):
        if i in points_ident:
            collinear_points.append(pts[idx])
            original_idx.append(idx)
    collinear_points, direction = fit_and_project_line(collinear_points)
    for i, pt in zip(original_idx, collinear_points):
        pts[i] = pt
    return collinear_points, direction, original_idx

def calculate_vanishing_points(pts, directions, original_idx, scale_factor=3000):
    """Calculate vanishing points and their bounding box."""
    min_x, max_x, min_y, max_y = float('inf'), -float('inf'), float('inf'), -float('inf')
    vanishing_pts = []
    for k, group_points in enumerate(original_idx):
        if len(group_points) < 1:
            continue
        ref_point = pts[group_points[0]]
        direction = directions[k]
        p1 = (ref_point + scale_factor * direction).astype(int)
        p2 = (ref_point - scale_factor * direction).astype(int)
        min_x = min(p1[0], p2[0], min_x)
        max_x = max(p1[0], p2[0], max_x)
        min_y = min(p1[1], p2[1], min_y)
        max_y = max(p1[1], p2[1], max_y)
        vanishing_pts.append([p1, p2])
    return vanishing_pts, min_x, max_x, min_y, max_y

def intersection(p1, p2, p3, p4):
    """Calculate the intersection of two lines."""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    D = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if D == 0:
        return None  # Lines are parallel

    Dx = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4))
    Dy = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4))
    px = Dx / D
    py = Dy / D
    return int(px), int(py)

def find_all_intersections(new_img, vanishing_pts, min_x, min_y):
    """Find and draw all intersections of vanishing lines."""
    intersections = []
    for idx, x1 in enumerate(vanishing_pts):
        for x2 in vanishing_pts[idx+1:]:
            p1, p2 = x1[0], x1[1]
            p3, p4 = x2[0], x2[1]
            r1, r2 = intersection(p1, p2, p3, p4)
            if r1 is not None and r2 is not None:
                intersections.append([r1, r2])
                r1_adj = r1 - min_x
                r2_adj = r2 - min_y
                cv2.circle(new_img, (r1_adj, r2_adj), 3, (255, 0, 255), -1)
    return intersections

def point_to_line_distances(points, pts_idxs, pts):
    """Calculate distances from points to lines."""
    distances = [[] for _ in range(len(points))]
    for i, point in enumerate(points):
        for idxs in pts_idxs:
            if len(idxs) < 2:
                continue
            (x1, y1), (x2, y2) = pts[idxs[0]], pts[idxs[1]]
            A = y2 - y1
            B = x1 - x2
            C = x2 * y1 - x1 * y2
            x0, y0 = point
            d = abs(A * x0 + B * y0 + C) / np.sqrt(A**2 + B**2)
            distances[i].append(d)
    return distances

def calculate_rmse(distances):
    """Calculate RMSE for each set of distances."""
    error_values_rmse = [np.sqrt(np.mean(np.square(dist))) for dist in distances]
    best_rmse = np.argmin(error_values_rmse)
    return best_rmse

def make_parallel(chosen_pt, pts_idxs, pts, img):
    """Adjust points to lie on parallel lines."""
    for pts_idx in pts_idxs:
        if len(pts_idx) > 2:
            line_points = pts[pts_idx]
            mean = np.mean(line_points, axis=0)
            line_to_be_traced = []
            for i, Q in enumerate(line_points):
                d = chosen_pt - mean
                v = Q - mean
                t = np.dot(v, d) / np.dot(d, d)
                proj_pt = mean + t * d
                line_to_be_traced.append(proj_pt)
                pts[pts_idx[i]] = proj_pt
            line_to_be_traced = np.array(line_to_be_traced, np.int32).reshape((-1, 1, 2))
            cv2.polylines(img, [line_to_be_traced], isClosed=False, color=(255, 234, 0), thickness=2)

def main(frames):
    i_frame = frames
    print(f"Processing frames: {i_frame}")
    current_dir = Path(__file__).parent.resolve()
    base_path = current_dir.parent 
    base_path = base_path / "data"
    images_folder = base_path / "input_imgs"
    videos_folder = base_path / "videos"
    annotations_folder = base_path / "annotations"
    output_folder = base_path / "output_imgs"
    for i in tqdm(range(len(i_frame))):
        img_name = images_folder/ f"img_{i_frame[i]}.png"
        pts_name = annotations_folder/ f"pts_dict_{i_frame[i]}.npy"

        img, pts, idents = load_data(img_name, pts_name)
        draw_points(img, pts)

        collinearH_Points = [[1, 3, 5], [19, 21], [7, 9, 13, 15], [11, 17, 23], [10, 16, 22], [6, 8, 12, 14], [18, 20], [0, 2, 4]]
        collinearV_Points = [[1, 19, 7, 6, 18, 0], [9, 11, 10, 8], [3, 23, 22, 2], [15, 17, 16, 14], [5, 21, 13, 12, 20, 4]]

        parallel_lines_H = []
        original_idx_H = []
        directions_H = []
        for colPoints in collinearH_Points:
            line_to_be_traced, direction, idxs = fit_collinear_points(pts, idents, colPoints)
            parallel_lines_H.append(line_to_be_traced)
            original_idx_H.append(idxs)
            if direction is not None:
                directions_H.append(direction)
            line_to_be_traced = np.array(line_to_be_traced, np.int32).reshape((-1, 1, 2))
            cv2.polylines(img, [line_to_be_traced], isClosed=False, color=(72, 125, 0), thickness=2)

        parallel_lines_V = []
        original_idx_V = []
        directions_V = []
        for colPoints in collinearV_Points:
            line_to_be_traced, direction, idxs = fit_collinear_points(pts, idents, colPoints)
            parallel_lines_V.append(line_to_be_traced)
            original_idx_V.append(idxs)
            directions_V.append(direction)
            line_to_be_traced = np.array(line_to_be_traced, np.int32).reshape((-1, 1, 2))
            cv2.polylines(img, [line_to_be_traced], isClosed=False, color=(0, 0, 255), thickness=2)

        #Vanishing points and new size window to show intersections
        vanishing_pts, min_x, max_x, min_y, max_y = calculate_vanishing_points(pts, directions_V, original_idx_V)
        paddle = 0
        new_width = max_x - min_x + paddle
        new_height = max_y - min_y + paddle
        new_img = np.ones((new_height, new_width, 3), dtype=np.uint8) * 255

        for tps in vanishing_pts:
            p1 = (tps[0][0] - min_x, tps[0][1] - min_y)
            p2 = (tps[1][0] - min_x, tps[1][1] - min_y)
            cv2.line(new_img, p1, p2, color=(0, 0, 0), thickness=3)

        intersections = find_all_intersections(new_img,vanishing_pts, min_x, min_y)
        distances = point_to_line_distances(intersections, original_idx_V, pts)
        chosen_pt_idx = calculate_rmse(distances)
        chosen_pt = intersections[chosen_pt_idx]
        make_parallel(chosen_pt, original_idx_V, pts, img)

        # Draw the chosen point with the least RMSE
        r1,r2 = chosen_pt
        r1_adj = r1 - min_x
        r2_adj = r2 - min_y
        cv2.circle(new_img, (r1_adj, r2_adj), 3, (25, 50, 255), -1)

        vanish = output_folder / f"vanishing_arrests_{i_frame[i]}.jpg"
        terrain_arrests = output_folder / f"terrain_arrests_{i_frame[i]}.jpg"
        cv2.imwrite(vanish, new_img)
        cv2.imwrite(terrain_arrests, img)
        pts_dict = {"pts": pts, "ident": idents}
        pts_name = annotations_folder/ f"pts_dict_{i_frame[i]}_New.npy"
        print(f"Saving points to {pts_name}")
        np.save(pts_name, pts_dict)

if __name__ == "__main__":
    #Config comes from superpoint_config.json
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help="Path to JSON config file")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    print(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = json.load(f)
    frames = config["i_frame"]
    main(frames)