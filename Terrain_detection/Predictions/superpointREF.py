from pathlib import Path
from matplotlib import pyplot as plt
from transformers import AutoImageProcessor, SuperPointForKeypointDetection
import json
import argparse
import os
import time
import cv2
import numpy as np
import torch
import time

class VideoProcessor:
    def __init__(self, config):
        self.config = config
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self._setup_paths()
        self._load_models()
        self._load_reference_data()
        
    def _setup_paths(self):
        """Initialize all required paths."""
        current_dir = Path(__file__).parent.resolve()
        base_path = current_dir.parent 
        base_path = base_path / "data"
        self.images_folder = base_path / "input_imgs"
        self.videos_folder = base_path / "videos"
        self.annotations_folder = base_path / "annotations"
        
        self.Hs_name = self.annotations_folder / f"Hs_supt{self.config['size_ratio']}.npy"
        self.video_out = self.videos_folder / f"pitch_supt{self.config['size_ratio']}.mp4"
        self.video_in = self.videos_folder / self.config['video_in']

    def _load_models(self):
        """Load the SuperPoint model and processor."""
        self.processor = AutoImageProcessor.from_pretrained("magic-leap-community/superpoint")
        self.model = SuperPointForKeypointDetection.from_pretrained("magic-leap-community/superpoint")
        self.model = self.model.to(self.device)
        
    def _load_reference_data(self):
        """Load reference images and annotations."""
        self.imgs = []
        self.annots = []
        self.annots_idx = []
        
        for i in self.config['i_frame']:
            annots_name = self.annotations_folder / f"pts_dict_{i}_New.npy"
            imgs_path = self.images_folder / f"img_{i}.png"
            
            img, pts, idents = self._load_image_and_points(imgs_path, annots_name)
            self.imgs.append(img)
            self.annots.append(pts)
            self.annots_idx.append(idents)
        self._process_reference_images()
        
    def _load_image_and_points(self, img_path, pts_path):
        raw = np.load(pts_path, allow_pickle=True)

        # Caso o conteúdo seja já um dicionário (correto)
        if isinstance(raw, dict):
            data = raw

        # Caso seja um array escalar contendo um dict
        elif isinstance(raw, np.ndarray) and raw.shape == () and isinstance(raw.item(), dict):
            data = raw.item()

        elif isinstance(raw, np.ndarray) and len(raw) == 1 and isinstance(raw[0], dict):
            data = raw[0]

        elif isinstance(raw, np.ndarray) and raw.ndim == 2 and raw.shape[1] == 2:
            data = {
                "pts": raw,
                "ident": np.arange(len(raw))
            }

        else:
            raise ValueError(f"Formato inválido em {pts_path}: tipo {type(raw)}")

        img = cv2.imread(str(img_path))
        return img, data["pts"], data["ident"]


        
    def _process_reference_images(self):
        """Process reference images for matching."""
        self.ref_hist = self._calc_reference_histograms(self.imgs)
        h, w = self.imgs[0].shape[:2]
        
        if self.config['size_ratio'] != 1:
            self.w_resize, self.h_resize = int(w / self.config['size_ratio']), int(h / self.config['size_ratio'])
        
        self.rgbs = self._prepare_rgb_images(self.imgs)
        self.kpts_ref, self.descs_ref = self._extract_keypoints(self.rgbs)
        
    def _calc_reference_histograms(self, images):
        """Calculate histograms for reference images."""
        return np.hstack([cv2.calcHist([img], [0], None, [256], [0, 256]) for img in images])
        
    def _prepare_rgb_images(self, images):
        """Convert and resize images to RGB format."""
        if self.config['size_ratio'] == 1:
            return [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in images]
        
        processed = []
        for img in images:
            img_r = cv2.resize(img, (self.w_resize, self.h_resize))
            processed.append(cv2.cvtColor(img_r, cv2.COLOR_BGR2RGB))
        return processed
        
    def _extract_keypoints(self, images):
        """Extract keypoints and descriptors from images."""
        with torch.no_grad():
            inputs = self.processor(images, return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)
        image_sizes = torch.tile(torch.tensor([self.h_resize, self.w_resize]), (len(self.rgbs), 1)).to(self.device)
        outputs = self.processor.post_process_keypoint_detection(outputs, image_sizes)
            
        kpts = []
        descs = []
        
        for output in outputs:
            kp = output['keypoints'].to('cpu').numpy()
            desc = output['descriptors'].to('cpu').numpy()
            scores = output['scores'].to('cpu').numpy()
            
            # Apply confidence threshold and outboard filter
            good_scores = scores > (self.config['conf_thresh'] / 100)
            kp = kp[good_scores]
            desc = desc[good_scores]
            outboard = np.logical_not((kp[:, 1] > 875) * (kp[:, 0] < 325))
            
            kpts.append(kp[outboard])
            descs.append(desc[outboard])
        return kpts, descs
        
    def process_video(self):
        """Process the video file and compute homographies."""
        video_in = self.videos_folder / self.config['video_in']
        video_capture = cv2.VideoCapture(video_in)
        if not video_capture.isOpened():
            raise IOError(f"Cannot open video file: {video_in}")
            
        self._init_video_properties(video_capture)
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, self.config['init_frame'])
        
        Hs = []
        t0 = time.time()
        t_match = 0
        
        with torch.no_grad():
            for i in range(0, self.config['max_frames'], self.config['batch_size']):
                batch_data = self._process_batch(video_capture)
                
                if not batch_data:
                    break
                batch_results, t_match  = self._process_keypoints(batch_data, t_match)
                Hs.extend(batch_results)
                if i % 100 == 0:
                    print(f"Processed {i} frames in {time.time() - t0:.2f}s (matching: {t_match:.2f}s)")
        
        self._save_homographies(Hs)
        self._create_output_video()
        
    def _init_video_properties(self, video_capture):
        """Initialize video properties."""
        self.video_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.video_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = video_capture.get(cv2.CAP_PROP_FPS)
        self.frame_count = video_capture.get(cv2.CAP_PROP_FRAME_COUNT)
        
    def _process_batch(self, video_capture):
        """Process a batch of video frames."""
        batch_imgs = []
        hist_matches = []
        
        for _ in range(self.config['batch_size']):
            ret, frame = video_capture.read()
            if not ret:
                return None
                
            i_match, _ = self._best_match(frame, self.ref_hist)
            hist_matches.append(i_match)
            batch_imgs.append(frame)
            
        return {'images': batch_imgs, 'matches': hist_matches}
        
    def _best_match(self, image, ref_hist):
        """Find the best matching reference image based on histogram."""
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        match_probs = [cv2.matchTemplate(ref_hist[:, i], hist, cv2.TM_CCOEFF_NORMED)[0][0] 
                      for i in range(ref_hist.shape[1])]
        best_match = np.argmax(match_probs)
        return best_match, match_probs
        
    def _process_keypoints(self, batch_data, t_match):
        """Process keypoints for a batch of frames."""
        rgbs = self._prepare_rgb_images(batch_data['images'])
        inputs = self.processor(rgbs, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        
        image_sizes = torch.tile(torch.tensor([self.h_resize, self.w_resize]), 
                               (len(rgbs), 1)).to(self.device)
        outputs = self.processor.post_process_keypoint_detection(outputs, image_sizes)
        
        batch_results = []
        t_match_start = time.time()
        
        for i, (i_match, output) in enumerate(zip(batch_data['matches'], outputs)):
            kp, desc = self._filter_keypoints(output)
            
            # Feature matching
            flann = cv2.FlannBasedMatcher(self.config['index_params'], self.config['search_params'])
            matches = flann.knnMatch(self.descs_ref[i_match], desc, k=2)
            
            # Filter good matches
            good = [m for m, n in matches if m.distance < 0.7 * n.distance]
            
            if len(good) > self.config['min_match_count']:
                src_pts = np.float32([self.kpts_ref[i_match][m.queryIdx] for m in good]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp[m.trainIdx] for m in good]).reshape(-1, 1, 2)
                
                # Compute homographies
                Mratio, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                M = np.diag([self.config['size_ratio'], self.config['size_ratio'], 1]) @ Mratio @ np.diag(
                    [1 / self.config['size_ratio'], 1 / self.config['size_ratio'], 1])
                M[2,2] = i_match
                new_pts = cv2.perspectiveTransform(self.annots[i_match].reshape(-1, 1, 2), M).squeeze()
                if len(self.annots[i_match]) >= 4 and len(new_pts) >= 4:
                    M2img, _ = cv2.findHomography(self.annots[i_match], new_pts, cv2.RANSAC)
                    M2pitch, _ = cv2.findHomography(new_pts, self.annots[i_match], cv2.RANSAC)
                    batch_results.append(np.stack((M, M2img, M2pitch)))
                else:
                    print(f"Frame {i} ignorado: menos de 4 pontos correspondentes.")
                                
                
            if self.config['plot_pts']:
                self._plot_points(batch_data['images'][i], new_pts, i, i_match)
                
        t_match += time.time() - t_match_start
        return batch_results, t_match
        
    def _filter_keypoints(self, output):
        """Filter keypoints based on confidence and position."""
        kp = output['keypoints'].to('cpu').numpy()
        desc = output['descriptors'].to('cpu').numpy()
        scores = output['scores'].to('cpu').numpy()
        
        good_scores = scores > (self.config['conf_thresh'] / 100)
        kp = kp[good_scores]
        desc = desc[good_scores]
        outboard = np.logical_not((kp[:, 1] > 875) * (kp[:, 0] < 325))
        
        return kp[outboard], desc[outboard]
        
    def _plot_points(self, frame, points, frame_num, match_idx):
        """Visualize points on frame."""
        for pt in points.astype(np.int16):
            cv2.circle(frame, pt, 10, (0, 255, 0), -1)
            
        plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        plt.title(f"Frame {frame_num} (Match: {match_idx})")
        plt.axis('off')
        plt.show()
        
    def _save_homographies(self, Hs):
        """Save computed homographies to file."""
        Hs_array = np.array(Hs)
        np.save(self.Hs_name, Hs_array)
        
    def _create_output_video(self):
        """Create output video with annotated frames."""
        Hs = np.load(self.Hs_name)
        i_ref = Hs[:, 0, 2, 2].copy().astype(np.int16)
        Hs[:, 0, 2, 2] = 1
        
        video_capture = cv2.VideoCapture(self.video_in)
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, self.config['init_frame'])
        
        avi_name = self.videos_folder / "results.avi"
        video_writer = cv2.VideoWriter(
            str(avi_name),
            cv2.VideoWriter_fourcc(*'MJPG'),
            self.fps,
            (self.video_width, self.video_height)
        )
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 2
        font_color = (0, 0, 255)
        thickness = 3
        
        for i in range(min(len(Hs), self.config['max_frames'])):
            ret, frame = video_capture.read()
            if not ret:
                break
                
            i_match = i_ref[i]
            new_pts = cv2.perspectiveTransform(
                self.annots[i_match].reshape(-1, 1, 2),
                Hs[i, 0]
            ).squeeze()
            for pt in new_pts.astype(np.int16):
                cv2.circle(frame, pt, 10, (0, 255, 0), -1)
                
            cv2.putText(frame, str(i), (100, 100), 
                       font, font_scale, font_color, thickness, cv2.LINE_AA)
            video_writer.write(frame)
            
        video_capture.release()
        video_writer.release()
        
        # Convert to MP4
        self._convert_video_format(avi_name, self.video_out)
        os.remove(avi_name)
        
    def _convert_video_format(self, input_path, output_path):
        """Convert video format using ffmpeg."""
        cmd = f"ffmpeg -v quiet -i {input_path} -vf yadif=0 -vcodec mpeg4 -qmin 3 -qmax 3 {output_path}"
        os.system(cmd)

if __name__ == "__main__":
    #Config comes from superpoint_config.json
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help="Path to JSON config file")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = json.load(f)
    processor = VideoProcessor(config)
    start = time.time()
    processor.process_video()
    print(f"Time to process video: {time.time() - start}")
