from ultralytics import YOLO
from personal.basketball_analysis.utils import read_stub, save_stub

NUM_COCO_KEYPOINTS = 17
DEFAULT_BBOX_PAD_FRAC = 0.15  # pad crop by 15% of max(bbox_w, bbox_h) on each side
DEFAULT_BBOX_PAD_MIN_PX = 10  # floor padding in pixels, so small/close bboxes still get padded


class PoseEstimator:
    """
    A class that runs pose estimation on cropped player regions using already-known
    player track bounding boxes, rather than running pose detection on full frames.

    Design rationale (crop-then-detect, not detect-then-match):

    - Cropping each player's tracked bbox gives you the player's `track_id` for free.
      There is no separate IoU-matching step needed between pose detections and
      player tracks -- full-frame pose estimation would first detect all people in
      the frame, then have to match each detected person back to a track ID, which
      is fragile in crowded under-the-basket scenarios where players overlap.
    - Cropping to only tracked *players* skips wastefully re-detecting referees and
      spectators that a full-frame pass would also have to run pose estimation on.
    - All of one frame's player crops are batched into a single
      `model.predict(list_of_crops, ...)` call, matching `PlayerTracker`'s batching
      pattern, rather than being N separate (slow) per-player calls.
    """

    def __init__(self, model_path='yolo11n-pose.pt', device='cpu', conf=0.5,
                 bbox_pad_frac=DEFAULT_BBOX_PAD_FRAC, bbox_pad_min_px=DEFAULT_BBOX_PAD_MIN_PX):
        """
        Initialize the PoseEstimator with a YOLO pose model.

        Args:
            model_path (str): Path to the YOLO pose model weights.
            device (str): Inference device passed to YOLO ('cpu', 'cuda', 'mps').
                'mps' is silently downgraded to 'cpu' -- Ultralytics itself
                warns of a known Apple-MPS pose-model bug (confirmed live on
                this machine: "Apple MPS known Pose bug. Recommend
                device=cpu for Pose models", see
                https://github.com/ultralytics/ultralytics/issues/4031).
                Getting keypoints wrong isn't just slower, it would silently
                corrupt the already-heuristic violation detectors downstream,
                so this trades pose-estimation speed for correctness rather
                than leaving it to chance. 'cuda' passes through unchanged
                (no known equivalent bug there).
            conf (float): Detection confidence threshold.
            bbox_pad_frac (float): Fraction of max(bbox_w, bbox_h) to pad the crop
                by on each side. A tight player bbox can clip an extended
                dribbling/shooting arm, so padding matters for keypoint quality.
            bbox_pad_min_px (float): Floor padding in pixels, so small/close
                bboxes still get a usable amount of padding.
        """
        self.model = YOLO(model_path)
        self.device = 'cpu' if device == 'mps' else device
        self.conf = conf
        self.bbox_pad_frac = bbox_pad_frac
        self.bbox_pad_min_px = bbox_pad_min_px

    def _pad_bbox(self, bbox, frame_w, frame_h):
        """
        Symmetrically pad a bbox and clamp the result to frame bounds.

        Args:
            bbox (list): [x1, y1, x2, y2] player bounding box.
            frame_w (int): Frame width in pixels.
            frame_h (int): Frame height in pixels.

        Returns:
            tuple: (px1, py1, px2, py2) padded and clamped crop coordinates.
        """
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        pad = max(self.bbox_pad_min_px, self.bbox_pad_frac * max(w, h))

        px1 = max(0, x1 - pad)
        py1 = max(0, y1 - pad)
        px2 = min(frame_w, x2 + pad)
        py2 = min(frame_h, y2 + pad)

        return px1, py1, px2, py2

    def detect_poses(self, frames, player_tracks):
        """
        Detect pose keypoints for every tracked player in every frame.

        For each frame, crops every tracked player's padded bbox (skipping
        degenerate crops where the padded/clamped box has zero area), runs one
        batched `model.predict(crops, ...)` call across all of that frame's
        crops, then for each result picks the highest-`boxes.conf` detected
        person in that crop (there should normally be exactly one, but a
        partially-visible neighboring player at the crop edge can occasionally
        produce a second detection) and translates its keypoints from
        crop-local back to full-frame pixel coordinates.

        Args:
            frames (list): List of video frames (numpy arrays) to process.
            player_tracks (list): List of dictionaries (one per frame) mapping
                player track_id to {"bbox": [x1, y1, x2, y2]}, as produced by
                `PlayerTracker.get_object_tracks`.

        Returns:
            list: One dict per frame, indexed by player track_id. Each
                per-player dict has the shape:
                {
                    "keypoints": [[x, y, conf], ...],  # exactly 17 entries, COCO order
                    "bbox": [x1, y1, x2, y2],          # original (unpadded) player bbox
                    "crop_bbox": [px1, py1, px2, py2], # padded region fed to the model
                }
                A track_id with no valid crop this frame (degenerate bbox, or
                the pose model detected nobody in that crop) is omitted from
                that frame's dict.
        """
        pose_tracks = []

        for frame_num, frame in enumerate(frames):
            frame_h, frame_w = frame.shape[:2]
            player_dict = player_tracks[frame_num] if frame_num < len(player_tracks) else {}

            track_ids = []
            crops = []
            original_bboxes = []
            crop_bboxes = []

            for track_id, player in player_dict.items():
                bbox = player["bbox"]
                px1, py1, px2, py2 = self._pad_bbox(bbox, frame_w, frame_h)

                ipx1, ipy1, ipx2, ipy2 = int(px1), int(py1), int(px2), int(py2)
                if ipx2 <= ipx1 or ipy2 <= ipy1:
                    # Degenerate crop (fully out of frame bounds, or zero area
                    # after padding/clamping) -- skip this track_id entirely
                    # for this frame rather than feeding an empty array to the model.
                    continue

                crop = frame[ipy1:ipy2, ipx1:ipx2]

                track_ids.append(track_id)
                crops.append(crop)
                original_bboxes.append(bbox)
                crop_bboxes.append([px1, py1, px2, py2])

            frame_poses = {}

            if crops:
                # One batched predict() call across all of this frame's player
                # crops, matching PlayerTracker's batching pattern rather than
                # issuing N separate slow per-player calls.
                results = self.model.predict(crops, conf=self.conf, device=self.device, verbose=False)

                for idx, result in enumerate(results):
                    keypoints_xy = result.keypoints.xy
                    keypoints_conf = result.keypoints.conf
                    boxes_conf = result.boxes.conf

                    num_detections = len(boxes_conf) if boxes_conf is not None else 0
                    if num_detections == 0:
                        # The pose model detected nobody in this crop.
                        continue

                    # Pick the highest-confidence detected person in this crop.
                    # There should normally be exactly one, but a partially-visible
                    # neighboring player at the crop edge can occasionally produce
                    # a second detection.
                    best_idx = int(boxes_conf.argmax())

                    crop_x1, crop_y1, _, _ = crop_bboxes[idx]

                    keypoints = []
                    for kp_idx in range(NUM_COCO_KEYPOINTS):
                        x, y = keypoints_xy[best_idx][kp_idx].tolist()
                        conf = float(keypoints_conf[best_idx][kp_idx])
                        # Translate from crop-local back to full-frame pixel
                        # coordinates by adding back the crop's origin offset.
                        keypoints.append([x + crop_x1, y + crop_y1, conf])

                    track_id = track_ids[idx]
                    frame_poses[track_id] = {
                        "keypoints": keypoints,
                        "bbox": original_bboxes[idx],
                        "crop_bbox": crop_bboxes[idx],
                    }

            pose_tracks.append(frame_poses)

        return pose_tracks

    def get_object_poses(self, frames, player_tracks, read_from_stub=False, stub_path=None):
        """
        Get player pose estimation results for a sequence of frames with optional caching.

        Uses the same stub-caching contract as `PlayerTracker.get_object_tracks`:
        check the cache first, compute if missing or stale (frame count
        mismatch), and save at the end. Note that this stub's validity
        implicitly depends on `player_tracks` being stable between runs --
        this isn't a new caching invariant, just the same one downstream
        stages already assume for upstream tracks/detections.

        Args:
            frames (list): List of video frames to process.
            player_tracks (list): List of dictionaries (one per frame) mapping
                player track_id to {"bbox": [...]}.
            read_from_stub (bool): Whether to attempt reading cached results.
            stub_path (str): Path to the cache file.

        Returns:
            list: One dict per frame, indexed by player track_id (see
                `detect_poses` for the exact per-player shape).
        """
        pose_tracks = read_stub(read_from_stub, stub_path)
        if pose_tracks is not None:
            if len(pose_tracks) == len(frames):
                return pose_tracks

        pose_tracks = self.detect_poses(frames, player_tracks)

        save_stub(stub_path, pose_tracks)
        return pose_tracks
