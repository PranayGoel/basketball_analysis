from ultralytics import YOLO
import supervision as sv
import numpy as np
import pandas as pd
import sys
sys.path.append('../')
from utils import read_stub, save_stub
from trackers.kalman_ball_tracker import ConstantVelocityKalmanFilter2D


class BallTracker:
    """
    A class that handles basketball detection and tracking using YOLO.

    This class provides methods to detect the ball in video frames, process detections
    in batches, and refine tracking results through filtering and interpolation.
    """
    def __init__(self, model_path, device='cpu', conf=0.5, batch_size=20,
                 augment=False, half=False,
                 kf_process_noise=5.0, kf_measurement_noise=10.0,
                 max_reasonable_distance_per_frame=60.0,
                 class_name='Ball'):
        """
        Args:
            model_path (str): Path to the YOLO model weights.
            device (str): Inference device passed to YOLO ('cpu', 'cuda', 'mps').
            conf (float): Detection confidence threshold.
            batch_size (int): Number of frames per prediction batch.
            augment (bool): Whether to enable YOLO test-time augmentation.
            half (bool): Whether to run inference in half precision (FP16).
            kf_process_noise (float): Process noise passed to the Kalman filter
                used by `get_object_tracks_with_kalman`.
            kf_measurement_noise (float): Measurement noise passed to the Kalman
                filter used by `get_object_tracks_with_kalman`.
            max_reasonable_distance_per_frame (float): Maximum pixel distance a
                chosen candidate's center may be from the Kalman-predicted
                position in a single frame before it's rejected as implausible.
            class_name (str): The model's class name to track as "the ball".
                Defaults to 'Ball', matching this project's custom-trained
                detector. A generic COCO-pretrained fallback model (see
                pipeline/model_resolution.py) uses 'sports ball' instead.
        """
        self.model = YOLO(model_path)
        self.device = device
        self.conf = conf
        self.batch_size = batch_size
        self.augment = augment
        self.half = half
        self.kf_process_noise = kf_process_noise
        self.kf_measurement_noise = kf_measurement_noise
        self.max_reasonable_distance_per_frame = max_reasonable_distance_per_frame
        self.class_name = class_name

    def detect_frames(self, frames):
        """
        Detect the ball in a sequence of frames using batch processing.

        Args:
            frames (list): List of video frames to process.

        Returns:
            list: YOLO detection results for each frame.
        """
        detections = []
        for i in range(0, len(frames), self.batch_size):
            detections_batch = self.model.predict(
                frames[i:i+self.batch_size], conf=self.conf, device=self.device,
                augment=self.augment, half=self.half
            )
            detections += detections_batch
        return detections

    def get_all_ball_candidates(self, frames):
        """
        Detect ALL "Ball"-class candidates per frame, without collapsing to a
        single max-confidence pick.

        Unlike `get_object_tracks`, this preserves every YOLO "Ball" detection
        so that downstream motion-aware selection (see
        `get_object_tracks_with_kalman`) can choose among multiple candidates
        using more than raw confidence (e.g. a scoreboard logo or kneepad that
        YOLO misclassifies as "Ball" with lower confidence than the real ball).

        Args:
            frames (list): List of video frames to process.

        Returns:
            list[list[dict]]: Outer list index = frame number. Each inner list
                holds zero or more `{'bbox': [x1, y1, x2, y2], 'confidence': float}`
                dicts, one per detected "Ball"-class candidate in that frame.
        """
        detections = self.detect_frames(frames)

        all_candidates = []

        for detection in detections:
            cls_names = detection.names
            cls_names_inv = {v: k for k, v in cls_names.items()}

            detection_supervision = sv.Detections.from_ultralytics(detection)

            frame_candidates = []
            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                confidence = frame_detection[2]

                if cls_id == cls_names_inv[self.class_name]:
                    frame_candidates.append({'bbox': bbox, 'confidence': float(confidence)})

            all_candidates.append(frame_candidates)

        return all_candidates

    def get_object_tracks_with_kalman(self, frames, read_from_stub=False, stub_path=None,
                                       confidence_closeness_margin=0.15):
        """
        Get ball tracking results using a Kalman-filter-gated single-pass pipeline.

        This REPLACES the old `get_object_tracks()` -> `remove_wrong_detections()`
        -> `interpolate_ball_positions()` chain. Instead of always trusting the
        single highest-confidence detection per frame and then linearly
        interpolating gaps, this tracks a constant-velocity motion model of the
        ball's bbox center and uses it to:
          - disambiguate between multiple same-confidence-ish candidates (pick
            the one consistent with the ball's recent motion), and
          - reject implausible high-confidence outliers (e.g. a misclassified
            scoreboard logo) that don't fit the motion model, and
          - reconstruct a plausible position during brief occlusions using the
            predicted (not linearly-interpolated) trajectory, which tolerates
            arcing/bouncing motion far better than a straight-line fill.

        Args:
            frames (list): List of video frames to process.
            read_from_stub (bool): Whether to attempt reading cached results.
            stub_path (str): Path to the cache file.
            confidence_closeness_margin (float): If the top-2 candidates by
                confidence in a frame are within this margin of each other,
                motion-consistency (closeness to the Kalman prediction) is used
                to break the tie instead of blindly trusting confidence.

        Returns:
            list[dict]: One dict per frame. Each non-empty dict maps track id 1
                to `{"bbox": [x1, y1, x2, y2], "source": "detection"|"predicted"}`.
                Frames with neither a surviving detection nor prior Kalman state
                are empty dicts `{}`. This is backward-compatible with the shape
                produced by the old `interpolate_ball_positions()` -- the only
                addition is the `"source"` key; `track[1]["bbox"]` keeps working
                unchanged.
        """
        tracks = read_stub(read_from_stub, stub_path)
        if tracks is not None:
            if len(tracks) == len(frames):
                return tracks

        all_candidates = self.get_all_ball_candidates(frames)

        kf = ConstantVelocityKalmanFilter2D(
            process_noise=self.kf_process_noise,
            measurement_noise=self.kf_measurement_noise,
        )

        last_smoothed_width = None
        last_smoothed_height = None

        tracks = []

        for frame_candidates in all_candidates:
            predicted_cx, predicted_cy = None, None
            if kf.initialized:
                predicted_cx, predicted_cy = kf.predict()

            chosen_candidate = self._select_candidate(
                frame_candidates, predicted_cx, predicted_cy, kf.initialized,
                confidence_closeness_margin,
            )

            if chosen_candidate is not None and kf.initialized:
                cx, cy = self._bbox_center(chosen_candidate['bbox'])
                distance = float(np.linalg.norm(
                    np.array([cx, cy]) - np.array([predicted_cx, predicted_cy])
                ))
                if distance > self.max_reasonable_distance_per_frame:
                    chosen_candidate = None

            if chosen_candidate is not None:
                bbox = chosen_candidate['bbox']
                cx, cy = self._bbox_center(bbox)
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]

                if last_smoothed_width is None:
                    last_smoothed_width, last_smoothed_height = width, height
                else:
                    last_smoothed_width = 0.7 * last_smoothed_width + 0.3 * width
                    last_smoothed_height = 0.7 * last_smoothed_height + 0.3 * height

                if not kf.initialized:
                    kf.initialize(cx, cy)
                else:
                    kf.update(cx, cy)

                tracks.append({1: {"bbox": bbox, "source": "detection"}})

            elif kf.initialized and last_smoothed_width is not None:
                reconstructed_bbox = self._bbox_from_center(
                    predicted_cx, predicted_cy, last_smoothed_width, last_smoothed_height
                )
                tracks.append({1: {"bbox": reconstructed_bbox, "source": "predicted"}})

            else:
                tracks.append({})

        self._backfill_leading_gap(tracks)

        if stub_path is not None:
            save_stub(stub_path, tracks)

        return tracks

    @staticmethod
    def _bbox_center(bbox):
        """Return the (cx, cy) center of an [x1, y1, x2, y2] bbox."""
        x1, y1, x2, y2 = bbox
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    @staticmethod
    def _bbox_from_center(cx, cy, width, height):
        """Reconstruct an [x1, y1, x2, y2] bbox centered at (cx, cy) with the given size."""
        half_w = width / 2.0
        half_h = height / 2.0
        return [cx - half_w, cy - half_h, cx + half_w, cy + half_h]

    def _select_candidate(self, frame_candidates, predicted_cx, predicted_cy,
                           kf_initialized, confidence_closeness_margin):
        """
        Choose which candidate (if any) represents the ball in this frame.

        With 2+ candidates and an initialized filter, if the top-2 candidates by
        confidence are within `confidence_closeness_margin` of each other, picks
        whichever of the top-3 (or fewer) candidates by confidence is closest to
        the Kalman-predicted position. Otherwise falls back to max confidence.

        Args:
            frame_candidates (list[dict]): This frame's `{'bbox', 'confidence'}` candidates.
            predicted_cx (float | None): Kalman-predicted center x, or None if uninitialized.
            predicted_cy (float | None): Kalman-predicted center y, or None if uninitialized.
            kf_initialized (bool): Whether the Kalman filter has prior state.
            confidence_closeness_margin (float): Confidence-tie threshold.

        Returns:
            dict | None: The chosen `{'bbox', 'confidence'}` candidate, or None if
                `frame_candidates` is empty.
        """
        if not frame_candidates:
            return None

        candidates_by_confidence = sorted(
            frame_candidates, key=lambda c: c['confidence'], reverse=True
        )

        if len(candidates_by_confidence) == 1 or not kf_initialized:
            return candidates_by_confidence[0]

        top_confidence = candidates_by_confidence[0]['confidence']
        second_confidence = candidates_by_confidence[1]['confidence']

        if (top_confidence - second_confidence) > confidence_closeness_margin:
            return candidates_by_confidence[0]

        # Confidences are close -- break the tie using motion consistency among
        # the top-3 (or fewer) candidates.
        contenders = candidates_by_confidence[:3]
        predicted = np.array([predicted_cx, predicted_cy])

        def distance_to_prediction(candidate):
            cx, cy = self._bbox_center(candidate['bbox'])
            return float(np.linalg.norm(np.array([cx, cy]) - predicted))

        return min(contenders, key=distance_to_prediction)

    @staticmethod
    def _backfill_leading_gap(tracks):
        """
        Backfill any leading gap (frames before the first non-empty entry) with
        that first available track entry, mirroring the old code's
        `df.bfill()` behavior for leading NaNs. Mutates `tracks` in place.
        """
        first_non_empty_index = None
        for i, track in enumerate(tracks):
            if track:
                first_non_empty_index = i
                break

        if first_non_empty_index is None or first_non_empty_index == 0:
            return

        first_entry = tracks[first_non_empty_index]
        for i in range(first_non_empty_index):
            tracks[i] = dict(first_entry)

    # DEPRECATED — superseded by get_object_tracks_with_kalman(); retained so scripts/benchmark_detection.py can still run the old behavior for before/after comparison.
    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        """
        Get ball tracking results for a sequence of frames with optional caching.

        Args:
            frames (list): List of video frames to process.
            read_from_stub (bool): Whether to attempt reading cached results.
            stub_path (str): Path to the cache file.

        Returns:
            list: List of dictionaries containing ball tracking information for each frame.
        """
        tracks = read_stub(read_from_stub,stub_path)
        if tracks is not None:
            if len(tracks) == len(frames):
                return tracks

        detections = self.detect_frames(frames)

        tracks=[]

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v:k for k,v in cls_names.items()}

            # Covert to supervision Detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)

            tracks.append({})
            chosen_bbox =None
            max_confidence = 0
            
            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                confidence = frame_detection[2]
                
                if cls_id == cls_names_inv[self.class_name]:
                    if max_confidence<confidence:
                        chosen_bbox = bbox
                        max_confidence = confidence

            if chosen_bbox is not None:
                tracks[frame_num][1] = {"bbox":chosen_bbox}

        save_stub(stub_path,tracks)
        
        return tracks

    # DEPRECATED — superseded by get_object_tracks_with_kalman(); retained so scripts/benchmark_detection.py can still run the old behavior for before/after comparison.
    def remove_wrong_detections(self,ball_positions):
        """
        Filter out incorrect ball detections based on maximum allowed movement distance.

        Args:
            ball_positions (list): List of detected ball positions across frames.

        Returns:
            list: Filtered ball positions with incorrect detections removed.
        """
        
        maximum_allowed_distance = 25
        last_good_frame_index = -1

        for i in range(len(ball_positions)):
            current_box = ball_positions[i].get(1, {}).get('bbox', [])

            if len(current_box) == 0:
                continue

            if last_good_frame_index == -1:
                # First valid detection
                last_good_frame_index = i
                continue

            last_good_box = ball_positions[last_good_frame_index].get(1, {}).get('bbox', [])
            frame_gap = i - last_good_frame_index
            adjusted_max_distance = maximum_allowed_distance * frame_gap

            if np.linalg.norm(np.array(last_good_box[:2]) - np.array(current_box[:2])) > adjusted_max_distance:
                ball_positions[i] = {}
            else:
                last_good_frame_index = i

        return ball_positions

    # DEPRECATED — superseded by get_object_tracks_with_kalman(); retained so scripts/benchmark_detection.py can still run the old behavior for before/after comparison.
    def interpolate_ball_positions(self,ball_positions):
        """
        Interpolate missing ball positions to create smooth tracking results.

        Args:
            ball_positions (list): List of ball positions with potential gaps.

        Returns:
            list: List of ball positions with interpolated values filling the gaps.
        """
        ball_positions = [x.get(1,{}).get('bbox',[]) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions,columns=['x1','y1','x2','y2'])

        # Interpolate missing values
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [{1: {"bbox":x}} for x in df_ball_positions.to_numpy().tolist()]
        return ball_positions