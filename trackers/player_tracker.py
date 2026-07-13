from ultralytics import YOLO
import supervision as sv
from personal.basketball_analysis.utils import read_stub, save_stub

class PlayerTracker:
    """
    A class that handles player detection and tracking using YOLO and ByteTrack.

    This class combines YOLO object detection with ByteTrack tracking to maintain consistent
    player identities across frames while processing detections in batches.
    """
    def __init__(self, model_path, device='cpu', conf=0.5, batch_size=20,
                 augment=False, half=False, video_fps=30.0,
                 track_activation_threshold=0.4, lost_track_buffer=60,
                 minimum_matching_threshold=0.75, minimum_consecutive_frames=2,
                 class_name='Player'):
        """
        Initialize the PlayerTracker with YOLO model and ByteTrack tracker.

        Args:
            model_path (str): Path to the YOLO model weights.
            device (str): Inference device passed to YOLO ('cpu', 'cuda', 'mps').
            conf (float): Detection confidence threshold.
            batch_size (int): Number of frames per prediction batch.
            augment (bool): Whether to enable YOLO test-time augmentation.
            half (bool): Whether to run inference in half precision (FP16).
            class_name (str): The model's class name to track as "player".
                Defaults to 'Player', matching this project's custom-trained
                detector. A generic COCO-pretrained fallback model (see
                pipeline/model_resolution.py) uses 'person' instead -- the
                class name is a property of which WEIGHTS are loaded, not of
                this tracker's logic, so it's a constructor param rather than
                a hardcoded string.
            video_fps (float): The source video's actual frame rate, passed to
                ByteTrack so `lost_track_buffer` (an occlusion-tolerance window
                measured in frames) corresponds to the real wall-clock duration
                intended -- previously this was never set, so ByteTrack silently
                assumed 30fps regardless of the source video's real fps.
            track_activation_threshold (float): Minimum detection confidence to
                start a brand-new track (raised above ByteTrack's 0.25 default
                to suppress spurious short-lived tracks from flicker detections,
                since PLAYER_CONF_THRESHOLD already filters low-confidence boxes).
            lost_track_buffer (int): Frames a lost track is kept alive awaiting
                re-matching before being dropped (doubled from ByteTrack's 30
                default to extend occlusion tolerance).
            minimum_matching_threshold (float): IoU threshold for re-matching a
                lost track (loosened slightly from ByteTrack's 0.8 default to
                help re-acquisition after a player moves during occlusion).
            minimum_consecutive_frames (int): Consecutive frames required before
                confirming a new track (raised from ByteTrack's 1 default to
                filter one-off flicker detections, e.g. a referee briefly
                misclassified as a player).
        """
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack(
            track_activation_threshold=track_activation_threshold,
            lost_track_buffer=lost_track_buffer,
            minimum_matching_threshold=minimum_matching_threshold,
            frame_rate=video_fps,
            minimum_consecutive_frames=minimum_consecutive_frames,
        )
        self.device = device
        self.conf = conf
        self.batch_size = batch_size
        self.augment = augment
        self.half = half
        self.class_name = class_name

    def detect_frames(self, frames):
        """
        Detect players in a sequence of frames using batch processing.

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

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        """
        Get player tracking results for a sequence of frames with optional caching.

        Args:
            frames (list): List of video frames to process.
            read_from_stub (bool): Whether to attempt reading cached results.
            stub_path (str): Path to the cache file.

        Returns:
            list: List of dictionaries containing player tracking information for each frame,
                where each dictionary maps player IDs to their bounding box coordinates.
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

            # Track Objects
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            tracks.append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_names_inv[self.class_name]:
                    tracks[frame_num][track_id] = {"bbox":bbox}
        
        save_stub(stub_path,tracks)
        return tracks
