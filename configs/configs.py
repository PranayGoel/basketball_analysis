STUBS_DEFAULT_PATH = 'stubs'
PLAYER_DETECTOR_PATH = 'models/player_detector.pt'
BALL_DETECTOR_PATH = 'models/ball_detector_model.pt'
COURT_KEYPOINT_DETECTOR_PATH = 'models/court_keypoint_detector.pt'
OUTPUT_VIDEO_PATH = 'output_videos/output_video.avi'

# Inference defaults (overridable via CLI flags in main.py)
DEVICE = 'auto'          # 'auto' resolves to cuda -> mps -> cpu; or force 'cpu'/'cuda'/'mps'
CONF_THRESHOLD = 0.5     # YOLO detection confidence threshold
BATCH_SIZE = 20          # frames per YOLO prediction batch