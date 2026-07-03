import os
import argparse
import time
from utils import read_video, save_video, get_video_fps
from trackers import PlayerTracker, BallTracker
from team_assigner import TeamAssigner
from court_keypoint_detector import CourtKeypointDetector
from ball_aquisition import BallAquisitionDetector
from pass_and_interception_detector import PassAndInterceptionDetector
from tactical_view_converter import TacticalViewConverter
from speed_and_distance_calculator import SpeedAndDistanceCalculator
from drawers import (
    PlayerTracksDrawer, 
    BallTracksDrawer,
    CourtKeypointDrawer,
    TeamBallControlDrawer,
    FrameNumberDrawer,
    PassInterceptionDrawer,
    TacticalViewDrawer,
    SpeedAndDistanceDrawer
)
from configs import(
    STUBS_DEFAULT_PATH,
    PLAYER_DETECTOR_PATH,
    BALL_DETECTOR_PATH,
    COURT_KEYPOINT_DETECTOR_PATH,
    OUTPUT_VIDEO_PATH,
    DEVICE,
    CONF_THRESHOLD,
    BATCH_SIZE
)


def resolve_device(choice):
    """Resolve 'auto' to the best available backend (cuda -> mps -> cpu).

    Any explicit value ('cpu', 'cuda', 'mps') is passed through unchanged.
    """
    if choice != 'auto':
        return choice
    try:
        import torch
        if torch.cuda.is_available():
            return 'cuda'
        if getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
            return 'mps'
    except Exception:
        pass
    return 'cpu'


def parse_args():
    parser = argparse.ArgumentParser(description='Basketball Video Analysis')
    parser.add_argument('input_video', type=str, help='Path to input video file')
    parser.add_argument('--output_video', type=str, default=OUTPUT_VIDEO_PATH, 
                        help='Path to output video file')
    parser.add_argument('--stub_path', type=str, default=STUBS_DEFAULT_PATH,
                        help='Path to stub directory')
    parser.add_argument('--device', type=str, default=DEVICE,
                        choices=['auto', 'cpu', 'cuda', 'mps'],
                        help="Inference device; 'auto' picks cuda -> mps -> cpu")
    parser.add_argument('--conf', type=float, default=CONF_THRESHOLD,
                        help='YOLO detection confidence threshold')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                        help='Number of frames per YOLO prediction batch')
    parser.add_argument('--no-stubs', dest='use_stubs', action='store_false',
                        help='Ignore cached stub files and recompute all detections')
    parser.add_argument('--profile', action='store_true',
                        help='Print per-stage wall-clock timings and end-to-end FPS')
    parser.add_argument('--report', type=str, default=None,
                        help='Path to write a post-game JSON analytics report (see game_report.py)')
    parser.set_defaults(use_stubs=True)
    return parser.parse_args()

def main():
    args = parse_args()

    device = resolve_device(args.device)
    print(f"[config] device={device} conf={args.conf} batch_size={args.batch_size} use_stubs={args.use_stubs}")

    t0 = time.perf_counter()

    # Read Video
    video_frames = read_video(args.input_video)
    source_fps = get_video_fps(args.input_video)
    t_read = time.perf_counter()

    ## Initialize Tracker
    player_tracker = PlayerTracker(PLAYER_DETECTOR_PATH, device=device, conf=args.conf, batch_size=args.batch_size)
    ball_tracker = BallTracker(BALL_DETECTOR_PATH, device=device, conf=args.conf, batch_size=args.batch_size)

    ## Initialize Keypoint Detector
    court_keypoint_detector = CourtKeypointDetector(COURT_KEYPOINT_DETECTOR_PATH, device=device, conf=args.conf, batch_size=args.batch_size)

    # Run Detectors
    player_tracks = player_tracker.get_object_tracks(video_frames,
                                       read_from_stub=args.use_stubs,
                                       stub_path=os.path.join(args.stub_path, 'player_track_stubs.pkl')
                                      )

    ball_tracks = ball_tracker.get_object_tracks(video_frames,
                                                 read_from_stub=args.use_stubs,
                                                 stub_path=os.path.join(args.stub_path, 'ball_track_stubs.pkl')
                                                )
    ## Run KeyPoint Extractor
    court_keypoints_per_frame = court_keypoint_detector.get_court_keypoints(video_frames,
                                                                    read_from_stub=args.use_stubs,
                                                                    stub_path=os.path.join(args.stub_path, 'court_key_points_stub.pkl')
                                                                    )
    t_detect = time.perf_counter()

    # Remove Wrong Ball Detections
    ball_tracks = ball_tracker.remove_wrong_detections(ball_tracks)
    # Interpolate Ball Tracks
    ball_tracks = ball_tracker.interpolate_ball_positions(ball_tracks)
    t_ball_cleanup = time.perf_counter()

    # Assign Player Teams
    team_assigner = TeamAssigner()
    player_assignment = team_assigner.get_player_teams_across_frames(video_frames,
                                                                    player_tracks,
                                                                    read_from_stub=args.use_stubs,
                                                                    stub_path=os.path.join(args.stub_path, 'player_assignment_stub.pkl')
                                                                    )
    t_team_assignment = time.perf_counter()

    # Ball Acquisition
    ball_aquisition_detector = BallAquisitionDetector()
    ball_aquisition = ball_aquisition_detector.detect_ball_possession(player_tracks,ball_tracks)
    t_ball_acquisition = time.perf_counter()

    # Detect Passes
    pass_and_interception_detector = PassAndInterceptionDetector()
    passes = pass_and_interception_detector.detect_passes(ball_aquisition,player_assignment)
    interceptions = pass_and_interception_detector.detect_interceptions(ball_aquisition,player_assignment)
    t_pass_interception = time.perf_counter()

    # Tactical View
    tactical_view_converter = TacticalViewConverter(
        court_image_path="./images/basketball_court.png"
    )

    court_keypoints_per_frame = tactical_view_converter.validate_keypoints(court_keypoints_per_frame)
    tactical_player_positions = tactical_view_converter.transform_players_to_tactical_view(court_keypoints_per_frame,player_tracks)
    t_tactical_view = time.perf_counter()

    # Speed and Distance Calculator
    speed_and_distance_calculator = SpeedAndDistanceCalculator(
        tactical_view_converter.width,
        tactical_view_converter.height,
        tactical_view_converter.actual_width_in_meters,
        tactical_view_converter.actual_height_in_meters
    )
    player_distances_per_frame = speed_and_distance_calculator.calculate_distance(tactical_player_positions)
    player_speed_per_frame = speed_and_distance_calculator.calculate_speed(player_distances_per_frame, fps=source_fps)
    t_speed_distance = time.perf_counter()

    # Draw output   
    # Initialize Drawers
    player_tracks_drawer = PlayerTracksDrawer()
    ball_tracks_drawer = BallTracksDrawer()
    court_keypoint_drawer = CourtKeypointDrawer()
    team_ball_control_drawer = TeamBallControlDrawer()
    frame_number_drawer = FrameNumberDrawer()
    pass_and_interceptions_drawer = PassInterceptionDrawer()
    tactical_view_drawer = TacticalViewDrawer()
    speed_and_distance_drawer = SpeedAndDistanceDrawer()

    ## Draw object Tracks
    output_video_frames = player_tracks_drawer.draw(video_frames, 
                                                    player_tracks,
                                                    player_assignment,
                                                    ball_aquisition)
    output_video_frames = ball_tracks_drawer.draw(output_video_frames, ball_tracks)

    ## Draw KeyPoints
    output_video_frames = court_keypoint_drawer.draw(output_video_frames, court_keypoints_per_frame)

    ## Draw Frame Number
    output_video_frames = frame_number_drawer.draw(output_video_frames)

    # Draw Team Ball Control
    output_video_frames = team_ball_control_drawer.draw(output_video_frames,
                                                        player_assignment,
                                                        ball_aquisition)

    # Draw Passes and Interceptions
    output_video_frames = pass_and_interceptions_drawer.draw(output_video_frames,
                                                             passes,
                                                             interceptions)
    
    # Speed and Distance Drawer
    output_video_frames = speed_and_distance_drawer.draw(output_video_frames,
                                                         player_tracks,
                                                         player_distances_per_frame,
                                                         player_speed_per_frame
                                                         )

    ## Draw Tactical View
    output_video_frames = tactical_view_drawer.draw(output_video_frames,
                                                    tactical_view_converter.court_image_path,
                                                    tactical_view_converter.width,
                                                    tactical_view_converter.height,
                                                    tactical_view_converter.key_points,
                                                    tactical_player_positions,
                                                    player_assignment,
                                                    ball_aquisition,
                                                    )

    t_draw = time.perf_counter()

    # Save video
    save_video(output_video_frames, args.output_video, fps=source_fps)
    t_save = time.perf_counter()

    if args.report:
        # team_ball_control was previously computed only inside TeamBallControlDrawer and
        # discarded after rendering; compute it once here so the report can use it too.
        team_ball_control = team_ball_control_drawer.get_team_ball_control(player_assignment, ball_aquisition)
        from game_report import build_game_report
        report = build_game_report(
            player_assignment=player_assignment,
            ball_aquisition=ball_aquisition,
            passes=passes,
            interceptions=interceptions,
            tactical_player_positions=tactical_player_positions,
            player_distances_per_frame=player_distances_per_frame,
            player_speed_per_frame=player_speed_per_frame,
            team_ball_control=team_ball_control,
        )
        import json
        with open(args.report, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"[report] wrote game report to {args.report}")

    if args.profile:
        stages = [
            ("read_video", t_read - t0),
            ("detection", t_detect - t_read),
            ("ball_cleanup", t_ball_cleanup - t_detect),
            ("team_assign", t_team_assignment - t_ball_cleanup),
            ("ball_acq", t_ball_acquisition - t_team_assignment),
            ("pass_intercept", t_pass_interception - t_ball_acquisition),
            ("tactical_view", t_tactical_view - t_pass_interception),
            ("speed_distance", t_speed_distance - t_tactical_view),
            ("draw", t_draw - t_speed_distance),
            ("save", t_save - t_draw),
        ]
        total = t_save - t0
        num_frames = len(video_frames)
        print("\n[profile] stage timings (seconds):")
        for name, secs in stages:
            print(f"  {name:15s} {secs:8.2f}")
        print(f"  {'TOTAL':15s} {total:8.2f}")
        if total > 0:
            print(f"[profile] {num_frames} frames -> {num_frames / total:.2f} FPS end-to-end")

if __name__ == '__main__':
    main()
    