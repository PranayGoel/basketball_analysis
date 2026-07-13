"""
run_analysis(): the single source of truth for the full CV pipeline.

This is the exact pipeline body that used to live inline in main.py's main()
function, extracted so both the CLI (main.py, now a thin argparse shim) and
the web backend (webapp/backend/app/worker/tasks.py) call the identical
function -- one implementation, not a forked copy kept in sync by hand.

run_analysis() is intentionally side-effect-light: it doesn't print anything
itself (the CLI does that with the profile timings it can compute from its
own perf_counter() calls around this function, or via progress_callback for
live per-stage feedback), and it always builds+returns the game_report dict
regardless of whether the caller wants it persisted to disk (report_path is
optional -- the web backend always wants the dict back to index into its DB,
even if it doesn't care about a JSON file on disk).
"""

import os

from personal.basketball_analysis.utils import read_video, save_video, get_video_fps
from personal.basketball_analysis.trackers import PlayerTracker, BallTracker
from personal.basketball_analysis.team_assigner import TeamAssigner
from personal.basketball_analysis.court_keypoint_detector import CourtKeypointDetector
from personal.basketball_analysis.ball_aquisition import BallAquisitionDetector
from personal.basketball_analysis.pass_and_interception_detector import PassAndInterceptionDetector
from personal.basketball_analysis.tactical_view_converter import TacticalViewConverter
from personal.basketball_analysis.speed_and_distance_calculator import SpeedAndDistanceCalculator
from personal.basketball_analysis.pose_estimator import PoseEstimator
from personal.basketball_analysis.rule_violation_detector import DoubleDribbleDetector, TravelingDetector
from personal.basketball_analysis.drawers import (
    PlayerTracksDrawer,
    BallTracksDrawer,
    CourtKeypointDrawer,
    TeamBallControlDrawer,
    FrameNumberDrawer,
    PassInterceptionDrawer,
    TacticalViewDrawer,
    SpeedAndDistanceDrawer,
    PoseDrawer,
    ViolationDrawer,
)
from personal.basketball_analysis.configs import (
    PLAYER_CONF_THRESHOLD,
    BALL_CONF_THRESHOLD,
    COURT_KEYPOINT_CONF_THRESHOLD,
    PLAYER_AUGMENT,
    BALL_AUGMENT,
    COURT_KEYPOINT_AUGMENT,
    BALL_KF_PROCESS_NOISE,
    BALL_KF_MEASUREMENT_NOISE,
    BALL_KF_MAX_REASONABLE_DISTANCE_PER_FRAME,
    PLAYER_TRACK_ACTIVATION_THRESHOLD,
    PLAYER_LOST_TRACK_BUFFER,
    PLAYER_MINIMUM_MATCHING_THRESHOLD,
    PLAYER_MINIMUM_CONSECUTIVE_FRAMES,
)
from personal.basketball_analysis.game_report import build_game_report

from personal.basketball_analysis.pipeline.progress import ProgressReporter
from personal.basketball_analysis.pipeline.model_resolution import (
    resolve_player_model,
    resolve_ball_model,
    resolve_court_keypoint_model,
    describe_resolution,
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


def run_analysis(config, progress_callback=None):
    """
    Run the full basketball video analysis pipeline end to end.

    Args:
        config (pipeline.config.PipelineConfig): every pipeline knob, already
            resolved to concrete values by the caller (the CLI resolves its
            --conf/--player-conf/etc. precedence before constructing this; the
            web backend just passes its request's explicit values directly).
        progress_callback (callable, optional): forwarded to ProgressReporter
            -- called as (stage_name, stage_index, total_stages, status) at
            the start and end of each pipeline stage. None means no progress
            reporting (e.g. a quick script that doesn't care).

    Returns:
        dict: the game_report dict (see game_report.build_game_report), always
            built and returned regardless of whether config.report_path was
            set. If config.report_path is set, the same dict is also written
            there as JSON.
    """
    reporter = ProgressReporter(progress_callback)

    device = resolve_device(config.device)
    player_conf = config.player_conf if config.player_conf is not None else PLAYER_CONF_THRESHOLD
    ball_conf = config.ball_conf if config.ball_conf is not None else BALL_CONF_THRESHOLD
    court_conf = config.court_conf if config.court_conf is not None else COURT_KEYPOINT_CONF_THRESHOLD

    with reporter.stage("read_video"):
        video_frames = read_video(config.video_path)
        source_fps = get_video_fps(config.video_path)

    # Model construction (YOLO weight loading) happens INSIDE each stage's
    # `with` block, not before it -- a stage boundary should own everything
    # that can fail as part of that stage, including loading its own model.
    # Previously the constructors ran between stage blocks, so a missing-
    # weights error surfaced as "failed during the *previous* stage" (e.g.
    # "read_video") instead of the stage that actually failed -- caught via
    # the live web-platform smoke test, not a unit test.
    #
    # Every model is resolved (not hardcoded to a path) so the pipeline can
    # run on a generic COCO fallback when the real custom-trained weights
    # aren't present (see pipeline/model_resolution.py) -- and always prints
    # which source it's actually using, since a silent substitution would
    # contradict this project's own honesty culture (the LLM section's
    # "honest gap, not a claim," the pose section's "worth a human reviewing
    # this clip").
    with reporter.stage("player_tracking"):
        player_resolution = resolve_player_model()
        print(describe_resolution("player", player_resolution))
        player_tracker = PlayerTracker(
            player_resolution.weights_path, device=device, conf=player_conf, batch_size=config.batch_size,
            augment=PLAYER_AUGMENT, half=config.half, video_fps=source_fps,
            track_activation_threshold=PLAYER_TRACK_ACTIVATION_THRESHOLD,
            lost_track_buffer=PLAYER_LOST_TRACK_BUFFER,
            minimum_matching_threshold=PLAYER_MINIMUM_MATCHING_THRESHOLD,
            minimum_consecutive_frames=PLAYER_MINIMUM_CONSECUTIVE_FRAMES,
            class_name=player_resolution.class_name,
        )
        player_tracks = player_tracker.get_object_tracks(
            video_frames,
            read_from_stub=config.use_stubs,
            stub_path=os.path.join(config.stub_dir, 'player_track_stubs.pkl'),
        )

    with reporter.stage("ball_tracking"):
        ball_resolution = resolve_ball_model()
        print(describe_resolution("ball", ball_resolution))
        ball_tracker = BallTracker(
            ball_resolution.weights_path, device=device, conf=ball_conf, batch_size=config.batch_size,
            augment=BALL_AUGMENT, half=config.half,
            kf_process_noise=BALL_KF_PROCESS_NOISE,
            kf_measurement_noise=BALL_KF_MEASUREMENT_NOISE,
            max_reasonable_distance_per_frame=BALL_KF_MAX_REASONABLE_DISTANCE_PER_FRAME,
            class_name=ball_resolution.class_name,
        )
        ball_tracks = ball_tracker.get_object_tracks_with_kalman(
            video_frames,
            read_from_stub=config.use_stubs,
            stub_path=os.path.join(config.stub_dir, 'ball_track_stubs.pkl'),
        )

    # Court keypoints have no generic fallback (a bespoke task with no COCO
    # equivalent) -- resolve_court_keypoint_model() returns None when the real
    # weights aren't present, and everything downstream that depends on court
    # keypoints (tactical/bird's-eye view, real-world-units speed & distance)
    # degrades cleanly rather than crashing or fabricating numbers. Player/ball
    # tracking, team assignment, possession, passes, pose, and violations are
    # all fully independent of this and run exactly as normal either way.
    court_keypoints_per_frame = None
    with reporter.stage("court_keypoints"):
        court_resolution = resolve_court_keypoint_model()
        print(describe_resolution("court_keypoints", court_resolution))
        if court_resolution is not None:
            court_keypoint_detector = CourtKeypointDetector(
                court_resolution.weights_path, device=device, conf=court_conf, batch_size=config.batch_size,
                augment=COURT_KEYPOINT_AUGMENT, half=config.half,
            )
            court_keypoints_per_frame = court_keypoint_detector.get_court_keypoints(
                video_frames,
                read_from_stub=config.use_stubs,
                stub_path=os.path.join(config.stub_dir, 'court_key_points_stub.pkl'),
            )

    with reporter.stage("team_assignment"):
        team_assigner = TeamAssigner()
        player_assignment = team_assigner.get_player_teams_across_frames(
            video_frames, player_tracks,
            read_from_stub=config.use_stubs,
            stub_path=os.path.join(config.stub_dir, 'player_assignment_stub.pkl'),
        )

    with reporter.stage("ball_acquisition"):
        ball_aquisition_detector = BallAquisitionDetector()
        ball_aquisition = ball_aquisition_detector.detect_ball_possession(player_tracks, ball_tracks)

    # pose_tracks/violations stay None when config.pose is off, so
    # build_game_report() below correctly omits the "violations" key entirely
    # (distinct from an empty list, which means this ran and found nothing).
    pose_tracks = None
    violations = None
    with reporter.stage("pose_violations"):
        if config.pose:
            pose_estimator = PoseEstimator(model_path=config.pose_model, device=device, conf=player_conf)
            pose_tracks = pose_estimator.get_object_poses(
                video_frames, player_tracks,
                read_from_stub=config.use_stubs,
                stub_path=os.path.join(config.stub_dir, 'pose_stub.pkl'),
            )
            violations = (
                DoubleDribbleDetector().detect(ball_tracks, ball_aquisition, pose_tracks)
                + TravelingDetector().detect(ball_aquisition, pose_tracks, ball_tracks)
            )

    with reporter.stage("pass_interception"):
        pass_and_interception_detector = PassAndInterceptionDetector()
        passes = pass_and_interception_detector.detect_passes(ball_aquisition, player_assignment)
        interceptions = pass_and_interception_detector.detect_interceptions(ball_aquisition, player_assignment)

    # Degraded mode (no court keypoints): tactical_view_converter stays None,
    # and tactical_player_positions is a correctly-shaped list of empty
    # per-frame dicts -- game_report.py's compute_player_movement_stats()
    # already handles empty per-frame dicts gracefully (verified by reading
    # its loop logic: it just never populates a player_id, no crash), so the
    # final report is honest (zeroed movement stats, since there's no
    # real-world coordinate system to compute them from) rather than crashing
    # or fabricating numbers.
    tactical_view_converter = None
    with reporter.stage("tactical_view"):
        if court_keypoints_per_frame is not None:
            tactical_view_converter = TacticalViewConverter(
                court_image_path=os.path.join(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "images", "basketball_court.png",
                )
            )
            court_keypoints_per_frame = tactical_view_converter.validate_keypoints(court_keypoints_per_frame)
            tactical_player_positions = tactical_view_converter.transform_players_to_tactical_view(
                court_keypoints_per_frame, player_tracks
            )
        else:
            tactical_player_positions = [{} for _ in range(len(video_frames))]

    with reporter.stage("speed_distance"):
        if tactical_view_converter is not None:
            speed_and_distance_calculator = SpeedAndDistanceCalculator(
                tactical_view_converter.width,
                tactical_view_converter.height,
                tactical_view_converter.actual_width_in_meters,
                tactical_view_converter.actual_height_in_meters,
            )
            player_distances_per_frame = speed_and_distance_calculator.calculate_distance(tactical_player_positions)
            player_speed_per_frame = speed_and_distance_calculator.calculate_speed(player_distances_per_frame, fps=source_fps)
        else:
            player_distances_per_frame = [{} for _ in range(len(video_frames))]
            player_speed_per_frame = [{} for _ in range(len(video_frames))]

    with reporter.stage("drawing"):
        player_tracks_drawer = PlayerTracksDrawer()
        ball_tracks_drawer = BallTracksDrawer()
        court_keypoint_drawer = CourtKeypointDrawer()
        team_ball_control_drawer = TeamBallControlDrawer()
        frame_number_drawer = FrameNumberDrawer()
        pass_and_interceptions_drawer = PassInterceptionDrawer()
        tactical_view_drawer = TacticalViewDrawer()
        speed_and_distance_drawer = SpeedAndDistanceDrawer()
        pose_drawer = PoseDrawer()
        violation_drawer = ViolationDrawer()

        output_video_frames = player_tracks_drawer.draw(
            video_frames, player_tracks, player_assignment, ball_aquisition
        )
        output_video_frames = ball_tracks_drawer.draw(output_video_frames, ball_tracks)
        output_video_frames = frame_number_drawer.draw(output_video_frames)
        output_video_frames = team_ball_control_drawer.draw(output_video_frames, player_assignment, ball_aquisition)
        output_video_frames = pass_and_interceptions_drawer.draw(output_video_frames, passes, interceptions)

        # Court-keypoint-dependent overlays (minimap corners, bird's-eye
        # tactical view, real-world-units speed/distance labels) are skipped
        # entirely in degraded mode -- there's no keypoint/tactical data to
        # draw, and these three drawers all assume it exists.
        if court_keypoints_per_frame is not None:
            output_video_frames = court_keypoint_drawer.draw(output_video_frames, court_keypoints_per_frame)
        if tactical_view_converter is not None:
            output_video_frames = speed_and_distance_drawer.draw(
                output_video_frames, player_tracks, player_distances_per_frame, player_speed_per_frame
            )
            output_video_frames = tactical_view_drawer.draw(
                output_video_frames,
                tactical_view_converter.court_image_path,
                tactical_view_converter.width,
                tactical_view_converter.height,
                tactical_view_converter.key_points,
                tactical_player_positions,
                player_assignment,
                ball_aquisition,
            )
        if config.pose:
            output_video_frames = pose_drawer.draw(output_video_frames, pose_tracks)
            output_video_frames = violation_drawer.draw(output_video_frames, violations, pose_tracks)

    with reporter.stage("save_video"):
        save_video(output_video_frames, config.output_video_path, fps=source_fps)

    with reporter.stage("report"):
        # team_ball_control was previously computed only inside
        # TeamBallControlDrawer and discarded after rendering; computed once
        # here so the report can use it too.
        team_ball_control = team_ball_control_drawer.get_team_ball_control(player_assignment, ball_aquisition)
        report = build_game_report(
            player_assignment=player_assignment,
            ball_aquisition=ball_aquisition,
            passes=passes,
            interceptions=interceptions,
            tactical_player_positions=tactical_player_positions,
            player_distances_per_frame=player_distances_per_frame,
            player_speed_per_frame=player_speed_per_frame,
            team_ball_control=team_ball_control,
            violations=violations,
        )
        if config.report_path:
            import json
            os.makedirs(os.path.dirname(config.report_path) or ".", exist_ok=True)
            with open(config.report_path, 'w') as f:
                json.dump(report, f, indent=2)

    return report
