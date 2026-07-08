import argparse
import time

from pipeline import PipelineConfig, run_analysis
from configs import (
    STUBS_DEFAULT_PATH,
    OUTPUT_VIDEO_PATH,
    DEVICE,
    BATCH_SIZE,
    PLAYER_CONF_THRESHOLD,
    BALL_CONF_THRESHOLD,
    COURT_KEYPOINT_CONF_THRESHOLD,
    USE_HALF_PRECISION,
)


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
    parser.add_argument('--conf', type=float, default=None,
                        help='Global YOLO detection confidence threshold override, '
                             'applied to all three models. Leave unset to use each '
                             "model's own tuned default (player/ball/court differ). "
                             'A --player-conf/--ball-conf/--court-conf flag, if also '
                             'passed, takes precedence over this for that model.')
    parser.add_argument('--player-conf', dest='player_conf', type=float, default=None,
                        help=f'Player detector confidence threshold (default {PLAYER_CONF_THRESHOLD})')
    parser.add_argument('--ball-conf', dest='ball_conf', type=float, default=None,
                        help=f'Ball detector confidence threshold (default {BALL_CONF_THRESHOLD}, '
                             'intentionally lower than the player default -- see configs/configs.py)')
    parser.add_argument('--court-conf', dest='court_conf', type=float, default=None,
                        help=f'Court keypoint detector confidence threshold (default {COURT_KEYPOINT_CONF_THRESHOLD})')
    parser.add_argument('--half', action='store_true', default=USE_HALF_PRECISION,
                        help='Run inference in half precision (FP16). Off by default -- '
                             'benchmark on your actual device before enabling (MPS fp16 '
                             'maturity varies by torch version).')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                        help='Number of frames per YOLO prediction batch')
    parser.add_argument('--no-stubs', dest='use_stubs', action='store_false',
                        help='Ignore cached stub files and recompute all detections')
    parser.add_argument('--profile', action='store_true',
                        help='Print per-stage wall-clock timings and end-to-end FPS')
    parser.add_argument('--report', type=str, default=None,
                        help='Path to write a post-game JSON analytics report (see game_report.py)')
    parser.add_argument('--pose', action='store_true', default=False,
                        help='Run pose estimation and heuristic rule-violation detection '
                             '(double dribble, traveling). Off by default -- adds real '
                             'per-player-per-frame inference cost on top of the other three '
                             'models. Flags are best-effort heuristics, not validated '
                             'officiating calls -- see README.md.')
    parser.add_argument('--pose_model', type=str, default='yolo11n-pose.pt',
                        help="Ultralytics pose model variant. Auto-downloads on first use "
                             "(no manual Google Drive step needed, unlike the other 3 models).")
    parser.set_defaults(use_stubs=True)
    return parser.parse_args()


def config_from_args(args):
    """Resolve CLI-specific flag precedence (per-model conf > global --conf > tuned
    default) into a plain PipelineConfig -- the web backend builds its own
    PipelineConfig directly from a request body, with no equivalent "global
    override" concept to resolve."""
    return PipelineConfig(
        video_path=args.input_video,
        output_video_path=args.output_video,
        report_path=args.report,
        stub_dir=args.stub_path,
        device=args.device,
        use_stubs=args.use_stubs,
        player_conf=args.player_conf if args.player_conf is not None else args.conf,
        ball_conf=args.ball_conf if args.ball_conf is not None else args.conf,
        court_conf=args.court_conf if args.court_conf is not None else args.conf,
        half=args.half,
        batch_size=args.batch_size,
        pose=args.pose,
        pose_model=args.pose_model,
    )


class _CliProfiler:
    """
    Turns ProgressReporter's honest stage-boundary events (index/total, no
    fabricated weighted %) into the same kind of per-stage wall-clock timing
    table --profile printed before the pipeline/core.py extraction. The
    timing bookkeeping lives here (CLI-specific presentation), not in
    ProgressReporter itself, which stays a plain boundary-event emitter that
    both the CLI and the web backend's SSE stream can consume identically.
    """
    def __init__(self):
        self.stage_order = []
        self._start_times = {}
        self.durations = {}

    def callback(self, stage_name, stage_index, total_stages, status):
        if status == "running":
            print(f"[{stage_index + 1}/{total_stages}] {stage_name} ...")
            self._start_times[stage_name] = time.perf_counter()
            self.stage_order.append(stage_name)
        else:
            started = self._start_times.get(stage_name)
            if started is not None:
                self.durations[stage_name] = time.perf_counter() - started

    def print_table(self, total_wall_clock, num_frames):
        print("\n[profile] stage timings (seconds):")
        for name in self.stage_order:
            print(f"  {name:18s} {self.durations.get(name, float('nan')):8.2f}")
        print(f"  {'TOTAL':18s} {total_wall_clock:8.2f}")
        if total_wall_clock > 0:
            print(f"[profile] {num_frames} frames -> {num_frames / total_wall_clock:.2f} FPS end-to-end")


def main():
    args = parse_args()
    config = config_from_args(args)

    print(f"[config] device={config.device} batch_size={config.batch_size} "
          f"use_stubs={config.use_stubs} half={config.half} pose={config.pose}")

    profiler = _CliProfiler() if args.profile else None
    progress_callback = profiler.callback if profiler else None

    t0 = time.perf_counter()
    report = run_analysis(config, progress_callback=progress_callback)
    total = time.perf_counter() - t0

    if args.report:
        print(f"[report] wrote game report to {args.report}")

    if args.profile:
        profiler.print_table(total, report['num_frames'])


if __name__ == '__main__':
    main()
