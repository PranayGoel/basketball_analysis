"""
Fine-tunes one of the 3 basketball YOLO models FROM an already-working
checkpoint you provide, using a Roboflow-hosted dataset -- NOT a from-scratch
training run.

This matters: the original training_notebooks/ recipes train from a generic
COCO base checkpoint (yolov8x-pose.pt / yolov5l6u.pt -- both large model
variants) for 500 / 250 / 100 epochs respectively, clearly sized for a real
cloud GPU (the notebooks' own pip-freeze output shows a Colab environment).
This script instead loads YOUR real, already-converged weights as the
starting point (`YOLO(args.base_weights)`) and asks for far fewer epochs,
because fine-tuning nudges an already-good model rather than teaching it the
task from zero.

Time expectations, stated honestly, not precisely: this Mac has no dedicated
GPU (MPS only, confirmed via torch.backends.mps.is_available()). MPS YOLO
training is well-documented to run meaningfully slower than a cloud GPU, and
worse on large model variants -- exactly what the original weights were
trained on. There's no way to promise a number in advance; this script prints
live per-epoch wall-clock time from the first epoch onward so you get a real,
machine-specific projection early. Use --dry-run to see the resolved plan
(dataset size, epochs, batch, device) before committing any time at all.

Setup: needs your OWN Roboflow API key (get one free at
https://app.roboflow.com/settings/api), set as ROBOFLOW_API_KEY in your
environment or in a .env file at the repo root (see .env.example). Do NOT
reuse the API key hardcoded in training_notebooks/basketball_player_detection_training.ipynb
-- that's a real, leaked credential belonging to the original tutorial
author, not something to borrow.

Usage:
    # See the resolved plan without downloading or training anything:
    python scripts/finetune_model.py --model-type player --base-weights models/player_detector.pt --dry-run

    # Actually fine-tune (needs ROBOFLOW_API_KEY set):
    python scripts/finetune_model.py --model-type player --base-weights models/player_detector.pt --epochs 30

    # Adopt the result into the live pipeline once you're happy with it
    # (validate with scripts/benchmark_detection.py before AND after --adopt):
    python scripts/finetune_model.py --model-type ball --base-weights models/ball_detector_model.pt --adopt
"""

import argparse
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from personal.basketball_analysis.pipeline.core import resolve_device
from personal.basketball_analysis.configs import PLAYER_DETECTOR_PATH, BALL_DETECTOR_PATH, COURT_KEYPOINT_DETECTOR_PATH

# Per-model-type defaults, matching llm_client.py's PROVIDER_CONFIG shape: one
# dict keyed by the discriminator, each value carrying everything type-specific.
#
# The player/ball default dataset is the exact one referenced in the existing
# training_notebooks/ (workspace-5ujvu/basketball-players-fy4c2-vfsuv:17) --
# kept as the default for continuity with whatever the user's real weights
# were originally shaped by, not because it's been independently re-verified
# here. The ball notebook pointing at the SAME project as the player notebook
# is suspicious (looks like a copy-paste bug in the original tutorial, or the
# dataset might genuinely be multi-class) -- this script resolves that with a
# runtime self-check (see _self_check_ball_dataset), not a guess.
MODEL_TYPE_CONFIG = {
    "player": {
        "task": "detect",
        "default_dataset": ("workspace-5ujvu", "basketball-players-fy4c2-vfsuv", 17),
        "default_format": "yolov5",
        "default_batch": 8,
        "expected_class_hint": None,
    },
    "ball": {
        "task": "detect",
        "default_dataset": ("workspace-5ujvu", "basketball-players-fy4c2-vfsuv", 17),
        "default_format": "yolov5",
        "default_batch": 8,
        "expected_class_hint": "ball",
    },
    "court_keypoint": {
        "task": "pose",
        "default_dataset": ("fyp-3bwmg", "reloc2-den7l", 1),
        "default_format": "yolov8",
        "default_batch": 16,
        "expected_class_hint": None,
    },
}

LIVE_PATH_BY_MODEL_TYPE = {
    "player": PLAYER_DETECTOR_PATH,
    "ball": BALL_DETECTOR_PATH,
    "court_keypoint": COURT_KEYPOINT_DETECTOR_PATH,
}


def _require_roboflow_key():
    key = os.environ.get("ROBOFLOW_API_KEY")
    if not key:
        print(
            "ERROR: ROBOFLOW_API_KEY not set. Set it in your shell or in a .env\n"
            "file at the repo root (see .env.example). Get your own key at\n"
            "https://app.roboflow.com/settings/api -- never reuse a key found in\n"
            "someone else's notebook or code (training_notebooks/ has a real,\n"
            "leaked one from the original tutorial author -- do not use it).",
            file=sys.stderr,
        )
        sys.exit(1)
    return key


def _check_base_weights_exist(path):
    if not os.path.exists(path):
        print(
            f"ERROR: --base-weights file not found: {path}\n"
            "This script fine-tunes FROM your own already-working checkpoint --\n"
            "it does not train from scratch. Provide the real weights file first.",
            file=sys.stderr,
        )
        sys.exit(1)


def _parse_dataset_arg(dataset_arg, default_triple):
    """Parses '--dataset WORKSPACE/PROJECT:VERSION' into (workspace, project, version).
    Falls back to the model-type's default triple when --dataset isn't passed."""
    if not dataset_arg:
        return default_triple
    try:
        workspace_project, version_str = dataset_arg.rsplit(":", 1)
        workspace, project = workspace_project.split("/", 1)
        return workspace, project, int(version_str)
    except ValueError:
        print(
            f"ERROR: --dataset must be in the form WORKSPACE/PROJECT:VERSION, got: {dataset_arg!r}",
            file=sys.stderr,
        )
        sys.exit(1)


def _read_dataset_class_names(dataset_location):
    """Parses the downloaded dataset's data.yaml `names:` list. Returns [] if
    missing/unparseable rather than raising -- this is a transparency/self-check
    aid, not a hard requirement for training to proceed."""
    import yaml

    data_yaml_path = os.path.join(dataset_location, "data.yaml")
    if not os.path.exists(data_yaml_path):
        return []
    try:
        with open(data_yaml_path) as f:
            data = yaml.safe_load(f)
        names = data.get("names", [])
        if isinstance(names, dict):
            names = list(names.values())
        return list(names)
    except Exception:
        return []


def _self_check_ball_dataset(class_names, allow_mismatched):
    """
    The ball-training notebook's dataset reference looks like it might be a
    copy-paste bug (points at the same 'basketball-players' project as the
    player notebook). Rather than guessing, check the ACTUAL downloaded
    dataset's class list: if nothing resembling "ball" is in it, this is
    very likely training a ball detector on a dataset that doesn't have ball
    annotations -- refuse to proceed silently.
    """
    has_ball_class = any("ball" in name.lower() for name in class_names)
    if not has_ball_class and not allow_mismatched:
        print(
            f"WARNING: downloaded dataset's classes are {class_names!r} -- none look\n"
            "like a ball/Ball class. This dataset may not be intended for ball\n"
            "detection (the original training_notebooks/basketball_ball_training.ipynb\n"
            "points at the same project as the player-detection notebook, which looks\n"
            "like a copy-paste bug, not a confirmed multi-class dataset).\n"
            "Pass --allow-mismatched-dataset to proceed anyway, or use --dataset to\n"
            "point at a different, confirmed ball-annotated Roboflow project.",
            file=sys.stderr,
        )
        sys.exit(1)


def _print_plan(args, model_config, dataset_triple):
    workspace, project, version = dataset_triple
    print("Fine-tuning plan:")
    print(f"  model_type:    {args.model_type} (task={model_config['task']})")
    print(f"  base_weights:  {args.base_weights}")
    print(f"  dataset:       {workspace}/{project} version {version} (format={args.dataset_format})")
    print(f"  epochs:        {args.epochs}")
    print(f"  imgsz:         {args.imgsz}")
    print(f"  batch:         {args.batch}")
    print(f"  device:        {resolve_device(args.device)}")
    print(f"  output_dir:    {args.output_dir}")
    print(f"  run_name:      {args.run_name or '(auto timestamp)'}")
    print(f"  adopt:         {args.adopt}")
    print()
    print("Time expectation: no dedicated GPU on this machine -- MPS YOLO training")
    print("is meaningfully slower than a cloud GPU, especially for large model")
    print("variants (which is what the original weights were trained on). This is")
    print("fine-tuning from your real checkpoint (far fewer epochs than training")
    print("from scratch), but there's no way to promise a number in advance --")
    print("watch the per-epoch wall-clock time once training starts and judge from")
    print("there. Consider lowering --epochs if the first epoch alone takes long.")


def _default_run_name(model_type):
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{model_type}_detector_{timestamp}_ft"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-type", required=True, choices=list(MODEL_TYPE_CONFIG.keys()),
                        help="Which model to fine-tune.")
    parser.add_argument("--base-weights", required=True,
                        help="Path to your real, already-working checkpoint to fine-tune FROM. "
                             "Never modified by this script.")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Fine-tuning epochs (default 30 -- a starting point for continuation "
                             "training, NOT the original notebooks' 500/250/100-epoch from-scratch "
                             "recipes). Adjust based on the per-epoch timing you observe.")
    parser.add_argument("--dataset", default=None,
                        help="Override dataset as WORKSPACE/PROJECT:VERSION. Defaults to the exact "
                             "dataset referenced in the original training_notebooks/ for this model type.")
    parser.add_argument("--dataset-format", default=None,
                        help="Roboflow download format (default: yolov5 for player/ball, yolov8 for "
                             "court_keypoint pose format -- matching the original notebooks).")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size (default 640, matches all 3 original notebooks).")
    parser.add_argument("--batch", type=int, default=None, help="Batch size (default: model-type specific, matching originals).")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"],
                        help="Training device (default: auto -> resolves to mps on this machine).")
    parser.add_argument("--output-dir", default="models/finetuned", help="Where fine-tuned runs are written.")
    parser.add_argument("--run-name", default=None, help="Run name (default: auto timestamp+model-type slug).")
    parser.add_argument("--adopt", action="store_true",
                        help="After training, copy the result into the live pipeline's canonical model "
                             "path. The previous file there is backed up (renamed with a timestamp "
                             "suffix), never deleted.")
    parser.add_argument("--allow-mismatched-dataset", action="store_true",
                        help="Required to proceed with --model-type ball if the downloaded dataset's "
                             "class list doesn't look like it contains a ball annotation.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the resolved plan and exit without downloading or training anything.")
    args = parser.parse_args()

    model_config = MODEL_TYPE_CONFIG[args.model_type]
    dataset_triple = _parse_dataset_arg(args.dataset, model_config["default_dataset"])
    if args.dataset_format is None:
        args.dataset_format = model_config["default_format"]
    if args.batch is None:
        args.batch = model_config["default_batch"]
    if args.run_name is None:
        args.run_name = _default_run_name(args.model_type)

    _check_base_weights_exist(args.base_weights)

    if args.dry_run:
        _print_plan(args, model_config, dataset_triple)
        return

    _print_plan(args, model_config, dataset_triple)
    api_key = _require_roboflow_key()

    from roboflow import Roboflow
    from ultralytics import YOLO

    workspace, project_name, version_num = dataset_triple
    print(f"\nDownloading dataset {workspace}/{project_name}:{version_num} ({args.dataset_format})...")
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(workspace).project(project_name)
    version = project.version(version_num)
    dataset = version.download(args.dataset_format)

    class_names = _read_dataset_class_names(dataset.location)
    print(f"Dataset classes: {class_names}")

    if args.model_type == "ball":
        _self_check_ball_dataset(class_names, args.allow_mismatched_dataset)

    resolved_device = resolve_device(args.device)
    print(f"\nLoading base checkpoint: {args.base_weights}")
    model = YOLO(args.base_weights)

    print(f"Starting fine-tuning ({args.epochs} epochs, device={resolved_device})...\n")
    model.train(
        data=os.path.join(dataset.location, "data.yaml"),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=resolved_device,
        project=args.output_dir,
        name=args.run_name,
        task=model_config["task"],
    )

    best_weights_path = os.path.join(args.output_dir, args.run_name, "weights", "best.pt")
    if not os.path.exists(best_weights_path):
        print(f"\nWARNING: expected fine-tuned weights not found at {best_weights_path} -- "
              "check the training run's output directory manually.", file=sys.stderr)
        return

    print(f"\nFine-tuning complete. New weights: {best_weights_path}")
    print(f"Your original --base-weights file was never modified: {args.base_weights}")

    if args.adopt:
        live_path = LIVE_PATH_BY_MODEL_TYPE[args.model_type]
        if os.path.exists(live_path):
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            backup_path = f"{live_path}.pre_finetune_{timestamp}.bak"
            shutil.move(live_path, backup_path)
            print(f"Backed up previous live weights to: {backup_path}")
        os.makedirs(os.path.dirname(live_path), exist_ok=True)
        shutil.copy2(best_weights_path, live_path)
        print(f"Adopted: copied to {live_path}")
    else:
        live_path = LIVE_PATH_BY_MODEL_TYPE[args.model_type]
        print(
            "\nNOT adopted. To use these weights, either:\n"
            f"  1. Re-run with --adopt, or\n"
            f"  2. Manually: cp {best_weights_path} {live_path}\n"
            "Run scripts/benchmark_detection.py once BEFORE and once AFTER adopting\n"
            "to confirm the change is actually an improvement on your own footage --\n"
            "YOLO's own training-time metrics (printed above) tell you whether it got\n"
            "better at the dataset's task; benchmark_detection.py tells you whether it\n"
            "got better on your actual game videos. These answer different questions."
        )


if __name__ == "__main__":
    main()
