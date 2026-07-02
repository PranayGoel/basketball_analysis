"""
Post-game analytics report: aggregates the pipeline's per-frame outputs into a single
JSON-serializable summary (per-player distance/speed, team ball-control %, pass/
interception counts).

Every input here is a plain list/dict already produced by main.py's pipeline (bboxes,
tracks, and derived stats) -- nothing in this module imports ultralytics/torch/
transformers, so it can be built and unit-tested with synthetic fixtures alone.

This is also the data layer the AI insight features (llm_client.py's narrative
generation and tool-calling Q&A) sit on top of.
"""

from collections import Counter


def resolve_player_teams(player_assignment):
    """
    Resolve a single, stable team_id per player across the whole clip.

    TeamAssigner resets its internal mapping every 50 frames, so a naive per-frame
    lookup can flip a player between teams mid-video. This takes the majority-vote
    team across every frame the player appears in, which is far more stable for a
    post-game summary than trusting any single frame.

    Args:
        player_assignment (list[dict[int, int]]): per-frame player_id -> team_id (1 or 2).

    Returns:
        dict[int, int]: player_id -> the most common team_id observed for that player.
    """
    votes = {}
    for frame in player_assignment:
        for player_id, team_id in frame.items():
            votes.setdefault(player_id, Counter())[team_id] += 1

    return {player_id: counter.most_common(1)[0][0] for player_id, counter in votes.items()}


def compute_team_possession_pct(team_ball_control):
    """
    Percentage of frames each team controlled the ball, matching the same definition
    already drawn on-video by TeamBallControlDrawer (percentage of ALL frames, not just
    frames with a decided possessor).

    Args:
        team_ball_control: sequence of {-1, 1, 2} (numpy array or plain list).

    Returns:
        dict: {"team_1_pct": float, "team_2_pct": float, "undecided_pct": float}
    """
    total = len(team_ball_control)
    if total == 0:
        return {"team_1_pct": 0.0, "team_2_pct": 0.0, "undecided_pct": 0.0}

    team_1 = sum(1 for v in team_ball_control if v == 1)
    team_2 = sum(1 for v in team_ball_control if v == 2)
    undecided = total - team_1 - team_2

    return {
        "team_1_pct": round(100 * team_1 / total, 2),
        "team_2_pct": round(100 * team_2 / total, 2),
        "undecided_pct": round(100 * undecided / total, 2),
    }


def compute_player_movement_stats(player_distances_per_frame, player_speed_per_frame):
    """
    Per-player total distance (meters) and speed stats (km/h) across the whole clip.

    Speed frames where the calculator had insufficient samples default to 0.0 (see
    SpeedAndDistanceCalculator.calculate_speed) -- those are excluded from avg/max here
    since they represent "not enough data yet", not a genuine measured zero.

    Args:
        player_distances_per_frame (list[dict[int, float]]): per-frame incremental distance.
        player_speed_per_frame (list[dict[int, float]]): per-frame instantaneous speed (km/h).

    Returns:
        dict[int, dict]: player_id -> {"total_distance_m": float, "avg_speed_kmh": float,
            "max_speed_kmh": float}.
    """
    total_distance = {}
    for frame in player_distances_per_frame:
        for player_id, dist in frame.items():
            total_distance[player_id] = total_distance.get(player_id, 0.0) + dist

    speed_samples = {}
    for frame in player_speed_per_frame:
        for player_id, speed in frame.items():
            if speed > 0:
                speed_samples.setdefault(player_id, []).append(speed)

    player_ids = set(total_distance) | set(speed_samples)
    stats = {}
    for player_id in player_ids:
        samples = speed_samples.get(player_id, [])
        stats[player_id] = {
            "total_distance_m": round(total_distance.get(player_id, 0.0), 2),
            "avg_speed_kmh": round(sum(samples) / len(samples), 2) if samples else 0.0,
            "max_speed_kmh": round(max(samples), 2) if samples else 0.0,
        }
    return stats


def compute_event_counts(passes, interceptions):
    """
    Per-team pass and interception counts.

    Args:
        passes (list[int]): per-frame, -1/1/2 (team that completed a pass, if any).
        interceptions (list[int]): per-frame, -1/1/2 (team that intercepted, if any).

    Returns:
        dict: {"passes": {"team_1": int, "team_2": int}, "interceptions": {...}}
    """
    return {
        "passes": {
            "team_1": sum(1 for p in passes if p == 1),
            "team_2": sum(1 for p in passes if p == 2),
        },
        "interceptions": {
            "team_1": sum(1 for i in interceptions if i == 1),
            "team_2": sum(1 for i in interceptions if i == 2),
        },
    }


def build_game_report(
    player_assignment,
    ball_aquisition,
    passes,
    interceptions,
    tactical_player_positions,
    player_distances_per_frame,
    player_speed_per_frame,
    team_ball_control,
    player_labels=None,
):
    """
    Build the full post-game JSON report.

    Args:
        player_assignment, ball_aquisition, passes, interceptions,
        tactical_player_positions, player_distances_per_frame, player_speed_per_frame:
            outputs already produced by main.py's pipeline (see main.py for exact shapes).
        team_ball_control: output of TeamBallControlDrawer.get_team_ball_control -- pass
            this in explicitly rather than recomputing, since main.py now computes it once
            and shares it between drawing and reporting.
        player_labels (dict[int, str], optional): player_id -> a human-readable label
            (e.g. a resolved jersey number, "#23"). Falls back to "Player {id}" when not
            provided or missing a given id -- jersey-number recognition itself is future
            work (documented in the README), this parameter just keeps the report format
            forward-compatible with it.

    Returns:
        dict: JSON-serializable report.
    """
    player_labels = player_labels or {}
    player_teams = resolve_player_teams(player_assignment)
    movement_stats = compute_player_movement_stats(player_distances_per_frame, player_speed_per_frame)

    player_ids = set(player_teams) | set(movement_stats)
    players = {}
    for player_id in sorted(player_ids):
        players[str(player_id)] = {
            "label": player_labels.get(player_id, f"Player {player_id}"),
            "team": player_teams.get(player_id),
            **movement_stats.get(player_id, {"total_distance_m": 0.0, "avg_speed_kmh": 0.0, "max_speed_kmh": 0.0}),
        }

    return {
        "players": players,
        "team_possession": compute_team_possession_pct(team_ball_control),
        "events": compute_event_counts(passes, interceptions),
        "num_frames": len(ball_aquisition),
    }
