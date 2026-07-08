"""
report_indexer.py: flattens a completed job's game_report dict into
Player/GameEvent/Violation rows and the Video row's summary columns.

This is the ONE place that understands game_report.py's dict shape well
enough to translate it into SQL-queryable columns -- everything downstream
(library filters, sort, NL-search tool functions) reads flattened columns,
never re-parses report JSON.
"""

from typing import Any, Dict

from sqlalchemy.orm import Session

from personal.basketball_analysis.webapp.backend.app.db.models import GameEvent, Player, Video
from personal.basketball_analysis.webapp.backend.app.db.models import Violation as ViolationModel


def index_report(db: Session, video_id: str, report: Dict[str, Any]) -> None:
    """
    Populate Player/GameEvent/Violation rows and Video's flattened analytics
    columns from a completed report dict.

    Idempotent-ish: deletes any pre-existing Player/GameEvent/Violation rows
    for this video_id first, so re-indexing (e.g. a retry) doesn't duplicate
    rows. Raises if no Video row with this id exists -- indexing only makes
    sense against an already-ingested video.

    Args:
        db: an active Session (caller commits).
        video_id: the Video.id these rows belong to.
        report: a dict shaped exactly like game_report.build_game_report()'s
            return value -- "players", "team_possession", "events",
            "num_frames", and optionally "violations".
    """
    video = db.get(Video, video_id)
    if video is None:
        raise ValueError(f"No Video row with id {video_id!r} -- ingest before indexing.")

    # Clear any prior rows for this video (safe on a first index too, when
    # there are none) so re-running indexing never duplicates rows.
    db.query(Player).filter(Player.video_id == video_id).delete()
    db.query(GameEvent).filter(GameEvent.video_id == video_id).delete()
    db.query(ViolationModel).filter(ViolationModel.video_id == video_id).delete()

    players = report.get("players", {})
    max_speed = 0.0
    max_distance = 0.0
    for tracker_player_id_str, player_data in players.items():
        player_row = Player(
            video_id=video_id,
            tracker_player_id=int(tracker_player_id_str),
            label=player_data.get("label", f"Player {tracker_player_id_str}"),
            team=player_data.get("team"),
            total_distance_m=player_data.get("total_distance_m", 0.0),
            avg_speed_kmh=player_data.get("avg_speed_kmh", 0.0),
            max_speed_kmh=player_data.get("max_speed_kmh", 0.0),
        )
        db.add(player_row)
        max_speed = max(max_speed, player_row.max_speed_kmh)
        max_distance = max(max_distance, player_row.total_distance_m)

    events = report.get("events", {})
    passes = events.get("passes", {})
    interceptions = events.get("interceptions", {})
    total_passes = 0
    total_interceptions = 0
    for team_key, count in passes.items():
        team_num = _team_num_from_key(team_key)
        if team_num is not None and count:
            db.add(GameEvent(video_id=video_id, event_type="pass", team=team_num))
        total_passes += count
    for team_key, count in interceptions.items():
        team_num = _team_num_from_key(team_key)
        if team_num is not None and count:
            db.add(GameEvent(video_id=video_id, event_type="interception", team=team_num))
        total_interceptions += count

    # violations absent (key not present at all) means pose/violation
    # detection didn't run this pass -- has_violations stays False, no
    # Violation rows are created, and this must NOT be conflated with an
    # explicit empty list (which means it ran and found nothing -- also
    # has_violations=False, but a real, completed analysis). Either way no
    # Violation rows are created; the distinction only matters for a future
    # "not yet analyzed for violations" UI affordance, which is out of scope
    # for this indexer -- it just needs to not crash on either case.
    violations = report.get("violations")
    has_violations = False
    if violations:
        has_violations = True
        for v in violations:
            db.add(
                ViolationModel(
                    video_id=video_id,
                    player_id=v["player_id"],
                    start_frame=v["start_frame"],
                    end_frame=v["end_frame"],
                    violation_type=v["violation_type"],
                    confidence=v["confidence"],
                )
            )

    team_possession = report.get("team_possession", {})
    video.team_a_possession_pct = team_possession.get("team_1_pct")
    video.team_b_possession_pct = team_possession.get("team_2_pct")
    video.player_count = len(players)
    video.total_passes = total_passes
    video.total_interceptions = total_interceptions
    video.max_player_speed_kmh = max_speed if players else None
    video.max_player_distance_m = max_distance if players else None
    video.has_violations = has_violations

    db.commit()


def _team_num_from_key(team_key: str):
    """"team_1" -> 1, "team_2" -> 2, anything else -> None."""
    if team_key == "team_1":
        return 1
    if team_key == "team_2":
        return 2
    return None
