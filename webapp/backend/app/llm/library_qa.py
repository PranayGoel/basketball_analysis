"""
library_qa.py: library-wide natural-language search, generalizing
game_qa.py's tool-calling pattern from a single report to the whole video
library.

Same philosophy as game_qa.py (see that module's docstring for the full
rationale, arXiv:2211.12588): the LLM's ONLY job is picking which tool
function to call and phrasing the final answer from the tool's (already
computed, SQL-derived) result. It never ranks, sorts, or computes a number
itself -- every tool function below is plain, deterministic, DB-query-backed
Python with zero LLM/network dependency, independently testable with a
synthetic DB fixture.

matched_video_ids is threaded through every tool result (as a `_video_ids`
key the dispatch loop pulls out, never shown to the model) so the frontend
can highlight the videos a search actually matched, without asking the model
to enumerate ids in prose (a second, less reliable extraction step).
"""

import json
from typing import Any, Dict, List, Optional

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.db.models import Player, Video, Violation

# ---------------------------------------------------------------------------
# Tool functions -- pure(ish: read-only DB queries), deterministic, no LLM
# involved. Each returns a plain JSON-serializable dict; video ids worth
# surfacing to the frontend are always included under "video_ids".
# ---------------------------------------------------------------------------


def find_closest_possession_split(db: Session) -> Dict[str, Any]:
    """The video whose team_a_possession_pct is closest to an even 50/50 split."""
    video = (
        db.query(Video)
        .filter(Video.team_a_possession_pct.isnot(None))
        .order_by(func.abs(Video.team_a_possession_pct - 50.0).asc())
        .first()
    )
    if video is None:
        return {"result": None, "video_ids": []}
    return {
        "result": _video_summary(video),
        "video_ids": [video.id],
    }


def find_videos_by_min_distance(db: Session, min_meters: float) -> Dict[str, Any]:
    """Videos containing at least one player whose total_distance_m >= min_meters."""
    video_ids = (
        db.query(Player.video_id)
        .filter(Player.total_distance_m >= min_meters)
        .distinct()
        .all()
    )
    ids = [row[0] for row in video_ids]
    videos = db.query(Video).filter(Video.id.in_(ids)).all() if ids else []
    return {
        "result": [_video_summary(v) for v in videos],
        "video_ids": [v.id for v in videos],
    }


_RANKABLE_STATS = {"total_passes", "total_interceptions", "max_player_speed_kmh"}


def rank_videos_by_stat(db: Session, stat: str, order: str = "desc", limit: int = 5) -> Dict[str, Any]:
    """
    Rank videos by one of the flattened analytics columns.

    Args:
        stat: one of "total_passes", "total_interceptions", "max_player_speed_kmh".
        order: "asc" or "desc".
        limit: max rows to return.
    """
    if stat not in _RANKABLE_STATS:
        return {"error": f"stat must be one of {sorted(_RANKABLE_STATS)}, got {stat!r}", "video_ids": []}

    column = getattr(Video, stat)
    query = db.query(Video).filter(column.isnot(None))
    query = query.order_by(column.desc() if order == "desc" else column.asc())
    videos = query.limit(limit).all()
    return {
        "result": [_video_summary(v) for v in videos],
        "video_ids": [v.id for v in videos],
    }


def compare_videos(db: Session, video_id_a: str, video_id_b: str) -> Dict[str, Any]:
    """Side-by-side key stats for two videos, by id."""
    video_a = db.get(Video, video_id_a)
    video_b = db.get(Video, video_id_b)
    if video_a is None or video_b is None:
        missing = [vid for vid, v in [(video_id_a, video_a), (video_id_b, video_b)] if v is None]
        return {"error": f"Unknown video id(s): {missing}", "video_ids": []}
    return {
        "result": {video_id_a: _video_summary(video_a), video_id_b: _video_summary(video_b)},
        "video_ids": [video_id_a, video_id_b],
    }


def find_videos_with_violations(db: Session, violation_type: Optional[str] = None) -> Dict[str, Any]:
    """
    Videos with at least one matching Violation row.

    Returns gracefully ({"result": [], "video_ids": []}) when the Violation
    table is empty -- pose analysis may not have run on any video yet, and
    that must never surface as an error to the model or the user.
    """
    query = db.query(Violation.video_id).distinct()
    if violation_type:
        query = query.filter(Violation.violation_type == violation_type)
    video_ids = [row[0] for row in query.all()]
    videos = db.query(Video).filter(Video.id.in_(video_ids)).all() if video_ids else []
    return {
        "result": [_video_summary(v) for v in videos],
        "video_ids": [v.id for v in videos],
    }


def _video_summary(video: Video) -> Dict[str, Any]:
    return {
        "video_id": video.id,
        "filename": video.filename,
        "status": video.status,
        "team_a_possession_pct": video.team_a_possession_pct,
        "team_b_possession_pct": video.team_b_possession_pct,
        "player_count": video.player_count,
        "total_passes": video.total_passes,
        "total_interceptions": video.total_interceptions,
        "max_player_speed_kmh": video.max_player_speed_kmh,
        "max_player_distance_m": video.max_player_distance_m,
        "has_violations": video.has_violations,
    }


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "find_closest_possession_split",
            "description": "Find the video with the closest to an even 50/50 team possession split.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_videos_by_min_distance",
            "description": "Find videos containing a player who covered at least this many meters.",
            "parameters": {
                "type": "object",
                "properties": {"min_meters": {"type": "number"}},
                "required": ["min_meters"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rank_videos_by_stat",
            "description": (
                "Rank videos by a numeric stat, ascending or descending. Use this for any "
                "'which video had the most/highest/fewest X' question across the library."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "stat": {
                        "type": "string",
                        "enum": ["total_passes", "total_interceptions", "max_player_speed_kmh"],
                    },
                    "order": {"type": "string", "enum": ["asc", "desc"]},
                    "limit": {"type": "integer"},
                },
                "required": ["stat"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_videos",
            "description": "Compare key stats for two videos side by side, by video id.",
            "parameters": {
                "type": "object",
                "properties": {
                    "video_id_a": {"type": "string"},
                    "video_id_b": {"type": "string"},
                },
                "required": ["video_id_a", "video_id_b"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_videos_with_violations",
            "description": (
                "Find videos with at least one detected rule violation (double dribble or "
                "traveling). Optionally filter by violation_type."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "violation_type": {
                        "type": "string",
                        "enum": ["double_dribble", "traveling"],
                    }
                },
            },
        },
    },
]

_TOOL_IMPLS = {
    "find_closest_possession_split": find_closest_possession_split,
    "find_videos_by_min_distance": find_videos_by_min_distance,
    "rank_videos_by_stat": rank_videos_by_stat,
    "compare_videos": compare_videos,
    "find_videos_with_violations": find_videos_with_violations,
}


def dispatch_tool_call(db: Session, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute a tool call by name against the DB. Raises KeyError if `name`
    isn't a known tool -- callers should treat this as the model hallucinating
    a tool name and handle it (see query_library below), same convention as
    game_qa.py's dispatch_tool_call.
    """
    func_impl = _TOOL_IMPLS[name]
    return func_impl(db, **arguments)


LIBRARY_QA_SYSTEM_PROMPT = (
    "You are a basketball analyst answering questions across a whole library of "
    "analyzed game videos, using the tools provided. ALWAYS call the appropriate "
    "tool for any question involving a number, ranking, comparison, or search across "
    "videos -- never calculate, rank, or guess this yourself. Only answer directly, "
    "without a tool call, for purely qualitative questions with no computed content."
)


def query_library(client, model: str, question: str, db: Session, max_tool_turns: int = 3) -> Dict[str, Any]:
    """
    Tool-calling Q&A loop over the whole video library.

    Args:
        client: an OpenAI-SDK-shaped client (real or fake, dependency-injected
            exactly like game_qa.answer_question).
        model: model name to use.
        question: the user's natural-language search/question.
        db: an active Session the tool functions query against.
        max_tool_turns: safety cap on tool-call round-trips.

    Returns:
        dict: {"answer": str, "matched_video_ids": list[str]} -- matched_video_ids
        is the union of every "video_ids" list returned by any tool call made
        during this exchange, in first-seen order, deduplicated.
    """
    messages = [
        {"role": "system", "content": LIBRARY_QA_SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    matched_video_ids: List[str] = []

    def _record_ids(ids):
        for vid in ids or []:
            if vid not in matched_video_ids:
                matched_video_ids.append(vid)

    for _ in range(max_tool_turns):
        response = client.chat.completions.create(
            model=model, messages=messages, tools=TOOLS, tool_choice="auto"
        )
        choice = response.choices[0].message
        tool_calls = getattr(choice, "tool_calls", None)

        if not tool_calls:
            return {"answer": choice.content, "matched_video_ids": matched_video_ids}

        messages.append({"role": "assistant", "content": choice.content, "tool_calls": tool_calls})

        for tool_call in tool_calls:
            name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            try:
                result = dispatch_tool_call(db, name, arguments)
            except KeyError:
                result = {"error": f"Unknown tool '{name}'"}
            _record_ids(result.get("video_ids"))
            messages.append(
                {"role": "tool", "tool_call_id": tool_call.id, "content": json.dumps(result)}
            )

    # Exceeded max_tool_turns without a final answer -- ask once more without
    # tools so the caller still gets something rather than silently looping.
    response = client.chat.completions.create(model=model, messages=messages)
    return {"answer": response.choices[0].message.content, "matched_video_ids": matched_video_ids}
