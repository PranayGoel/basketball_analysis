"""
AI insight layer on top of game_report.py's output: a natural-language game summary,
and a tool-calling Q&A chat over the report.

Design rationale (see README for the full research citation): LLMs are measurably
unreliable doing arithmetic/ranking by reading raw data in-context (~12% accuracy gap
vs. delegating computation to real code, per the Program-of-Thoughts paper,
arXiv:2211.12588). So every *computed* answer here routes through one of the four
tool functions below -- plain, pure, unit-tested Python -- never through the model
doing math over the JSON itself. Qualitative questions still get the full report in
context, since there's nothing to compute for those.

All four tool functions operate on a `report` dict shaped like game_report.py's
build_game_report() output and have zero LLM/network dependency -- fully testable on
their own with a synthetic report fixture.
"""

import json


# ---------------------------------------------------------------------------
# Tool functions -- pure, deterministic, no LLM involved. The model's only job is
# deciding which of these to call and with what arguments; it never computes a
# number itself.
# ---------------------------------------------------------------------------

def get_player_stats(report, player_id, stats=None):
    """
    Look up one player's stats.

    Args:
        report (dict): output of game_report.build_game_report.
        player_id (int or str): the player's numeric id.
        stats (list[str], optional): which fields to return (subset of
            "label", "team", "total_distance_m", "avg_speed_kmh", "max_speed_kmh").
            Defaults to all.

    Returns:
        dict: the requested fields, or {"error": ...} if the player isn't in the report.
    """
    player = report["players"].get(str(player_id))
    if player is None:
        return {"error": f"No player with id {player_id} in this report."}
    if not stats:
        return dict(player)
    return {k: player.get(k) for k in stats}


def rank_players_by_stat(report, stat, top_n=5):
    """
    Rank players by a numeric stat, descending.

    Args:
        report (dict): output of game_report.build_game_report.
        stat (str): one of "total_distance_m", "avg_speed_kmh", "max_speed_kmh".
        top_n (int): how many to return.

    Returns:
        list[dict]: [{"player_id": int, "label": str, stat: value}, ...], best first.
    """
    ranked = []
    for player_id_str, player in report["players"].items():
        if stat not in player:
            continue
        ranked.append({"player_id": int(player_id_str), "label": player["label"], stat: player[stat]})
    ranked.sort(key=lambda p: p[stat], reverse=True)
    return ranked[:top_n]


def compute_team_possession_pct(report, team_id):
    """
    Look up a team's ball-possession percentage from the report.

    Args:
        report (dict): output of game_report.build_game_report.
        team_id (int): 1 or 2.

    Returns:
        float or dict: the percentage, or {"error": ...} for an invalid team_id.
    """
    key = f"team_{team_id}_pct"
    if key not in report["team_possession"]:
        return {"error": f"team_id must be 1 or 2, got {team_id}"}
    return report["team_possession"][key]


def compare_players(report, player_ids, stats=None):
    """
    Compare multiple players side by side.

    Args:
        report (dict): output of game_report.build_game_report.
        player_ids (list[int]): players to compare.
        stats (list[str], optional): which fields to include per player.

    Returns:
        dict[str, dict]: player_id (as string) -> their stats dict.
    """
    return {str(pid): get_player_stats(report, pid, stats=stats) for pid in player_ids}


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_player_stats",
            "description": "Look up a single player's stats by id.",
            "parameters": {
                "type": "object",
                "properties": {
                    "player_id": {"type": "integer"},
                    "stats": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional subset of fields to return.",
                    },
                },
                "required": ["player_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rank_players_by_stat",
            "description": "Rank all players by a numeric stat, descending. Use this for any 'who had the most/best/highest X' question.",
            "parameters": {
                "type": "object",
                "properties": {
                    "stat": {"type": "string", "enum": ["total_distance_m", "avg_speed_kmh", "max_speed_kmh"]},
                    "top_n": {"type": "integer"},
                },
                "required": ["stat"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compute_team_possession_pct",
            "description": "Get a team's ball-possession percentage for the game.",
            "parameters": {
                "type": "object",
                "properties": {"team_id": {"type": "integer", "enum": [1, 2]}},
                "required": ["team_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_players",
            "description": "Compare two or more players' stats side by side.",
            "parameters": {
                "type": "object",
                "properties": {
                    "player_ids": {"type": "array", "items": {"type": "integer"}},
                    "stats": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["player_ids"],
            },
        },
    },
]

_TOOL_IMPLS = {
    "get_player_stats": get_player_stats,
    "rank_players_by_stat": rank_players_by_stat,
    "compute_team_possession_pct": compute_team_possession_pct,
    "compare_players": compare_players,
}


def dispatch_tool_call(report, name, arguments):
    """
    Execute a tool call by name against `report`. `arguments` is the dict already
    parsed from the model's tool-call JSON.

    Raises:
        KeyError: if `name` isn't a known tool -- callers should treat this as the
        model hallucinating a tool name and handle it (e.g. return an error message
        back to the model rather than crashing).
    """
    func = _TOOL_IMPLS[name]
    return func(report, **arguments)


# ---------------------------------------------------------------------------
# LLM-facing features. Both take an already-constructed `client` (dependency
# injection) so they're testable with a fake client, independent of llm_client.py's
# get_client()/the real `openai` package being installed.
# ---------------------------------------------------------------------------

NARRATIVE_SYSTEM_PROMPT = (
    "You are a basketball analyst. Given a structured post-game report (JSON), write "
    "a concise, natural-language summary: overall pace/possession narrative, standout "
    "players (by distance/speed), and any notable pass/interception activity. Do not "
    "invent any numbers not present in the report -- only describe what's there."
)


def generate_game_narrative(client, model, report, max_tokens=400):
    """
    One-shot narrative generation: the full report JSON goes in context, no tool
    calling needed since this is descriptive generation, not computed lookup --
    matches the established "stats-to-narrative" pattern (see README).
    """
    messages = [
        {"role": "system", "content": NARRATIVE_SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(report)},
    ]
    response = client.chat.completions.create(model=model, messages=messages, max_tokens=max_tokens)
    return response.choices[0].message.content


QA_SYSTEM_PROMPT = (
    "You are a basketball analyst answering questions about a specific game using the "
    "tools provided. ALWAYS call the appropriate tool for any question involving a "
    "number, ranking, comparison, or computed statistic -- never calculate or estimate "
    "a stat yourself from memory or by reading the report text. Only answer directly, "
    "without a tool call, for purely qualitative questions with no numeric content."
)


def answer_question(client, model, report, question, max_tool_turns=3):
    """
    Tool-calling Q&A loop: the model may call one of the four report-query tools
    (possibly more than once) before giving a final answer. Every computed number in
    the final answer traces back to a real function call against `report`, not the
    model's own arithmetic.

    Args:
        client: an OpenAI-SDK-shaped client (real or fake).
        model (str): model name to use.
        report (dict): the game report tool calls operate on.
        question (str): the user's natural-language question.
        max_tool_turns (int): safety cap on tool-call round-trips.

    Returns:
        str: the model's final natural-language answer.
    """
    messages = [
        {"role": "system", "content": QA_SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    for _ in range(max_tool_turns):
        response = client.chat.completions.create(
            model=model, messages=messages, tools=TOOLS, tool_choice="auto"
        )
        choice = response.choices[0].message
        tool_calls = getattr(choice, "tool_calls", None)

        if not tool_calls:
            return choice.content

        messages.append({"role": "assistant", "content": choice.content, "tool_calls": tool_calls})

        for tool_call in tool_calls:
            name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            try:
                result = dispatch_tool_call(report, name, arguments)
            except KeyError:
                result = {"error": f"Unknown tool '{name}'"}
            messages.append(
                {"role": "tool", "tool_call_id": tool_call.id, "content": json.dumps(result)}
            )

    # Exceeded max_tool_turns without a final answer -- ask once more without tools
    # so the caller still gets something rather than silently looping forever.
    response = client.chat.completions.create(model=model, messages=messages)
    return response.choices[0].message.content
