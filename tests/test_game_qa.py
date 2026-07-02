import json
import unittest

from game_qa import (
    get_player_stats,
    rank_players_by_stat,
    compute_team_possession_pct,
    compare_players,
    dispatch_tool_call,
    generate_game_narrative,
    answer_question,
)
from tests.fakes import FakeClient, FakeResponse, FakeMessage, FakeToolCall


SAMPLE_REPORT = {
    "players": {
        "1": {"label": "Player 1", "team": 1, "total_distance_m": 100.0, "avg_speed_kmh": 8.0, "max_speed_kmh": 20.0},
        "2": {"label": "Player 2", "team": 2, "total_distance_m": 150.0, "avg_speed_kmh": 10.0, "max_speed_kmh": 22.0},
    },
    "team_possession": {"team_1_pct": 40.0, "team_2_pct": 55.0, "undecided_pct": 5.0},
    "events": {"passes": {"team_1": 3, "team_2": 5}, "interceptions": {"team_1": 1, "team_2": 0}},
    "num_frames": 100,
}


class TestToolFunctions(unittest.TestCase):
    def test_get_player_stats_all_fields(self):
        result = get_player_stats(SAMPLE_REPORT, 1)
        self.assertEqual(result["total_distance_m"], 100.0)

    def test_get_player_stats_subset(self):
        result = get_player_stats(SAMPLE_REPORT, 1, stats=["avg_speed_kmh"])
        self.assertEqual(result, {"avg_speed_kmh": 8.0})

    def test_get_player_stats_unknown_player(self):
        result = get_player_stats(SAMPLE_REPORT, 999)
        self.assertIn("error", result)

    def test_rank_players_by_stat_descending(self):
        ranked = rank_players_by_stat(SAMPLE_REPORT, "total_distance_m", top_n=2)
        self.assertEqual([r["player_id"] for r in ranked], [2, 1])

    def test_rank_players_respects_top_n(self):
        ranked = rank_players_by_stat(SAMPLE_REPORT, "total_distance_m", top_n=1)
        self.assertEqual(len(ranked), 1)

    def test_compute_team_possession_pct_valid_team(self):
        self.assertEqual(compute_team_possession_pct(SAMPLE_REPORT, 2), 55.0)

    def test_compute_team_possession_pct_invalid_team(self):
        result = compute_team_possession_pct(SAMPLE_REPORT, 3)
        self.assertIn("error", result)

    def test_compare_players(self):
        result = compare_players(SAMPLE_REPORT, [1, 2], stats=["total_distance_m"])
        self.assertEqual(result["1"], {"total_distance_m": 100.0})
        self.assertEqual(result["2"], {"total_distance_m": 150.0})

    def test_dispatch_tool_call_routes_correctly(self):
        result = dispatch_tool_call(SAMPLE_REPORT, "compute_team_possession_pct", {"team_id": 1})
        self.assertEqual(result, 40.0)

    def test_dispatch_tool_call_unknown_tool_raises_keyerror(self):
        with self.assertRaises(KeyError):
            dispatch_tool_call(SAMPLE_REPORT, "not_a_real_tool", {})


class TestGenerateGameNarrative(unittest.TestCase):
    def test_returns_model_text_and_sends_report_in_context(self):
        client = FakeClient([FakeResponse(FakeMessage(content="Team 2 controlled the pace."))])
        narrative = generate_game_narrative(client, "any-model", SAMPLE_REPORT)
        self.assertEqual(narrative, "Team 2 controlled the pace.")
        sent_messages = client.calls[0]["messages"]
        self.assertIn(json.dumps(SAMPLE_REPORT), sent_messages[-1]["content"])


class TestAnswerQuestion(unittest.TestCase):
    def test_qualitative_question_answered_directly_without_tool_call(self):
        client = FakeClient([FakeResponse(FakeMessage(content="It was a fast-paced game."))])
        answer = answer_question(client, "any-model", SAMPLE_REPORT, "How did the game feel?")
        self.assertEqual(answer, "It was a fast-paced game.")
        self.assertEqual(len(client.calls), 1)

    def test_numeric_question_routes_through_a_real_tool_call_not_free_text_math(self):
        # First turn: model requests the rank_players_by_stat tool.
        # Second turn: model gives a final answer after seeing the real tool result.
        tool_call = FakeToolCall("call_1", "rank_players_by_stat", json.dumps({"stat": "total_distance_m", "top_n": 1}))
        client = FakeClient([
            FakeResponse(FakeMessage(content=None, tool_calls=[tool_call])),
            FakeResponse(FakeMessage(content="Player 2 covered the most ground, at 150.0 meters.")),
        ])
        answer = answer_question(client, "any-model", SAMPLE_REPORT, "Who covered the most distance?")
        self.assertIn("150.0", answer)
        self.assertEqual(len(client.calls), 2)
        # The tool result actually fed back to the model must contain the real computed
        # value (150.0m for player 2) -- not something the model invented.
        tool_result_message = client.calls[1]["messages"][-1]
        self.assertEqual(tool_result_message["role"], "tool")
        parsed = json.loads(tool_result_message["content"])
        self.assertEqual(parsed[0]["total_distance_m"], 150.0)

    def test_unknown_tool_name_does_not_crash_the_loop(self):
        bad_tool_call = FakeToolCall("call_1", "not_a_real_tool", "{}")
        client = FakeClient([
            FakeResponse(FakeMessage(content=None, tool_calls=[bad_tool_call])),
            FakeResponse(FakeMessage(content="Sorry, I couldn't compute that.")),
        ])
        answer = answer_question(client, "any-model", SAMPLE_REPORT, "some question")
        self.assertEqual(answer, "Sorry, I couldn't compute that.")


if __name__ == "__main__":
    unittest.main()
