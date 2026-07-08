import unittest
from personal.basketball_analysis.game_report import (
    resolve_player_teams,
    compute_team_possession_pct,
    compute_player_movement_stats,
    compute_event_counts,
    build_game_report,
)


class TestResolvePlayerTeams(unittest.TestCase):
    def test_stable_player_stays_on_one_team(self):
        assignment = [{1: 1}, {1: 1}, {1: 1}]
        self.assertEqual(resolve_player_teams(assignment), {1: 1})

    def test_majority_vote_across_a_reassignment(self):
        # Simulates TeamAssigner's confirmed 50-frame reset misclassifying once --
        # majority vote should still land on the correct team.
        assignment = [{1: 1}] * 40 + [{1: 2}] * 10
        self.assertEqual(resolve_player_teams(assignment), {1: 1})

    def test_multiple_players_independent(self):
        assignment = [{1: 1, 2: 2}, {1: 1, 2: 2}]
        self.assertEqual(resolve_player_teams(assignment), {1: 1, 2: 2})


class TestComputeTeamPossessionPct(unittest.TestCase):
    def test_even_split(self):
        result = compute_team_possession_pct([1, 1, 2, 2])
        self.assertEqual(result["team_1_pct"], 50.0)
        self.assertEqual(result["team_2_pct"], 50.0)
        self.assertEqual(result["undecided_pct"], 0.0)

    def test_includes_undecided_in_denominator(self):
        # Matches TeamBallControlDrawer's own definition: percentages are of ALL
        # frames, including undecided (-1) ones, not just decided frames.
        result = compute_team_possession_pct([1, -1, -1, -1])
        self.assertEqual(result["team_1_pct"], 25.0)
        self.assertEqual(result["undecided_pct"], 75.0)

    def test_empty_input(self):
        result = compute_team_possession_pct([])
        self.assertEqual(result, {"team_1_pct": 0.0, "team_2_pct": 0.0, "undecided_pct": 0.0})


class TestComputePlayerMovementStats(unittest.TestCase):
    def test_total_distance_sums_across_frames(self):
        distances = [{1: 2.0}, {1: 3.0}]
        speeds = [{1: 5.0}, {1: 7.0}]
        stats = compute_player_movement_stats(distances, speeds)
        self.assertAlmostEqual(stats[1]["total_distance_m"], 5.0)

    def test_zero_speed_samples_excluded_from_average(self):
        # calculate_speed's own default is 0.0 when there aren't enough samples yet --
        # that's "not measured", not "measured zero", so it must not drag the average down.
        distances = [{1: 1.0}]
        speeds = [{1: 0.0}, {1: 0.0}, {1: 10.0}]
        stats = compute_player_movement_stats(distances, speeds)
        self.assertAlmostEqual(stats[1]["avg_speed_kmh"], 10.0)
        self.assertAlmostEqual(stats[1]["max_speed_kmh"], 10.0)

    def test_player_with_no_valid_speed_samples_gets_zero(self):
        distances = [{1: 1.0}]
        speeds = [{1: 0.0}]
        stats = compute_player_movement_stats(distances, speeds)
        self.assertEqual(stats[1]["avg_speed_kmh"], 0.0)
        self.assertEqual(stats[1]["max_speed_kmh"], 0.0)


class TestComputeEventCounts(unittest.TestCase):
    def test_counts_per_team(self):
        passes = [-1, 1, 1, 2]
        interceptions = [-1, -1, 2, -1]
        result = compute_event_counts(passes, interceptions)
        self.assertEqual(result["passes"], {"team_1": 2, "team_2": 1})
        self.assertEqual(result["interceptions"], {"team_1": 0, "team_2": 1})


class TestBuildGameReport(unittest.TestCase):
    def test_full_report_shape_and_values(self):
        report = build_game_report(
            player_assignment=[{1: 1, 2: 2}, {1: 1, 2: 2}],
            ball_aquisition=[1, 2],
            passes=[-1, -1],
            interceptions=[-1, 2],
            tactical_player_positions=[{1: [0, 0], 2: [10, 10]}, {1: [3, 4], 2: [10, 10]}],
            player_distances_per_frame=[{}, {1: 5.0}],
            player_speed_per_frame=[{1: 0.0}, {1: 12.0}],
            team_ball_control=[1, 2],
        )
        self.assertEqual(report["num_frames"], 2)
        self.assertEqual(report["players"]["1"]["team"], 1)
        self.assertEqual(report["players"]["1"]["label"], "Player 1")
        self.assertAlmostEqual(report["players"]["1"]["total_distance_m"], 5.0)
        self.assertEqual(report["team_possession"]["team_1_pct"], 50.0)
        self.assertEqual(report["events"]["interceptions"], {"team_1": 0, "team_2": 1})

    def test_player_labels_override_default(self):
        report = build_game_report(
            player_assignment=[{1: 1}],
            ball_aquisition=[1],
            passes=[-1],
            interceptions=[-1],
            tactical_player_positions=[{1: [0, 0]}],
            player_distances_per_frame=[{}],
            player_speed_per_frame=[{}],
            team_ball_control=[1],
            player_labels={1: "#23"},
        )
        self.assertEqual(report["players"]["1"]["label"], "#23")

    def test_report_is_json_serializable(self):
        import json
        report = build_game_report(
            player_assignment=[{1: 1}],
            ball_aquisition=[1],
            passes=[-1],
            interceptions=[-1],
            tactical_player_positions=[{1: [0, 0]}],
            player_distances_per_frame=[{}],
            player_speed_per_frame=[{}],
            team_ball_control=[1],
        )
        json.dumps(report)  # raises if anything non-serializable slipped in (e.g. a numpy scalar)

    def _base_report_kwargs(self):
        return dict(
            player_assignment=[{1: 1}],
            ball_aquisition=[1],
            passes=[-1],
            interceptions=[-1],
            tactical_player_positions=[{1: [0, 0]}],
            player_distances_per_frame=[{}],
            player_speed_per_frame=[{}],
            team_ball_control=[1],
        )

    def test_build_game_report_without_violations_arg_has_no_violations_key(self):
        # violations defaults to None -- pose/violation detection didn't run at
        # all this pass, so "violations" shouldn't appear in the report at all
        # (distinct from an empty list, which means it ran and found nothing).
        report = build_game_report(**self._base_report_kwargs())
        self.assertNotIn("violations", report)

    def test_build_game_report_with_empty_violations_list_includes_empty_key(self):
        report = build_game_report(**self._base_report_kwargs(), violations=[])
        self.assertIn("violations", report)
        self.assertEqual(report["violations"], [])

    def test_build_game_report_with_violations_list_passes_through_verbatim(self):
        violations = [
            {"violation_type": "double_dribble", "player_id": 1, "start_frame": 10,
             "end_frame": 15, "confidence": "heuristic"},
        ]
        report = build_game_report(**self._base_report_kwargs(), violations=violations)
        self.assertEqual(report["violations"], violations)


if __name__ == "__main__":
    unittest.main()
