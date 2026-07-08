"""
report_indexer tests: feed synthetic game_report.py-shaped dicts into
index_report() and assert the resulting Player/GameEvent/Violation rows and
Video flattened columns, including the violations-absent case (must not
crash, has_violations stays False, no Violation rows).
"""

from datetime import datetime, timezone

from personal.basketball_analysis.webapp.backend.app.db.models import GameEvent, Player, Video, Violation
from personal.basketball_analysis.webapp.backend.app.services.report_indexer import index_report
from personal.basketball_analysis.webapp.backend.tests.test_utils import BackendTestCase


def _synthetic_report(include_violations=None):
    """
    include_violations: None (key absent -- pose analysis didn't run),
    [] (ran, found nothing), or a list of violation dicts (ran, found some).
    """
    report = {
        "players": {
            "1": {"label": "Player 1", "team": 1, "total_distance_m": 120.5, "avg_speed_kmh": 8.2, "max_speed_kmh": 22.1},
            "2": {"label": "Player 2", "team": 2, "total_distance_m": 95.0, "avg_speed_kmh": 6.5, "max_speed_kmh": 18.4},
        },
        "team_possession": {"team_1_pct": 55.0, "team_2_pct": 40.0, "undecided_pct": 5.0},
        "events": {
            "passes": {"team_1": 12, "team_2": 8},
            "interceptions": {"team_1": 2, "team_2": 3},
        },
        "num_frames": 1500,
    }
    if include_violations is not None:
        report["violations"] = include_violations
    return report


class TestIndexReportBasicFields(BackendTestCase):
    def _insert_bare_video(self, video_id="v1"):
        db = self.db_session()
        try:
            db.add(
                Video(
                    id=video_id,
                    filename="game.mp4",
                    uploaded_at=datetime.now(timezone.utc),
                    upload_path=f"/tmp/{video_id}.mp4",
                    status="processing",
                    has_violations=False,
                )
            )
            db.commit()
        finally:
            db.close()

    def test_raises_for_unknown_video_id(self):
        db = self.db_session()
        try:
            with self.assertRaises(ValueError):
                index_report(db, "does-not-exist", _synthetic_report())
        finally:
            db.close()

    def test_creates_player_rows(self):
        self._insert_bare_video()
        db = self.db_session()
        try:
            index_report(db, "v1", _synthetic_report())
            players = db.query(Player).filter(Player.video_id == "v1").order_by(Player.tracker_player_id).all()
            self.assertEqual(len(players), 2)
            self.assertEqual(players[0].tracker_player_id, 1)
            self.assertEqual(players[0].label, "Player 1")
            self.assertEqual(players[0].team, 1)
            self.assertAlmostEqual(players[0].total_distance_m, 120.5)
            self.assertAlmostEqual(players[1].max_speed_kmh, 18.4)
        finally:
            db.close()

    def test_creates_game_event_rows_for_nonzero_counts(self):
        self._insert_bare_video()
        db = self.db_session()
        try:
            index_report(db, "v1", _synthetic_report())
            events = db.query(GameEvent).filter(GameEvent.video_id == "v1").all()
            event_pairs = {(e.event_type, e.team) for e in events}
            self.assertEqual(
                event_pairs,
                {("pass", 1), ("pass", 2), ("interception", 1), ("interception", 2)},
            )
        finally:
            db.close()

    def test_skips_game_event_rows_for_zero_counts(self):
        self._insert_bare_video()
        report = _synthetic_report()
        report["events"]["passes"]["team_2"] = 0
        db = self.db_session()
        try:
            index_report(db, "v1", report)
            events = db.query(GameEvent).filter(GameEvent.video_id == "v1", GameEvent.event_type == "pass").all()
            teams = {e.team for e in events}
            self.assertEqual(teams, {1})
        finally:
            db.close()

    def test_flattens_video_summary_columns(self):
        self._insert_bare_video()
        db = self.db_session()
        try:
            index_report(db, "v1", _synthetic_report())
            video = db.get(Video, "v1")
            self.assertAlmostEqual(video.team_a_possession_pct, 55.0)
            self.assertAlmostEqual(video.team_b_possession_pct, 40.0)
            self.assertEqual(video.player_count, 2)
            self.assertEqual(video.total_passes, 20)
            self.assertEqual(video.total_interceptions, 5)
            self.assertAlmostEqual(video.max_player_speed_kmh, 22.1)
            self.assertAlmostEqual(video.max_player_distance_m, 120.5)
        finally:
            db.close()

    def test_reindexing_does_not_duplicate_rows(self):
        self._insert_bare_video()
        db = self.db_session()
        try:
            index_report(db, "v1", _synthetic_report())
            index_report(db, "v1", _synthetic_report())
            players = db.query(Player).filter(Player.video_id == "v1").all()
            self.assertEqual(len(players), 2)
        finally:
            db.close()


class TestIndexReportViolations(BackendTestCase):
    def _insert_bare_video(self, video_id="v1"):
        db = self.db_session()
        try:
            db.add(
                Video(
                    id=video_id,
                    filename="game.mp4",
                    uploaded_at=datetime.now(timezone.utc),
                    upload_path=f"/tmp/{video_id}.mp4",
                    status="processing",
                    has_violations=False,
                )
            )
            db.commit()
        finally:
            db.close()

    def test_violations_absent_does_not_crash_and_has_violations_stays_false(self):
        self._insert_bare_video()
        db = self.db_session()
        try:
            index_report(db, "v1", _synthetic_report(include_violations=None))
            video = db.get(Video, "v1")
            self.assertFalse(video.has_violations)
            self.assertEqual(db.query(Violation).filter(Violation.video_id == "v1").count(), 0)
        finally:
            db.close()

    def test_violations_empty_list_does_not_crash_and_has_violations_stays_false(self):
        self._insert_bare_video()
        db = self.db_session()
        try:
            index_report(db, "v1", _synthetic_report(include_violations=[]))
            video = db.get(Video, "v1")
            self.assertFalse(video.has_violations)
            self.assertEqual(db.query(Violation).filter(Violation.video_id == "v1").count(), 0)
        finally:
            db.close()

    def test_violations_present_creates_rows_and_sets_flag(self):
        self._insert_bare_video()
        violations = [
            {
                "violation_type": "double_dribble",
                "player_id": 1,
                "start_frame": 100,
                "end_frame": 110,
                "confidence": "heuristic",
            },
            {
                "violation_type": "traveling",
                "player_id": 2,
                "start_frame": 200,
                "end_frame": 205,
                "confidence": "heuristic",
            },
        ]
        db = self.db_session()
        try:
            index_report(db, "v1", _synthetic_report(include_violations=violations))
            video = db.get(Video, "v1")
            self.assertTrue(video.has_violations)
            rows = db.query(Violation).filter(Violation.video_id == "v1").order_by(Violation.start_frame).all()
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0].violation_type, "double_dribble")
            self.assertEqual(rows[0].player_id, 1)
            self.assertEqual(rows[0].start_frame, 100)
            self.assertEqual(rows[0].end_frame, 110)
            self.assertEqual(rows[0].confidence, "heuristic")
            self.assertEqual(rows[1].violation_type, "traveling")
        finally:
            db.close()
