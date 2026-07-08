import unittest
from unittest.mock import patch

from personal.basketball_analysis.trackers.ball_tracker import BallTracker


def make_tracker():
    """
    Construct a BallTracker without loading a real YOLO model.

    BallTracker.__init__ unconditionally calls YOLO(model_path), so we patch
    the YOLO symbol imported into trackers.ball_tracker for the duration of
    construction. Tests then drive get_object_tracks_with_kalman() directly
    against synthetic per-frame candidate data by monkeypatching
    get_all_ball_candidates -- no real model, frames, or YOLO calls involved.
    """
    with patch('trackers.ball_tracker.YOLO'):
        return BallTracker('dummy_model_path.pt')


def candidate(bbox, confidence):
    return {'bbox': bbox, 'confidence': confidence}


class TestBallTrackerKalmanIntegration(unittest.TestCase):
    def test_prefers_closer_candidate_when_confidences_are_close(self):
        tracker = make_tracker()

        # Establish a clean, unambiguous rightward-moving ball for a few frames
        # so the Kalman filter has a confident motion estimate by the time the
        # ambiguous frame arrives. Ball moves +10px in x per frame.
        setup_frames = [
            [candidate([0, 0, 10, 10], 0.9)],
            [candidate([10, 0, 20, 10], 0.9)],
            [candidate([20, 0, 30, 10], 0.9)],
            [candidate([30, 0, 40, 10], 0.9)],
        ]

        # Ambiguous frame: two close-confidence candidates (0.31 vs 0.30, within
        # the default 0.15 margin). Predicted position by now is near x=40-50.
        # "near" candidate continues the established rightward motion; "far" is
        # a false-positive-like outlier elsewhere on the frame.
        ambiguous_frame = [
            candidate([300, 300, 310, 310], 0.31),  # far from prediction
            candidate([40, 0, 50, 10], 0.30),        # near the predicted trajectory
        ]

        all_frames = setup_frames + [ambiguous_frame]

        with patch.object(tracker, 'get_all_ball_candidates', return_value=all_frames):
            tracks = tracker.get_object_tracks_with_kalman(
                frames=[None] * len(all_frames), read_from_stub=False, stub_path=None
            )

        last_track = tracks[-1]
        self.assertIn(1, last_track)
        chosen_bbox = last_track[1]['bbox']
        # The near candidate's bbox should be chosen, not the far outlier.
        self.assertEqual(chosen_bbox, [40, 0, 50, 10])

    def test_rejects_high_confidence_outlier_far_from_prediction(self):
        tracker = make_tracker()

        # Establish a stable, stationary-ish ball so the prediction stays
        # tightly around (5, 5)-ish.
        setup_frames = [
            [candidate([0, 0, 10, 10], 0.9)],
            [candidate([1, 1, 11, 11], 0.9)],
            [candidate([2, 2, 12, 12], 0.9)],
            [candidate([3, 3, 13, 13], 0.9)],
        ]

        # A single, very-high-confidence candidate implausibly far away --
        # simulating a scoreboard/logo false positive.
        outlier_frame = [candidate([900, 900, 910, 910], 0.99)]

        all_frames = setup_frames + [outlier_frame]

        with patch.object(tracker, 'get_all_ball_candidates', return_value=all_frames):
            tracks = tracker.get_object_tracks_with_kalman(
                frames=[None] * len(all_frames), read_from_stub=False, stub_path=None,
                # default max_reasonable_distance_per_frame=60.0 on the tracker
            )

        last_track = tracks[-1]
        if last_track:
            # If populated, it must be the motion-predicted reconstruction,
            # never the outlier's bbox.
            self.assertEqual(last_track[1]['source'], 'predicted')
            self.assertNotEqual(last_track[1]['bbox'], [900, 900, 910, 910])
        # An empty dict (no data survived) is also an acceptable rejection outcome.

    def test_output_shape_matches_legacy_chain(self):
        tracker = make_tracker()

        # Synthetic sequence with exactly one candidate per frame (equivalent to
        # what get_object_tracks' max-confidence-per-frame selection would have
        # already picked), including one occlusion gap in the middle.
        single_candidate_frames = [
            [candidate([0, 0, 10, 10], 0.9)],
            [candidate([10, 0, 20, 10], 0.9)],
            [],  # occlusion / no detection this frame
            [candidate([30, 0, 40, 10], 0.9)],
            [candidate([40, 0, 50, 10], 0.9)],
        ]

        # (a) Legacy chain: feed the equivalent pre-selected single-candidate
        # data directly into the legacy methods.
        legacy_positions = [
            ({1: {"bbox": c[0]['bbox']}} if c else {})
            for c in single_candidate_frames
        ]
        legacy_positions = tracker.remove_wrong_detections(legacy_positions)
        legacy_result = tracker.interpolate_ball_positions(legacy_positions)

        # (b) New Kalman-gated pipeline, fed the same synthetic candidates.
        with patch.object(tracker, 'get_all_ball_candidates', return_value=single_candidate_frames):
            kalman_result = tracker.get_object_tracks_with_kalman(
                frames=[None] * len(single_candidate_frames), read_from_stub=False, stub_path=None
            )

        self.assertEqual(len(legacy_result), len(kalman_result))

        for legacy_frame, kalman_frame in zip(legacy_result, kalman_result):
            # Ignore the additive "source" key -- only compare the bbox contract.
            self.assertIn(1, legacy_frame)
            self.assertIn(1, kalman_frame)
            self.assertIn('bbox', legacy_frame[1])
            self.assertIn('bbox', kalman_frame[1])
            self.assertEqual(len(legacy_frame[1]['bbox']), 4)
            self.assertEqual(len(kalman_frame[1]['bbox']), 4)


if __name__ == "__main__":
    unittest.main()
