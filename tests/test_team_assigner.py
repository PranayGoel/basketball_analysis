import unittest
import numpy as np

from personal.basketball_analysis.team_assigner.team_assigner import TeamAssigner


class TestDiscoverTeamColors(unittest.TestCase):
    def setUp(self):
        self.assigner = TeamAssigner(discovery_frame_stride=1, discovery_frame_limit=60)

    def _make_frame_with_regions(self, red_bbox, blue_bbox):
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        frame[:, :] = (0, 0, 0)

        # Solid red region (BGR) for "team A" players.
        rx1, ry1, rx2, ry2 = red_bbox
        frame[ry1:ry2, rx1:rx2] = (30, 30, 220)

        # Solid blue region (BGR) for "team B" players.
        bx1, by1, bx2, by2 = blue_bbox
        frame[by1:by2, bx1:bx2] = (220, 30, 30)

        return frame

    def test_discover_team_colors_separates_two_synthetic_teams(self):
        red_bbox = [10, 10, 60, 90]
        blue_bbox = [110, 10, 160, 90]

        video_frames = []
        player_tracks = []

        num_frames = 20
        for _ in range(num_frames):
            frame = self._make_frame_with_regions(red_bbox, blue_bbox)
            video_frames.append(frame)
            player_tracks.append({
                1: {"bbox": red_bbox},
                2: {"bbox": blue_bbox},
            })

        self.assigner._discover_team_colors(video_frames, player_tracks)

        self.assertEqual(len(self.assigner.team_colors), 2)

        # Reference hues for pure red and pure blue (via OpenCV BGR->HSV mapping).
        import cv2
        red_hsv = cv2.cvtColor(np.uint8([[[30, 30, 220]]]), cv2.COLOR_BGR2HSV)[0][0]
        blue_hsv = cv2.cvtColor(np.uint8([[[220, 30, 30]]]), cv2.COLOR_BGR2HSV)[0][0]

        def hue_distance(h1, h2):
            diff = abs(float(h1) - float(h2))
            return min(diff, 180.0 - diff)

        centroids = list(self.assigner.team_colors.values())
        centroid_hues = [c[0] for c in centroids]

        # Each centroid should be closer to one reference hue than the other,
        # and the two centroids should be closer to different references
        # (i.e. not both collapsed onto the same team color).
        distances_to_red = [hue_distance(h, red_hsv[0]) for h in centroid_hues]
        distances_to_blue = [hue_distance(h, blue_hsv[0]) for h in centroid_hues]

        closest_ref = [
            "red" if distances_to_red[i] < distances_to_blue[i] else "blue"
            for i in range(2)
        ]
        self.assertEqual(set(closest_ref), {"red", "blue"})

        # And each centroid should be reasonably close to its assigned reference.
        for i in range(2):
            self.assertLess(min(distances_to_red[i], distances_to_blue[i]), 20.0)

    def test_discover_team_colors_raises_on_insufficient_observations(self):
        # Zero observations.
        with self.assertRaises(ValueError):
            self.assigner._discover_team_colors([], [])

        # A single observation is not enough for n_teams=2.
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[10:90, 10:90] = (30, 30, 220)
        video_frames = [frame]
        player_tracks = [{1: {"bbox": [10, 10, 90, 90]}}]

        with self.assertRaises(ValueError):
            self.assigner._discover_team_colors(video_frames, player_tracks)


class TestGetPlayerTeam(unittest.TestCase):
    def setUp(self):
        self.assigner = TeamAssigner(vote_window=10)

        import cv2
        red_hsv = cv2.cvtColor(np.uint8([[[30, 30, 220]]]), cv2.COLOR_BGR2HSV)[0][0]
        blue_hsv = cv2.cvtColor(np.uint8([[[220, 30, 30]]]), cv2.COLOR_BGR2HSV)[0][0]

        # Manually seed team_colors to isolate get_player_team from discovery.
        self.assigner.team_colors = {
            1: red_hsv.astype(np.float32),
            2: blue_hsv.astype(np.float32),
        }

    def _solid_frame(self, bgr_color, size=100):
        frame = np.zeros((size, size, 3), dtype=np.uint8)
        frame[:, :] = bgr_color
        return frame

    def test_get_player_team_stable_after_vote_window_warmup(self):
        bbox = [10, 10, 90, 90]
        red_frame = self._solid_frame((30, 30, 220))

        num_frames = self.assigner.vote_window + 5
        assignments = []
        for _ in range(num_frames):
            team_id = self.assigner.get_player_team(red_frame, bbox, player_id=1)
            assignments.append(team_id)

        # After warm-up (vote_window frames), assignment should be stable (no flips).
        warmed_up = assignments[self.assigner.vote_window - 1:]
        self.assertTrue(all(a == warmed_up[0] for a in warmed_up))
        self.assertEqual(warmed_up[0], 1)

    def test_get_player_team_majority_vote_damps_single_frame_noise(self):
        bbox = [10, 10, 90, 90]
        red_frame = self._solid_frame((30, 30, 220))
        blue_frame = self._solid_frame((220, 30, 30))

        # Mostly red (team 1) with 2 single-frame blue outliers mixed in.
        sequence = [red_frame] * 6 + [blue_frame] + [red_frame] * 2 + [blue_frame] + [red_frame] * 3

        final_team_id = None
        for frame in sequence:
            final_team_id = self.assigner.get_player_team(frame, bbox, player_id=1)

        self.assertEqual(final_team_id, 1)


if __name__ == "__main__":
    unittest.main()
