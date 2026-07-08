import sys
import numpy as np
import cv2

sys.path.append('../')
from personal.basketball_analysis.utils import read_stub, save_stub
from personal.basketball_analysis.team_assigner.color_extractor import extract_jersey_patch, dominant_jersey_color


class TeamAssigner:
    """
    A class that assigns players to teams based on their jersey colors using
    k-means color clustering.

    Rather than classifying jerseys against two hardcoded color descriptions,
    this class discovers each team's actual jersey color by clustering observed
    jersey colors across a sample of frames, then classifies each player's
    jersey against those discovered team centroids -- so it works on any pair
    of team colors, not just white-vs-navy.

    Attributes:
        team_colors (dict): Maps team_id (1..n_teams) to its discovered HSV
            centroid color. Populated by `_discover_team_colors`.
        player_team_dict (dict): Maps player_id to their current team assignment.
    """

    def __init__(self,
                 n_teams=2,
                 discovery_frame_stride=5,
                 discovery_frame_limit=60,
                 vote_window=10,
                 ):
        """
        Initialize the TeamAssigner.

        Args:
            n_teams (int): Number of teams to discover/classify (default 2).
            discovery_frame_stride (int): Sample every Nth frame during the
                one-time team-color discovery pass.
            discovery_frame_limit (int): Maximum number of sampled frames to
                use during discovery.
            vote_window (int): Number of most-recent observations per player
                to keep for majority-vote team classification.
        """
        self.n_teams = n_teams
        self.discovery_frame_stride = discovery_frame_stride
        self.discovery_frame_limit = discovery_frame_limit
        self.vote_window = vote_window

        self.team_colors = {}
        self.player_team_dict = {}
        self._player_color_history = {}

    def _discover_team_colors(self, video_frames, player_tracks):
        """
        One-time pass that discovers each team's jersey color by clustering
        observed jersey colors across a sample of frames.

        Samples every `discovery_frame_stride`-th frame, up to
        `discovery_frame_limit` sampled frames, extracting each visible
        player's dominant jersey HSV color. All observations across all
        players are then clustered with k-means (k=`n_teams`) to find the
        team centroids, stored in `self.team_colors`.

        Args:
            video_frames (list): List of video frames.
            player_tracks (list): List of per-frame player tracking dicts,
                each mapping player_id -> {"bbox": [...]}.

        Raises:
            ValueError: If fewer than `n_teams` valid color observations are
                found -- this indicates the video/tracks don't have enough
                visible players to safely discover team colors, and silently
                producing centroids from too few points would be unreliable.
        """
        observations = []
        sampled_count = 0

        for frame_num, player_track in enumerate(player_tracks):
            if frame_num % self.discovery_frame_stride != 0:
                continue
            if sampled_count >= self.discovery_frame_limit:
                break
            sampled_count += 1

            if frame_num >= len(video_frames):
                continue
            frame = video_frames[frame_num]

            for _player_id, track in player_track.items():
                bbox = track['bbox']
                patch = extract_jersey_patch(frame, bbox)
                color_hsv = dominant_jersey_color(patch)
                if color_hsv is not None:
                    observations.append(color_hsv)

        if len(observations) < self.n_teams:
            raise ValueError(
                f"Not enough valid jersey color observations to discover "
                f"{self.n_teams} team colors (found {len(observations)}). "
                f"Check that player_tracks contains visible player bboxes."
            )

        observations_arr = np.array(observations, dtype=np.float32)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1.0)
        _, _labels, centers = cv2.kmeans(
            observations_arr, self.n_teams, None, criteria,
            attempts=10, flags=cv2.KMEANS_PP_CENTERS
        )

        self.team_colors = {
            team_id: centers[team_id - 1] for team_id in range(1, self.n_teams + 1)
        }

    def _classify_color(self, color_hsv):
        """
        Classify an HSV color against the discovered team centroids using
        nearest-centroid assignment.

        Hue is weighted most heavily since it's the most jersey-color-
        discriminative channel; value is weighted least since it's
        lighting-sensitive. Hue distance accounts for wraparound at 180
        (OpenCV's HSV hue range).

        Args:
            color_hsv (numpy.ndarray): HSV triplet (H, S, V) to classify.

        Returns:
            int: The team_id (1..n_teams) of the nearest team centroid.
        """
        hue_weight = 2.0
        sat_weight = 1.0
        val_weight = 0.5

        best_team_id = None
        best_distance = None

        for team_id, centroid in self.team_colors.items():
            hue_diff = abs(float(color_hsv[0]) - float(centroid[0]))
            hue_diff = min(hue_diff, 180.0 - hue_diff)  # wraparound at 180

            sat_diff = float(color_hsv[1]) - float(centroid[1])
            val_diff = float(color_hsv[2]) - float(centroid[2])

            distance = (
                hue_weight * (hue_diff ** 2)
                + sat_weight * (sat_diff ** 2)
                + val_weight * (val_diff ** 2)
            )

            if best_distance is None or distance < best_distance:
                best_distance = distance
                best_team_id = team_id

        return best_team_id

    def get_player_team(self, frame, player_bbox, player_id):
        """
        Gets the team assignment for a player using a rolling majority vote
        over their recent jersey color observations.

        Extracts the player's current-frame jersey color, appends it to that
        player's rolling history (capped at `vote_window` most recent
        observations), classifies each historical observation, and takes the
        majority vote across the window. If color extraction fails this
        frame (e.g. a degenerate crop), the player's prior assignment is kept
        rather than resetting or crashing. Players never successfully
        classified default to team 1.

        Args:
            frame (numpy.ndarray): The video frame containing the player.
            player_bbox (list): Bounding box coordinates of the player.
            player_id (int): Unique identifier for the player.

        Returns:
            int: Team ID (1..n_teams) assigned to the player.
        """
        patch = extract_jersey_patch(frame, player_bbox)
        color_hsv = dominant_jersey_color(patch)

        if color_hsv is not None:
            history = self._player_color_history.setdefault(player_id, [])
            history.append(color_hsv)
            if len(history) > self.vote_window:
                del history[0:len(history) - self.vote_window]

        history = self._player_color_history.get(player_id)
        if not history:
            # Never successfully classified -- default to team 1, don't crash.
            team_id = self.player_team_dict.get(player_id, 1)
            self.player_team_dict[player_id] = team_id
            return team_id

        votes = [self._classify_color(observed_color) for observed_color in history]
        vote_counts = {}
        for vote in votes:
            vote_counts[vote] = vote_counts.get(vote, 0) + 1

        team_id = max(vote_counts.items(), key=lambda item: item[1])[0]

        self.player_team_dict[player_id] = team_id
        return team_id

    def get_player_teams_across_frames(self, video_frames, player_tracks, read_from_stub=False, stub_path=None):
        """
        Processes all video frames to assign teams to players, with optional caching.

        Args:
            video_frames (list): List of video frames to process.
            player_tracks (list): List of player tracking information for each frame.
            read_from_stub (bool): Whether to attempt reading cached results.
            stub_path (str): Path to the cache file.

        Returns:
            list: List of dictionaries mapping player IDs to team assignments for each frame.
        """

        player_assignment = read_stub(read_from_stub, stub_path)
        if player_assignment is not None:
            if len(player_assignment) == len(video_frames):
                return player_assignment

        self._discover_team_colors(video_frames, player_tracks)

        player_assignment = []
        for frame_num, player_track in enumerate(player_tracks):
            player_assignment.append({})

            for player_id, track in player_track.items():
                team = self.get_player_team(video_frames[frame_num],
                                             track['bbox'],
                                             player_id)
                player_assignment[frame_num][player_id] = team

        save_stub(stub_path, player_assignment)

        return player_assignment
