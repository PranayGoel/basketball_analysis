import cv2

COCO_SKELETON = [
    (5, 7), (7, 9), (6, 8), (8, 10),          # arms
    (5, 6), (5, 11), (6, 12), (11, 12),       # torso
    (11, 13), (13, 15), (12, 14), (14, 16),   # legs
    (0, 5), (0, 6),                            # neck-ish
]
MIN_KEYPOINT_CONF_TO_DRAW = 0.3
SKELETON_COLOR = (0, 255, 255)  # yellow (BGR) -- distinguishable from other overlays' colors
KEYPOINT_RADIUS = 3
LINE_THICKNESS = 2


class PoseDrawer:
    """
    A class responsible for drawing player skeleton overlays (keypoints + connecting
    lines) on a sequence of video frames, using pose estimation results.
    """

    def draw(self, video_frames, pose_tracks):
        """
        Draw skeleton overlays on a list of video frames.

        Only connects/draws keypoint pairs where both keypoints' confidence is
        at or above MIN_KEYPOINT_CONF_TO_DRAW.

        Args:
            video_frames (list): A list of frames (as NumPy arrays) on which to draw.
            pose_tracks (list): A list of dictionaries (one per frame) mapping
                player track_id to pose data, as produced by
                `PoseEstimator.get_object_poses` (each value has a "keypoints"
                list of [x, y, conf] entries in COCO order).

        Returns:
            list: A list of frames with skeleton lines and keypoint dots overlaid.
        """
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            frame_poses = pose_tracks[frame_num] if frame_num < len(pose_tracks) else {}

            for track_id, player_pose in frame_poses.items():
                frame = self.draw_pose(frame, player_pose)

            output_video_frames.append(frame)

        return output_video_frames

    def draw_pose(self, frame, player_pose):
        """
        Draw a single player's skeleton (keypoint dots + connecting lines) on a frame.

        Args:
            frame (numpy.ndarray): The frame to draw on.
            player_pose (dict): A single player's pose data with a "keypoints"
                list of [x, y, conf] entries in COCO order.

        Returns:
            numpy.ndarray: The frame with the player's skeleton drawn on it.
        """
        keypoints = player_pose.get("keypoints", [])

        # Draw connecting lines for skeleton pairs where both keypoints meet
        # the confidence threshold.
        for start_idx, end_idx in COCO_SKELETON:
            if start_idx >= len(keypoints) or end_idx >= len(keypoints):
                continue

            start_x, start_y, start_conf = keypoints[start_idx]
            end_x, end_y, end_conf = keypoints[end_idx]

            if start_conf < MIN_KEYPOINT_CONF_TO_DRAW or end_conf < MIN_KEYPOINT_CONF_TO_DRAW:
                continue

            cv2.line(
                frame,
                (int(start_x), int(start_y)),
                (int(end_x), int(end_y)),
                SKELETON_COLOR,
                LINE_THICKNESS,
            )

        # Draw keypoint dots for keypoints that meet the confidence threshold.
        for x, y, conf in keypoints:
            if conf < MIN_KEYPOINT_CONF_TO_DRAW:
                continue
            cv2.circle(frame, (int(x), int(y)), KEYPOINT_RADIUS, SKELETON_COLOR, -1)

        return frame
