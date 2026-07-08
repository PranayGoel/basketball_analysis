import cv2

VIOLATION_LABELS = {"double_dribble": "DOUBLE DRIBBLE?", "traveling": "TRAVELING?"}
VIOLATION_BOX_COLOR = (0, 0, 255)      # red (BGR)
VIOLATION_TEXT_COLOR = (255, 255, 255)  # white


class ViolationDrawer:
    """
    Draws a transient, near-player flag for flagged rule-violation events
    (from rule_violation_detector) on the frames where each is active.

    Labels are phrased as QUESTIONS ("DOUBLE DRIBBLE?"), not assertions, so
    the heuristic/unverified nature of these detections stays visible in the
    rendered video itself, not just in code comments -- see
    rule_violation_detector/__init__.py for the full limitations these flags
    are subject to.
    """

    def _build_frame_index(self, violations):
        """
        Build a frame_num -> [violation, ...] lookup so draw() doesn't have
        to rescan the whole violations list per frame.

        Args:
            violations (list[dict]): Combined violation-event dicts from both
                detectors, each with "start_frame" and "end_frame".

        Returns:
            dict[int, list[dict]]: Active violations per frame.
        """
        frame_index = {}
        for violation in violations:
            for frame_num in range(violation["start_frame"], violation["end_frame"] + 1):
                frame_index.setdefault(frame_num, []).append(violation)
        return frame_index

    def draw(self, video_frames, violations, pose_tracks):
        """
        Args:
            video_frames (list[np.ndarray]): Frames to draw on.
            violations (list[dict]): Combined violation-event dicts from both
                detectors ({"violation_type", "player_id", "start_frame",
                "end_frame", "confidence"}).
            pose_tracks: PoseEstimator output -- used ONLY to look up the
                flagged player's bbox position each frame
                (pose_tracks[frame][player_id]["bbox"]), so the label draws
                near them. If that player has no pose entry that frame,
                drawing is skipped for that frame rather than guessing a
                position.

        Returns:
            list[np.ndarray]: A new list of frames with violation labels drawn.
        """
        frame_index = self._build_frame_index(violations)

        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            for violation in frame_index.get(frame_num, []):
                frame_poses = pose_tracks[frame_num] if frame_num < len(pose_tracks) else {}
                player_pose = frame_poses.get(violation["player_id"])
                if player_pose is None:
                    continue

                frame = self._draw_violation_label(frame, violation, player_pose["bbox"])

            output_video_frames.append(frame)

        return output_video_frames

    def _draw_violation_label(self, frame, violation, player_bbox):
        """
        Draw a small semi-transparent red box + white text label near the
        top of the flagged player's bounding box.

        Args:
            frame (numpy.ndarray): The frame to draw on (mutated in place).
            violation (dict): The active violation event.
            player_bbox (list): The flagged player's [x1, y1, x2, y2] bbox.

        Returns:
            numpy.ndarray: The same frame, with the label drawn on it.
        """
        label = VIOLATION_LABELS.get(violation["violation_type"], violation["violation_type"].upper() + "?")

        font_scale = 0.6
        font_thickness = 2
        padding = 6

        x1, y1, x2, _ = player_bbox
        (text_width, text_height), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
        )

        box_x1 = int(x1)
        box_y2 = int(y1)
        box_y1 = box_y2 - text_height - 2 * padding
        box_x2 = box_x1 + text_width + 2 * padding

        overlay = frame.copy()
        cv2.rectangle(overlay, (box_x1, box_y1), (box_x2, box_y2), VIOLATION_BOX_COLOR, -1)
        alpha = 0.7
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        text_x = box_x1 + padding
        text_y = box_y2 - padding
        cv2.putText(
            frame,
            label,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            VIOLATION_TEXT_COLOR,
            font_thickness,
        )

        return frame
