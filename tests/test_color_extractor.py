import unittest
import numpy as np
import cv2

from team_assigner.color_extractor import extract_jersey_patch, dominant_jersey_color


class TestExtractJerseyPatch(unittest.TestCase):
    def test_extract_jersey_patch_returns_top_half(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        top_color = (10, 20, 200)   # BGR
        bottom_color = (200, 20, 10)  # BGR, distinctly different

        bbox = [20, 10, 80, 90]
        x1, y1, x2, y2 = bbox
        mid_y = (y1 + y2) // 2

        frame[y1:mid_y, x1:x2] = top_color
        frame[mid_y:y2, x1:x2] = bottom_color

        patch = extract_jersey_patch(frame, bbox)

        self.assertGreater(patch.size, 0)
        mean_color = patch.reshape(-1, 3).mean(axis=0)

        top_distance = np.linalg.norm(mean_color - np.array(top_color, dtype=np.float64))
        bottom_distance = np.linalg.norm(mean_color - np.array(bottom_color, dtype=np.float64))

        self.assertLess(top_distance, bottom_distance)
        self.assertLess(top_distance, 5.0)

    def test_extract_jersey_patch_handles_degenerate_bbox(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        # Zero-area bbox.
        zero_area_patch = extract_jersey_patch(frame, [50, 50, 50, 50])
        self.assertEqual(zero_area_patch.size, 0)

        # Out-of-bounds bbox (entirely outside the frame).
        out_of_bounds_patch = extract_jersey_patch(frame, [200, 200, 300, 300])
        self.assertEqual(out_of_bounds_patch.size, 0)

        # Inverted bbox (x2 < x1).
        inverted_patch = extract_jersey_patch(frame, [80, 10, 20, 90])
        self.assertEqual(inverted_patch.size, 0)


class TestDominantJerseyColor(unittest.TestCase):
    def test_dominant_jersey_color_separates_majority_from_minority(self):
        # ~70% one solid BGR color, ~30% a different color (simulating noise/skin).
        majority_bgr = (40, 40, 220)   # a strong red-ish jersey color in BGR
        minority_bgr = (100, 180, 140)  # distinctly different (skin-ish tone)

        height, width = 100, 100
        patch = np.zeros((height, width, 3), dtype=np.uint8)
        patch[:, :] = majority_bgr

        minority_rows = int(height * 0.3)
        patch[:minority_rows, :] = minority_bgr

        result_hsv = dominant_jersey_color(patch)
        self.assertIsNotNone(result_hsv)

        majority_bgr_patch = np.uint8([[majority_bgr]])
        expected_hsv = cv2.cvtColor(majority_bgr_patch, cv2.COLOR_BGR2HSV)[0][0].astype(np.float64)

        # Compare only hue distance (with wraparound) since that's the discriminative channel.
        hue_diff = abs(float(result_hsv[0]) - float(expected_hsv[0]))
        hue_diff = min(hue_diff, 180.0 - hue_diff)
        self.assertLess(hue_diff, 10.0)

    def test_dominant_jersey_color_returns_none_on_empty_patch(self):
        empty_patch = np.empty((0, 0, 3), dtype=np.uint8)
        result = dominant_jersey_color(empty_patch)
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
