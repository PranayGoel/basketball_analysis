"""
Pure color-extraction logic used by TeamAssigner to identify a player's jersey color.

Uses only numpy and cv2 (k-means color clustering) -- no ML framework, no network calls.
This replaces the previous CLIP zero-shot classification approach, which hardcoded two
fixed jersey-color strings ("white shirt" / "dark blue shirt") and broke on any video
where the two teams didn't match those exact colors.
"""

import cv2
import numpy as np


def extract_jersey_patch(frame: np.ndarray, bbox: list) -> np.ndarray:
    """
    Crop the jersey/torso region from a player's bounding box.

    Takes the top half of the bbox (torso/jersey area, avoiding shorts and the
    court floor beneath the player) and insets horizontally by ~15% on each side
    to avoid bleeding in arms or background pixels near the box edges.

    Args:
        frame (numpy.ndarray): The video frame containing the player (BGR).
        bbox (list): Bounding box [x1, y1, x2, y2] in frame pixel coordinates.

    Returns:
        numpy.ndarray: The cropped BGR jersey patch. May be a small or empty
            array if the bbox is degenerate (zero-area, out of bounds, etc.) --
            callers must handle that case rather than assume a valid patch.
    """
    x1, y1, x2, y2 = bbox

    frame_h, frame_w = frame.shape[0], frame.shape[1]

    # Clip bbox to frame bounds defensively.
    x1 = max(0, min(x1, frame_w))
    x2 = max(0, min(x2, frame_w))
    y1 = max(0, min(y1, frame_h))
    y2 = max(0, min(y2, frame_h))

    if x2 <= x1 or y2 <= y1:
        return np.empty((0, 0, 3), dtype=frame.dtype)

    box_width = x2 - x1
    box_height = y2 - y1

    # Top half only: torso/jersey, not shorts or court floor.
    top_half_y2 = y1 + box_height / 2

    # Inset horizontally by ~15% each side to avoid arms/background bleed.
    inset = box_width * 0.15
    inset_x1 = x1 + inset
    inset_x2 = x2 - inset

    inset_x1_i = int(round(inset_x1))
    inset_x2_i = int(round(inset_x2))
    y1_i = int(round(y1))
    top_half_y2_i = int(round(top_half_y2))

    # Re-clip after inset math in case of tiny/degenerate boxes.
    inset_x1_i = max(0, min(inset_x1_i, frame_w))
    inset_x2_i = max(0, min(inset_x2_i, frame_w))
    y1_i = max(0, min(y1_i, frame_h))
    top_half_y2_i = max(0, min(top_half_y2_i, frame_h))

    if inset_x2_i <= inset_x1_i or top_half_y2_i <= y1_i:
        return np.empty((0, 0, 3), dtype=frame.dtype)

    patch = frame[y1_i:top_half_y2_i, inset_x1_i:inset_x2_i]
    return patch


def dominant_jersey_color(patch_bgr: np.ndarray, k: int = 2) -> np.ndarray:
    """
    Determine the dominant jersey color of a cropped BGR patch via k-means clustering.

    Converts the patch to HSV, masks out pixels that look like skin tone or court
    floor (green/yellow), then clusters the surviving pixels with k-means and
    returns the centroid of the largest cluster by pixel count (jersey fabric
    should dominate the masked patch). If fewer than 10 pixels survive masking,
    falls back to clustering all pixels unmasked -- a slightly noisier answer is
    better than none.

    Args:
        patch_bgr (numpy.ndarray): Cropped BGR jersey patch.
        k (int): Number of clusters for k-means (default 2).

    Returns:
        numpy.ndarray | None: The dominant color as an HSV triplet (H, S, V), or
            None if the patch is empty or has fewer than `k` pixels to cluster.
    """
    if patch_bgr is None or patch_bgr.size == 0:
        return None

    if patch_bgr.ndim != 3 or patch_bgr.shape[0] == 0 or patch_bgr.shape[1] == 0:
        return None

    hsv_patch = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2HSV)
    pixels = hsv_patch.reshape(-1, 3).astype(np.float32)

    if pixels.shape[0] < k:
        return None

    hue = pixels[:, 0]
    sat = pixels[:, 1]
    val = pixels[:, 2]

    # Skin tone: low hue, moderate-high saturation, high value (OpenCV hue range 0-180).
    skin_mask = (hue < 25) & (sat > 40) & (sat < 180) & (val > 80)

    # Court floor: green/yellow band, low-moderate saturation.
    floor_mask = (hue > 25) & (hue < 95) & (sat < 130)

    exclude_mask = skin_mask | floor_mask
    keep_mask = ~exclude_mask

    masked_pixels = pixels[keep_mask]

    if masked_pixels.shape[0] < 10:
        # Not enough surviving pixels -- fall back to using everything unmasked.
        cluster_pixels = pixels
    else:
        cluster_pixels = masked_pixels

    if cluster_pixels.shape[0] < k:
        return None

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(
        cluster_pixels, k, None, criteria, attempts=10, flags=cv2.KMEANS_PP_CENTERS
    )

    labels = labels.flatten()
    counts = np.bincount(labels, minlength=k)
    largest_cluster_idx = int(np.argmax(counts))

    return centers[largest_cluster_idx]
