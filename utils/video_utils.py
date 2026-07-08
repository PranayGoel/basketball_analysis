"""
A module for reading and writing video files.

This module provides utility functions to load video frames into memory and save
processed frames back to video files, with support for common video formats.
"""

import os

# cv2 is imported lazily inside each function below (not at module load time) so
# that other modules importing through the `utils` package -- e.g. bbox_utils,
# speed_and_distance_calculator, which need none of OpenCV's functionality -- don't
# require opencv-python to be installed just to be imported. Real functional need
# for cv2 only arises when these functions are actually called.

def read_video(video_path):
    """
    Read all frames from a video file into memory.

    Args:
        video_path (str): Path to the input video file.

    Returns:
        list: List of video frames as numpy arrays.
    """
    import cv2
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    return frames

def get_video_fps(video_path, default_fps=24):
    """
    Read the source video's frame rate.

    Kept separate from read_video (rather than changing its return shape) so existing
    callers of read_video are unaffected.

    Args:
        video_path (str): Path to the input video file.
        default_fps (float): Fallback used if the container reports no/invalid fps
            (some formats/codecs don't expose this reliably via OpenCV).

    Returns:
        float: The source video's fps, or default_fps if it couldn't be read.
    """
    import cv2
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if not fps or fps <= 0:
        return default_fps
    return fps

def save_video(ouput_video_frames,output_video_path,fps=24):
    """
    Save a sequence of frames as a video file.

    Creates necessary directories if they don't exist and writes frames using
    the H.264 codec (fourcc 'avc1') in an MP4 container -- natively playable
    in every major browser's <video> element, which the original 'XVID'/.avi
    output was not (no browser decodes AVI/XviD, so the webapp's video player
    would silently fail to load any processed video; verified empirically on
    this machine that this OpenCV build's bundled FFmpeg can both write and
    read back real H.264 via the 'avc1' fourcc tag -- 'h264'/'H264' fourcc
    strings work too but OpenCV's FFmpeg backend just remaps them to 'avc1'
    internally with a logged fallback message, so 'avc1' is used directly).

    Args:
        ouput_video_frames (list): List of frames to save.
        output_video_path (str): Path where the video should be saved.
            Should use a .mp4 extension to match the actual container format.
        fps (float): Output frame rate. Defaults to 24 to preserve prior behavior for
            any existing callers that don't pass the source video's real fps.
    """
    import cv2
    # If folder doesn't exist, create it
    if not os.path.exists(os.path.dirname(output_video_path)):
        os.makedirs(os.path.dirname(output_video_path))

    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (ouput_video_frames[0].shape[1], ouput_video_frames[0].shape[0]))
    for frame in ouput_video_frames:
        out.write(frame)
    out.release()