"""
storage.py: the ONLY module that knows the on-disk layout under DATA_DIR.

Every other module that needs a path for a given video_id (worker task,
ingestion service, routes serving/streaming files) calls get_paths() rather
than constructing paths inline -- if the layout ever changes, this is the one
file to touch.

Layout:
    DATA_DIR/uploads/{video_id}{ext}          -- the original uploaded file
    DATA_DIR/outputs/{video_id}.mp4           -- annotated output video (H.264/MP4 --
                                                  browser-playable; see utils/video_utils.py)
    DATA_DIR/reports/{video_id}.json          -- full game_report JSON
    DATA_DIR/thumbnails/{video_id}.jpg        -- single JPEG frame for the library card
"""

import os
import shutil
from dataclasses import dataclass
from typing import BinaryIO, Optional

from personal.basketball_analysis.webapp.backend.app.config import settings


@dataclass
class VideoPaths:
    video_id: str
    upload_path: str
    output_path: str
    report_path: str
    thumbnail_path: str


def _ensure_dirs() -> None:
    os.makedirs(settings.uploads_dir, exist_ok=True)
    os.makedirs(settings.outputs_dir, exist_ok=True)
    os.makedirs(settings.reports_dir, exist_ok=True)
    os.makedirs(settings.thumbnails_dir, exist_ok=True)


def get_paths(video_id: str, upload_ext: str = ".mp4") -> VideoPaths:
    """
    Deterministic path set for a given video_id. Does not touch the
    filesystem beyond ensuring the parent directories exist -- callers decide
    when to actually create/write/delete the files themselves.
    """
    _ensure_dirs()
    return VideoPaths(
        video_id=video_id,
        upload_path=os.path.join(settings.uploads_dir, f"{video_id}{upload_ext}"),
        output_path=os.path.join(settings.outputs_dir, f"{video_id}.mp4"),
        report_path=os.path.join(settings.reports_dir, f"{video_id}.json"),
        thumbnail_path=os.path.join(settings.thumbnails_dir, f"{video_id}.jpg"),
    )


def save_upload(video_id: str, filename: str, file_obj: BinaryIO) -> VideoPaths:
    """
    Stream an uploaded file to disk at its deterministic upload path.

    Args:
        video_id: the newly-generated uuid4 hex id for this video.
        filename: the client-supplied original filename, used only to derive
            the extension (kept, e.g. ".mp4"/".mov") -- never used as the
            actual path component, so there's no path-traversal surface from
            a hostile filename.
        file_obj: a file-like object opened for binary reading (e.g.
            UploadFile.file from FastAPI's multipart parsing).

    Returns:
        VideoPaths: the resolved path set, with upload_path now populated.
    """
    ext = os.path.splitext(filename)[1] or ".mp4"
    paths = get_paths(video_id, upload_ext=ext)
    with open(paths.upload_path, "wb") as out:
        shutil.copyfileobj(file_obj, out)
    return paths


def delete_video_files(video_id: str) -> None:
    """
    Best-effort delete of every file that might exist for a video_id.
    Missing files are not an error -- a video that failed before its output
    video was ever written should still delete cleanly.
    """
    # upload extension is unknown at delete time without a DB lookup, so glob
    # for any extension under uploads/ rather than assuming .mp4.
    uploads_dir = settings.uploads_dir
    if os.path.isdir(uploads_dir):
        for name in os.listdir(uploads_dir):
            if os.path.splitext(name)[0] == video_id:
                _remove_if_exists(os.path.join(uploads_dir, name))

    paths = get_paths(video_id)
    _remove_if_exists(paths.output_path)
    _remove_if_exists(paths.report_path)
    _remove_if_exists(paths.thumbnail_path)


def generate_thumbnail(video_id: str, source_video_path: str, seek_seconds: float = 5.0) -> Optional[str]:
    """
    Extract a single JPEG frame from source_video_path at ~seek_seconds and
    write it to the deterministic thumbnail path for this video_id.

    Returns the thumbnail path on success, None on any failure (missing
    video, opencv not available, etc.) -- thumbnail absence is non-fatal.
    """
    try:
        import cv2  # opencv-python-headless is in the pipeline requirements
    except ImportError:
        return None

    if not os.path.isfile(source_video_path):
        return None

    cap = cv2.VideoCapture(source_video_path)
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        target_frame = int(fps * seek_seconds)
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        ok, frame = cap.read()
        if not ok:
            # Seek past end of video (very short clip) — try frame 0 instead.
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = cap.read()
        if not ok:
            return None

        paths = get_paths(video_id)
        ok = cv2.imwrite(paths.thumbnail_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return paths.thumbnail_path if ok else None
    finally:
        cap.release()


def _remove_if_exists(path: Optional[str]) -> None:
    if path and os.path.isfile(path):
        os.remove(path)
