"""
Video routes: upload, library list/filter/sort, detail, delete, and Range-
aware streaming of the annotated output video.
"""

import os
import re
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, UploadFile
from fastapi.responses import Response, StreamingResponse
from sqlalchemy import func
from sqlalchemy.orm import Session

from personal.basketball_analysis.webapp.backend.app.api.deps import get_db
from personal.basketball_analysis.webapp.backend.app.db.models import Video
from personal.basketball_analysis.webapp.backend.app.schemas.video import UploadResponse, VideoDetail, VideoListResponse, VideoSummary
from personal.basketball_analysis.webapp.backend.app.services.ingestion import ingest_upload
from personal.basketball_analysis.webapp.backend.app.services.storage import delete_video_files

router = APIRouter(prefix="/api", tags=["videos"])

_SORTABLE_COLUMNS = {
    "uploaded_at": Video.uploaded_at,
    "total_passes": Video.total_passes,
    "max_player_speed_kmh": Video.max_player_speed_kmh,
}

_CHUNK_SIZE = 1024 * 1024  # 1 MiB read chunks for the Range stream


@router.post("/videos", response_model=UploadResponse, status_code=201)
async def upload_video(file: UploadFile, db: Session = Depends(get_db)):
    video, job = await ingest_upload(db, file.filename or "upload.mp4", file.file)
    return UploadResponse(video_id=video.id, job_id=job.id, status=video.status)


@router.get("/videos", response_model=VideoListResponse)
def list_videos(
    db: Session = Depends(get_db),
    status: Optional[str] = None,
    min_possession_split: Optional[float] = Query(
        None,
        description=(
            "Filters to videos where abs(team_a_possession_pct - 50) <= X, i.e. "
            "'closer than X points to an even split'."
        ),
    ),
    sort_by: str = Query("uploaded_at", enum=list(_SORTABLE_COLUMNS)),
    sort_order: str = Query("desc", enum=["asc", "desc"]),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=200),
):
    query = db.query(Video)

    if status:
        query = query.filter(Video.status == status)
    if min_possession_split is not None:
        query = query.filter(Video.team_a_possession_pct.isnot(None))
        query = query.filter(func.abs(Video.team_a_possession_pct - 50.0) <= min_possession_split)

    total = query.count()

    column = _SORTABLE_COLUMNS[sort_by]
    query = query.order_by(column.desc() if sort_order == "desc" else column.asc())
    query = query.offset((page - 1) * page_size).limit(page_size)

    items = [VideoSummary.model_validate(v) for v in query.all()]
    return VideoListResponse(items=items, total=total, page=page, page_size=page_size)


@router.get("/videos/{video_id}", response_model=VideoDetail)
def get_video(video_id: str, db: Session = Depends(get_db)):
    video = db.get(Video, video_id)
    if video is None:
        raise HTTPException(status_code=404, detail="Video not found")
    return VideoDetail.model_validate(video)


@router.delete("/videos/{video_id}", status_code=204)
def delete_video(video_id: str, db: Session = Depends(get_db)):
    video = db.get(Video, video_id)
    if video is None:
        raise HTTPException(status_code=404, detail="Video not found")

    delete_video_files(video_id)
    db.delete(video)
    db.commit()
    return Response(status_code=204)


_RANGE_RE = re.compile(r"bytes=(\d*)-(\d*)")


@router.get("/videos/{video_id}/stream")
def stream_video(video_id: str, request: Request, db: Session = Depends(get_db)):
    """
    Serves the annotated output video with HTTP Range support so an HTML5
    <video> element can seek. Parses the Range header, seeks the file, and
    returns 206 Partial Content with Content-Range/Accept-Ranges -- the
    standard pattern browsers require to allow seeking on a <video> tag
    rather than only ever downloading from byte 0.
    """
    video = db.get(Video, video_id)
    if video is None:
        raise HTTPException(status_code=404, detail="Video not found")
    if not video.output_path or not os.path.isfile(video.output_path):
        raise HTTPException(status_code=404, detail="Output video not yet available")

    file_size = os.path.getsize(video.output_path)
    range_header = request.headers.get("range")

    if range_header is None:
        # No Range header: serve the whole file with 200, but still advertise
        # Accept-Ranges so the client knows seeking is supported for
        # subsequent requests.
        def full_file_iterator():
            with open(video.output_path, "rb") as f:
                while True:
                    chunk = f.read(_CHUNK_SIZE)
                    if not chunk:
                        break
                    yield chunk

        return StreamingResponse(
            full_file_iterator(),
            media_type="video/mp4",
            headers={"Accept-Ranges": "bytes", "Content-Length": str(file_size)},
        )

    match = _RANGE_RE.match(range_header)
    if not match:
        raise HTTPException(status_code=416, detail="Invalid Range header")

    start_str, end_str = match.groups()
    start = int(start_str) if start_str else 0
    end = int(end_str) if end_str else file_size - 1
    end = min(end, file_size - 1)

    if start > end or start >= file_size:
        raise HTTPException(status_code=416, detail="Requested range not satisfiable")

    content_length = end - start + 1

    def range_iterator():
        with open(video.output_path, "rb") as f:
            f.seek(start)
            remaining = content_length
            while remaining > 0:
                chunk = f.read(min(_CHUNK_SIZE, remaining))
                if not chunk:
                    break
                remaining -= len(chunk)
                yield chunk

    return StreamingResponse(
        range_iterator(),
        status_code=206,
        media_type="video/mp4",
        headers={
            "Content-Range": f"bytes {start}-{end}/{file_size}",
            "Accept-Ranges": "bytes",
            "Content-Length": str(content_length),
        },
    )
