"""
main.py - FastAPI application entrypoint.

Endpoints:
    POST /api/analyze       Upload 1-3 bowling videos (front/back/side),
                             get back a JSON biomechanics report + URLs to
                             stream the annotated skeleton-overlay videos.
    GET  /api/video/{id}     Stream (and then delete) a processed video.
    POST /api/chat           Context-aware Gemini chat proxy.
    GET  /api/health         Liveness check.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import tempfile
import time
import uuid
from pathlib import Path
from typing import Dict, Optional

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from biomechanics_analyzer import ALL_VIEWS, BiomechanicsAnalyzer
from gemini_service import GeminiService

logger = logging.getLogger("bowling_api")
logging.basicConfig(level=logging.INFO)


def _load_dotenv() -> None:
    env_path = Path(__file__).resolve().parent / ".env"
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv()

app = FastAPI(title="Bowling Biomechanics API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:5173").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
api_key=os.getenv("GEMINI_API_KEY")
analyzer = BiomechanicsAnalyzer()
gemini = GeminiService(api_key=os.getenv("GEMINI_API_KEY"))

# In-memory registry mapping a short-lived video_id -> (path, expires_at).
# A processed video is deleted either when it is streamed once, or when it
# expires, whichever comes first. For multi-instance deployments, replace
# this with Redis / a shared store.
_VIDEO_REGISTRY: Dict[str, Dict] = {}
_VIDEO_TTL_SECONDS = 15 * 60


def _register_video(path: Path) -> str:
    video_id = uuid.uuid4().hex
    _VIDEO_REGISTRY[video_id] = {"path": path, "expires_at": time.monotonic() + _VIDEO_TTL_SECONDS}
    return video_id


def _cleanup_paths(*paths: Optional[Path]) -> None:
    """BackgroundTasks target: remove temp files/dirs after the response is sent."""
    for p in paths:
        if not p:
            continue
        try:
            if p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
            elif p.exists():
                p.unlink()
        except OSError:
            logger.warning("Failed to clean up temp path: %s", p)


def _cleanup_expired_videos() -> None:
    now = time.monotonic()
    expired = [vid for vid, meta in _VIDEO_REGISTRY.items() if meta["expires_at"] < now]
    for vid in expired:
        meta = _VIDEO_REGISTRY.pop(vid, None)
        if meta:
            _cleanup_paths(meta["path"])


class ChatMessage(BaseModel):
    role: str  # "user" | "model"
    content: str


class ChatRequest(BaseModel):
    message: str
    history: list[ChatMessage] = []
    analysis_report: Optional[dict] = None  # current JSON report, sent when the
    # frontend detects the question is about the analysis


@app.get("/api/health")
async def health():
    return {"status": "ok"}


@app.post("/api/analyze")
async def analyze_bowling(
    background_tasks: BackgroundTasks,
    front: Optional[UploadFile] = File(None),
    back: Optional[UploadFile] = File(None),
    side: Optional[UploadFile] = File(None),
):
    _cleanup_expired_videos()

    uploads = {"front": front, "back": back, "side": side}
    provided = {view: f for view, f in uploads.items() if f is not None}

    if not provided:
        raise HTTPException(status_code=400, detail="Upload at least one video (front, back, or side).")
    if len(provided) > 3:
        raise HTTPException(status_code=400, detail="A maximum of 3 videos is supported.")

    for view, upload_file in provided.items():
        if upload_file.content_type and not upload_file.content_type.startswith("video/"):
            raise HTTPException(status_code=400, detail=f"'{view}' file must be a video.")

    # --- Save uploads to real temp files (tempfile module) ---
    work_dir = Path(tempfile.mkdtemp(prefix="bowling_in_"))
    output_dir = Path(tempfile.mkdtemp(prefix="bowling_out_"))
    input_paths: Dict[str, Path] = {}

    try:
        for view, upload_file in provided.items():
            suffix = Path(upload_file.filename or "").suffix or ".mp4"
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=suffix, dir=work_dir
            ) as tmp:
                content = await upload_file.read()
                tmp.write(content)
                input_paths[view] = Path(tmp.name)
            await upload_file.close()

        # Pose estimation over several videos is CPU-bound; keep the event
        # loop responsive by running it in a worker thread.
        report, output_video_paths = await asyncio.to_thread(
            analyzer.analyze, input_paths, output_dir
        )

    except (IOError, ValueError) as exc:
        _cleanup_paths(work_dir, output_dir)
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception:
        _cleanup_paths(work_dir, output_dir)
        logger.exception("Unexpected failure during analysis")
        raise HTTPException(status_code=500, detail="Analysis failed unexpectedly.")

    # Input videos are no longer needed once analysis has run: delete them
    # right after this response is sent back to the client.
    background_tasks.add_task(_cleanup_paths, work_dir)

    video_urls = {}
    for view, path in output_video_paths.items():
        video_id = _register_video(path)
        video_urls[view] = f"/api/video/{video_id}"

    report["video_urls"] = video_urls
    return JSONResponse(content=report)


@app.get("/api/video/{video_id}")
async def get_processed_video(video_id: str, background_tasks: BackgroundTasks):
    meta = _VIDEO_REGISTRY.get(video_id)
    if not meta or not Path(meta["path"]).exists():
        raise HTTPException(status_code=404, detail="Video not found or has expired.")

    path = Path(meta["path"])

    # Delete the file (and its now-empty output directory) and forget the
    # registry entry only after the video has finished streaming to the client.
    def _cleanup_after_stream():
        _cleanup_paths(path)
        _VIDEO_REGISTRY.pop(video_id, None)
        parent = path.parent
        if parent.exists() and not any(parent.iterdir()):
            _cleanup_paths(parent)

    background_tasks.add_task(_cleanup_after_stream)
    return FileResponse(
        path=path,
        media_type="video/mp4",
        filename=path.name,
        background=background_tasks,
    )


@app.post("/api/chat")
async def chat(request: ChatRequest):
    """
    Context-aware Gemini proxy. If `analysis_report` is present the request
    is assumed to be analysis-related and the report JSON is injected as
    grounding context; otherwise Gemini answers as a general cricket
    fitness/recovery assistant.
    """
    try:
        reply = await gemini.generate_reply(
            message=request.message,
            history=[m.model_dump() for m in request.history],
            analysis_report=request.analysis_report,
        )
    except Exception:
        logger.exception("Gemini call failed")
        raise HTTPException(status_code=502, detail="Chat assistant is temporarily unavailable.")
    return {"reply": reply}


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    logger.info("Starting backend server on %s:%s", host, port)
    uvicorn.run(app, host=host, port=port, reload=False)


