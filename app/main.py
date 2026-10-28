"""Локальная страница: загрузка m4a или видео и скачивание результата."""

from __future__ import annotations

import threading
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from app.paths import ROOT
from app.pipeline import process
from app.stages.video import EXTENSIONS, process_video

JOBS = ROOT / "data" / "jobs"
MAX_AUDIO_BYTES = 1024**3
MAX_VIDEO_BYTES = 4 * 1024**3
STATIC = Path(__file__).resolve().parent / "static"

app = FastAPI(title="Очистка голоса")


class Runner:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._busy = False
        self.jobs: dict[str, dict] = {}

    def reserve(self) -> bool:
        with self._lock:
            if self._busy:
                return False
            self._busy = True
            return True

    def release(self) -> None:
        with self._lock:
            self._busy = False

    def launch(self, job_id: str, src: Path, kind: str) -> None:
        self.jobs[job_id] = {
            "id": job_id,
            "kind": kind,
            "status": "running",
            "stage": "Очередь",
            "percent": 0,
            "error": "",
        }
        (src.parent / "kind.txt").write_text(kind, encoding="utf-8")
        thread = threading.Thread(target=self._run, args=(job_id, src, kind), daemon=True)
        thread.start()

    def _run(self, job_id: str, src: Path, kind: str) -> None:
        job = self.jobs[job_id]
        work = src.parent
        dst = work / ("voice.mp4" if kind == "video" else "voice.mp3")

        def progress(stage: str, percent: float) -> None:
            job["stage"] = stage
            job["percent"] = int(percent)

        try:
            if kind == "video":
                summary = process_video(src, dst, work, progress)
            else:
                summary = process(src, dst, work, progress)
            job["summary"] = {
                "breaths": len(summary["breaths"]),
                "events": len(summary["events"]),
                "clicks": summary["clicks"],
                "duration": summary["output_duration"],
                "bytes": dst.stat().st_size,
            }
            job["status"] = "done"
            job["stage"] = "Готово"
            job["percent"] = 100
        except Exception as exc:
            job["status"] = "error"
            job["error"] = str(exc) or "Обработка прервалась"
        finally:
            self.release()


runner = Runner()


def _kind(job_id: str, job: dict | None = None) -> str:
    if job and job.get("kind"):
        return str(job["kind"])
    marker = JOBS / job_id / "kind.txt"
    if marker.exists():
        return marker.read_text(encoding="utf-8").strip() or "audio"
    if (JOBS / job_id / "voice.mp4").exists():
        return "video"
    return "audio"


def _result_path(job_id: str) -> Path | None:
    folder = JOBS / job_id
    if _kind(job_id) == "video":
        video = folder / "voice.mp4"
        return video if video.exists() else None
    audio = folder / "voice.mp3"
    return audio if audio.exists() else None


async def _save_upload(file: UploadFile, target: Path, limit: int, too_big: str) -> None:
    size = 0
    with target.open("wb") as out:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            if size > limit:
                raise HTTPException(status_code=413, detail=too_big)
            out.write(chunk)
    if size == 0:
        raise HTTPException(status_code=400, detail="Файл пустой")


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.post("/jobs")
async def create_job(file: UploadFile = File(...)):
    return await _create(file, "audio")


@app.post("/jobs/video")
async def create_video_job(file: UploadFile = File(...)):
    return await _create(file, "video")


async def _create(file: UploadFile, kind: str):
    filename = (file.filename or "").lower()
    if kind == "video":
        suffix = Path(filename).suffix
        if suffix not in EXTENSIONS:
            raise HTTPException(status_code=400, detail="Нужен видеофайл mp4, mov, mkv или webm")
        limit = MAX_VIDEO_BYTES
        too_big = "Файл больше 4 ГБ"
        stored = "input" + suffix
    else:
        if not filename.endswith(".m4a"):
            raise HTTPException(status_code=400, detail="Нужен файл в формате m4a")
        limit = MAX_AUDIO_BYTES
        too_big = "Файл больше 1 ГБ"
        stored = "input.m4a"
    if not runner.reserve():
        raise HTTPException(status_code=409, detail="Уже идёт обработка другого файла")
    job_id = uuid.uuid4().hex
    folder = JOBS / job_id
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / stored
    try:
        await _save_upload(file, target, limit, too_big)
        runner.launch(job_id, target, kind)
    except HTTPException:
        runner.release()
        raise
    except Exception:
        runner.release()
        raise
    return {"id": job_id, "kind": kind}


@app.get("/jobs/{job_id}")
def job_status(job_id: str):
    job = runner.jobs.get(job_id)
    kind = _kind(job_id, job)
    if job is None:
        folder = JOBS / job_id
        if _result_path(job_id) is not None:
            return {"id": job_id, "kind": kind, "status": "done", "stage": "Готово", "percent": 100, "error": ""}
        if not (folder / "report.json").exists() and not folder.exists():
            raise HTTPException(status_code=404, detail="Задача не найдена")
        return {"id": job_id, "kind": kind, "status": "running", "stage": "Обработка", "percent": 0, "error": ""}
    return {
        "id": job_id,
        "kind": kind,
        "status": job["status"],
        "stage": job["stage"],
        "percent": job["percent"],
        "error": job["error"],
        "summary": job.get("summary"),
    }


@app.get("/jobs/{job_id}/file")
def job_file(job_id: str):
    path = _result_path(job_id)
    if path is None:
        raise HTTPException(status_code=404, detail="Файл ещё не готов")
    if path.suffix == ".mp4":
        return FileResponse(
            path,
            media_type="video/mp4",
            filename="voice.mp4",
            content_disposition_type="inline",
        )
    return FileResponse(
        path,
        media_type="audio/mpeg",
        filename="voice.mp3",
        content_disposition_type="inline",
    )
