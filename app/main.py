"""Локальная страница: загрузка m4a и скачивание mp3."""

from __future__ import annotations

import threading
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from app.paths import ROOT
from app.pipeline import process

JOBS = ROOT / "data" / "jobs"
MAX_BYTES = 1024**3
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

    def launch(self, job_id: str, src: Path) -> None:
        self.jobs[job_id] = {
            "id": job_id,
            "status": "running",
            "stage": "Очередь",
            "percent": 0,
            "error": "",
        }
        thread = threading.Thread(target=self._run, args=(job_id, src), daemon=True)
        thread.start()

    def _run(self, job_id: str, src: Path) -> None:
        job = self.jobs[job_id]
        work = src.parent
        dst = work / "voice.mp3"

        def progress(stage: str, percent: float) -> None:
            job["stage"] = stage
            job["percent"] = int(percent)

        try:
            summary = process(src, dst, work, progress)
            job["summary"] = {
                "breaths": len(summary["breaths"]),
                "events": len(summary["events"]),
                "clicks": summary["clicks"],
                "duration": summary["output_duration"],
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


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.post("/jobs")
async def create_job(file: UploadFile = File(...)):
    filename = (file.filename or "").lower()
    if not filename.endswith(".m4a"):
        raise HTTPException(status_code=400, detail="Нужен файл в формате m4a")
    if not runner.reserve():
        raise HTTPException(status_code=409, detail="Уже идёт обработка другого файла")
    job_id = uuid.uuid4().hex
    folder = JOBS / job_id
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / "input.m4a"
    size = 0
    try:
        with target.open("wb") as out:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > MAX_BYTES:
                    raise HTTPException(status_code=413, detail="Файл больше 1 ГБ")
                out.write(chunk)
        if size == 0:
            raise HTTPException(status_code=400, detail="Файл пустой")
        runner.launch(job_id, target)
    except HTTPException:
        runner.release()
        raise
    except Exception:
        runner.release()
        raise
    return {"id": job_id}


@app.get("/jobs/{job_id}")
def job_status(job_id: str):
    job = runner.jobs.get(job_id)
    if job is None:
        report = JOBS / job_id / "report.json"
        mp3 = JOBS / job_id / "voice.mp3"
        if mp3.exists():
            return {"id": job_id, "status": "done", "stage": "Готово", "percent": 100, "error": ""}
        if not report.exists() and not (JOBS / job_id).exists():
            raise HTTPException(status_code=404, detail="Задача не найдена")
        return {"id": job_id, "status": "running", "stage": "Обработка", "percent": 0, "error": ""}
    return {
        "id": job_id,
        "status": job["status"],
        "stage": job["stage"],
        "percent": job["percent"],
        "error": job["error"],
        "summary": job.get("summary"),
    }


@app.get("/jobs/{job_id}/file")
def job_file(job_id: str):
    path = JOBS / job_id / "voice.mp3"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Файл ещё не готов")
    return FileResponse(
        path,
        media_type="audio/mpeg",
        filename="voice.mp3",
        content_disposition_type="inline",
    )
