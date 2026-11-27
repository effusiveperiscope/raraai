#!/usr/bin/env python3
"""HTTP server for the stage-2 web client.

    python web_client/server.py                     # default config/port
    python web_client/server.py --port 8765 --preload
    python web_client/server.py --device cpu --no-preload

What it serves
--------------
* ``/``                  the single-page client (``web_client/static``)
* ``/api/ping``          unauthenticated liveness/auth probe
* ``/api/options``       models, variants, defaults, capabilities
* ``/api/speakers``      speaker ids + labels from the run's val filelist
* ``/api/preview``       phonemize a sentence (no model required)
* ``/api/generate``      queue a text -> audio generation
* ``/api/jobs/{id}``     poll one job (queued / running / done / error)
* ``/api/jobs/{id}/audio.wav``, ``/api/jobs/{id}/codes.pt``
* ``/api/health``        device, GPU memory, loaded models, queue state

Concurrency model
-----------------
A single worker thread owns the GPU: jobs run one at a time, in submission
order.  That is the safe default for one user and remains correct if the same
server is later exposed to several users (each browser session has its own
job namespace/history; nothing in the engine assumes one session).  For a
multi-user deployment set ``server.fixed_model`` (hides the model picker),
``server.max_queue`` and an ``api_key`` in ``web_client/models.yaml``.
"""
from __future__ import annotations

import argparse
import logging
import os
import queue
import sys
import threading
import time
import traceback
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

logging.basicConfig(level=logging.WARNING)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from fastapi import FastAPI, HTTPException, Request  # noqa: E402
from fastapi.responses import (FileResponse, JSONResponse,  # noqa: E402
                               Response)
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel, ConfigDict, Field  # noqa: E402

from web_client.engine import (  # noqa: E402
    EngineError,
    Render,
    RequestError,
    StudioEngine,
    load_registry,
)

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")


# ---------------------------------------------------------------------- #
# Request bodies
# ---------------------------------------------------------------------- #
class GenerateBody(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    text: str
    model: Optional[str] = None
    variant: Optional[str] = None
    sid: Optional[int] = None
    frames: Optional[int] = None
    temperature: Optional[float] = None
    top_k: Optional[int] = None
    seed: Optional[int] = None
    intensity: Optional[int] = None
    register_level: Optional[int] = Field(default=None, alias="register")
    pitch_shift: Optional[float] = None
    slot_shift: Optional[int] = None
    word_semantics: Optional[str] = None
    base: Optional[str] = None


    source_job: str
    model: Optional[str] = None
    variant: Optional[str] = None
    start: Optional[int] = None
    frames: Optional[int] = None
    temperature: Optional[float] = None
    top_k: Optional[int] = None
    seed: Optional[int] = None
    best_of: Optional[int] = None
    intensity: Optional[int] = None
    register_level: Optional[int] = Field(default=None, alias="register")
    pitch_shift: Optional[float] = None
    slot_shift: Optional[int] = None
    word_semantics: Optional[str] = None
    base: Optional[str] = None


class PreviewBody(BaseModel):
    text: str


# ---------------------------------------------------------------------- #
# Jobs
# ---------------------------------------------------------------------- #
@dataclass
class Job:
    id: str
    session: str
    kind: str
    model_id: str
    variant_id: str
    request: Dict[str, Any]
    source: Optional[Render] = None
    status: str = "queued"
    stage: str = "queued"
    created: float = field(default_factory=time.time)
    started: Optional[float] = None
    finished: Optional[float] = None
    error: Optional[str] = None
    http_status: int = 500
    render: Optional[Render] = None
    wav: Optional[bytes] = None
    sid_label: str = ""

    def elapsed(self) -> float:
        if self.started is None:
            return 0.0
        end = self.finished if self.finished is not None else time.time()
        return end - self.started


class JobStore:
    """Bounded, session-scoped job history.

    Jobs are the unit of both polling and history: the browser keeps only ids,
    and infill refers to a source job id, so a page reload (same session id)
    still works and no code tensors ever travel over the wire.
    """

    def __init__(self, max_jobs: int = 200):
        self.max_jobs = max(1, int(max_jobs))
        self._jobs: "OrderedDict[str, Job]" = OrderedDict()
        self._lock = threading.Lock()

    def add(self, job: Job) -> None:
        with self._lock:
            self._jobs[job.id] = job
            while len(self._jobs) > self.max_jobs:
                # Prefer a finished job; never discard the one being added, and
                # only fall back to the oldest entry if everything is in flight.
                victim = next((k for k, j in self._jobs.items()
                               if k != job.id
                               and j.status in ("done", "error")), None)
                if victim is None:
                    victim = next(iter(self._jobs))
                    if victim == job.id:
                        break
                self._jobs.pop(victim, None)

    def remove(self, job_id: str) -> None:
        with self._lock:
            self._jobs.pop(str(job_id), None)

    def get(self, job_id: str, session: str,
            allow_any: bool = False) -> Optional[Job]:
        with self._lock:
            job = self._jobs.get(str(job_id))
        if job is None:
            return None
        if not allow_any and job.session != session:
            return None
        return job

    def queued_before(self, job: Job) -> int:
        with self._lock:
            return sum(1 for j in self._jobs.values()
                       if j.status == "queued" and j.created < job.created)

    def count(self) -> Dict[str, int]:
        with self._lock:
            out: Dict[str, int] = {}
            for job in self._jobs.values():
                out[job.status] = out.get(job.status, 0) + 1
            return out

    def clear_session(self, session: str) -> int:
        with self._lock:
            ids = [k for k, j in self._jobs.items() if j.session == session]
            for key in ids:
                self._jobs.pop(key, None)
            return len(ids)


# ---------------------------------------------------------------------- #
# Runtime
# ---------------------------------------------------------------------- #
class StudioRuntime:
    def __init__(self, registry: Dict[str, Any], server_cfg: Dict[str, Any]):
        self.server_cfg = dict(server_cfg or {})
        self.engine = StudioEngine(registry, server_cfg)
        self.jobs = JobStore(int(self.server_cfg.get("max_jobs", 200)))
        self.defaults = dict(registry.get("defaults") or {})
        # Normalise the defaults against the (possibly pinned) registry so a
        # stale defaults block cannot make every request fail.
        fixed = self.server_cfg.get("fixed_model")
        if fixed:
            self.defaults["model"] = fixed
            self.defaults["variant"] = self.server_cfg.get("fixed_variant")
        try:
            model, variant = self.engine.variant(self.defaults.get("model"),
                                                 self.defaults.get("variant"))
            self.defaults["model"] = model.id
            self.defaults["variant"] = variant.id
            # A defaults block written for another model may name a speaker id
            # this run never used; fall back to its most frequent val voice.
            speakers = self.engine.speakers(model.id, variant.id)
            if speakers:
                valid = {int(s["id"]) for s in speakers}
                if int(self.defaults.get("sid") or -1) not in valid:
                    self.defaults["sid"] = int(speakers[0]["id"])
        except RequestError:
            pass
        self.queue: "queue.Queue[Optional[Job]]" = queue.Queue(
            maxsize=max(1, int(self.server_cfg.get("max_queue", 8))))
        self.started = time.time()
        self._worker = threading.Thread(target=self._run, daemon=True,
                                        name="generation-worker")
        self._worker.start()
        self._preload_thread = None
        if bool(self.server_cfg.get("preload", False)):
            self._preload_thread = threading.Thread(
                target=self._preload, daemon=True, name="model-preload")
            self._preload_thread.start()

    # -- queue ---------------------------------------------------------- #
    def submit(self, job: Job) -> None:
        self.jobs.add(job)
        try:
            self.queue.put_nowait(job)
        except queue.Full:
            self.jobs.remove(job.id)
            raise HTTPException(
                429, "the server's generation queue is full; try again in a "
                     "moment")

    def _run(self) -> None:
        while True:
            job = self.queue.get()
            if job is None:
                return
            try:
                self._execute(job)
            except Exception as exc:  # never kill the worker
                traceback.print_exc()
                job.status = "error"
                job.error = f"{type(exc).__name__}: {exc}"
                job.http_status = 500
                job.stage = "error"
                job.finished = time.time()
            finally:
                self.queue.task_done()

    def _execute(self, job: Job) -> None:
        job.status = "running"
        job.started = time.time()
        job.stage = "starting"

        def stage(message: str) -> None:
            job.stage = str(message)
            print(f"[job {job.id[:8]}] {message}")

        try:
            if job.kind == "generate":
                render = self.engine.generate(
                    job.model_id, job.variant_id, job.request, stage=stage)
            else:
                raise EngineError(f"unknown job kind {job.kind!r}")

            render.job_id = job.id
            stage("packaging audio")
            job.wav = render.wav_bytes()
            job.render = render
            job.sid_label = self._speaker_label(render)
            job.status = "done"
            job.stage = "done"
        except (RequestError, EngineError) as exc:
            job.status = "error"
            job.error = str(exc)
            job.http_status = getattr(exc, "status_code", 500)
            job.stage = "error"
            print(f"[job {job.id[:8]}] failed: {exc}")
        except SystemExit as exc:          # layout guard in repo helpers
            job.status = "error"
            job.error = str(exc)
            job.http_status = 500
            job.stage = "error"
            print(f"[job {job.id[:8]}] failed: {exc}")
        except Exception as exc:
            traceback.print_exc()
            job.status = "error"
            job.error = f"{type(exc).__name__}: {exc}"
            job.http_status = 500
            job.stage = "error"
        finally:
            job.finished = time.time()

    def _speaker_label(self, render: Render) -> str:
        try:
            for speaker in self.engine.speakers(render.model_id,
                                                render.variant_id):
                if int(speaker["id"]) == int(render.sid):
                    return str(speaker["label"])
        except Exception:
            pass
        return f"Speaker {render.sid}"

    def _preload(self) -> None:
        model = self.defaults.get("model")
        variant = self.defaults.get("variant")
        try:
            print(f"web_client: preloading {model}/{variant} ...")
            self.engine.ensure_loaded(model, variant)
            print("web_client: preload done")
        except Exception as exc:
            print(f"web_client: preload failed (the first job will retry): "
                  f"{exc}")

    # -- serialization -------------------------------------------------- #
    def job_dict(self, job: Job) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "job_id": job.id,
            "kind": job.kind,
            "status": job.status,
            "stage": job.stage,
            "model": job.model_id,
            "variant": job.variant_id,
            "created": job.created,
            "started": job.started,
            "finished": job.finished,
            "elapsed": round(job.elapsed(), 3),
            "error": job.error,
            "queue_position": (self.jobs.queued_before(job)
                               if job.status == "queued" else 0),
        }
        if job.status == "done" and job.render is not None:
            payload["result"] = self.render_dict(job)
        return payload

    def render_dict(self, job: Job) -> Dict[str, Any]:
        render = job.render
        assert render is not None
        base = f"/api/jobs/{job.id}"
        return {
            "job_id": job.id,
            "kind": render.kind,
            "model": render.model_id,
            "variant": render.variant_id,
            "text": render.text,
            "sid": render.sid,
            "sid_label": job.sid_label,
            "phones": render.phones,
            "n_phone_ids": len(render.phone_ids),
            "frames": render.frames,
            "frame_rate": render.frame_rate,
            "duration": render.duration,
            "sampling_rate": render.sampling_rate,
            "audio_seconds": (None if render.audio is None
                              else render.audio.size / render.sampling_rate),
            "temperature": render.temperature,
            "top_k": render.top_k,
            "seed": render.seed,
            "hints_state": render.hints_state,
            "hints": render.hints_params,
            "eos_length": render.eos_length,
            "unique_codes": render.unique_codes,
            "repeat_frac": render.repeat_frac,
            "f0": render.f0_summary(),
            "timings": {k: round(v, 1) for k, v in render.timings.items()},
            "notes": render.notes,
            "source_job": render.source_job,
            "span": render.span,
            "seam_ce": render.seam_ce,
            "best_of": render.best_of,
            "audio_url": (base + "/audio.wav"
                          if job.wav is not None else None),
            "codes_url": base + "/codes.pt",
            "audio_available": job.wav is not None,
        }

    def models_dict(self) -> Dict[str, Any]:
        models = []
        fixed_variant = self.server_cfg.get("fixed_variant")
        pinned = (self.server_cfg.get("fixed_model")
                  and fixed_variant) or None
        for model in self.engine.visible_models():
            variants = []
            for variant in model.variants:
                if (pinned and model.id == self.server_cfg.get("fixed_model")
                        and variant.id != fixed_variant):
                    continue
                try:
                    caps = self.engine.probe_caps(model.id, variant.id)
                except Exception as exc:
                    caps = {"loaded": False, "error": str(exc),
                            "model": model.id, "variant": variant.id}
                variants.append({
                    "id": variant.id,
                    "label": variant.label,
                    "notes": variant.notes,
                    "caps": caps,
                })
            models.append({
                "id": model.id,
                "label": model.label,
                "notes": model.notes,
                "variants": variants,
            })
        defaults = dict(self.defaults)
        # A fixed-model deployment must not advertise a hidden default.
        visible_ids = {m["id"] for m in models}
        if defaults.get("model") not in visible_ids and models:
            defaults["model"] = models[0]["id"]
            defaults["variant"] = models[0]["variants"][0]["id"]
        return {"defaults": defaults, "models": models}


# ---------------------------------------------------------------------- #
# App
# ---------------------------------------------------------------------- #
def create_app(config_path: Optional[str] = None,
               registry: Optional[Dict[str, Any]] = None) -> FastAPI:
    registry = registry or load_registry(str(config_path))
    server_cfg = registry["server"]
    runtime = StudioRuntime(registry, server_cfg)

    app = FastAPI(title=str(server_cfg.get("title", "Stage-2 Studio")),
                  docs_url="/api/docs", openapi_url="/api/openapi.json")
    app.state.runtime = runtime

    api_key = server_cfg.get("api_key")

    @app.middleware("http")
    async def _auth(request: Request, call_next):
        path = request.url.path
        if (api_key and path.startswith("/api")
                and path not in ("/api/ping",)):
            supplied = (request.headers.get("X-API-Key")
                        or request.query_params.get("key"))
            if supplied != str(api_key):
                return JSONResponse({"detail": "invalid or missing API key"},
                                    status_code=401)
        response = await call_next(request)
        response.headers.setdefault("X-Studio-Server", "stage2-web")
        # Keep a cookie copy of the browser's session id so media elements
        # (<audio src>, download links), which cannot send X-Studio-Session,
        # resolve to the same job namespace.
        session = (request.headers.get("X-Studio-Session")
                   or request.cookies.get("studio_session"))
        if session and not request.cookies.get("studio_session"):
            response.set_cookie("studio_session", str(session)[:128],
                                max_age=30 * 24 * 3600, samesite="lax",
                                httponly=True)
        return response

    @app.exception_handler(RequestError)
    async def _bad_request(_request, exc):
        return JSONResponse({"detail": str(exc)}, status_code=400)

    @app.exception_handler(EngineError)
    async def _engine_error(_request, exc):
        return JSONResponse({"detail": str(exc)}, status_code=500)

    def session_of(request: Request) -> str:
        sid = (request.headers.get("X-Studio-Session")
               or request.query_params.get("session")
               or request.cookies.get("studio_session")
               or uuid.uuid4().hex)
        return str(sid)[:128]

    # -- static --------------------------------------------------------- #
    if not os.path.isdir(STATIC_DIR):
        raise SystemExit(f"static directory not found: {STATIC_DIR}")
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    async def index():
        return FileResponse(os.path.join(STATIC_DIR, "index.html"))

    @app.get("/favicon.svg", include_in_schema=False)
    async def favicon():
        return FileResponse(os.path.join(STATIC_DIR, "favicon.svg"))

    # -- api ------------------------------------------------------------ #
    @app.get("/api/ping")
    async def ping():
        return {
            "ok": True,
            "title": str(server_cfg.get("title", "Stage-2 Studio")),
            "auth_required": bool(api_key),
            "single_client": bool(server_cfg.get("single_client", True)),
        }

    @app.get("/api/options")
    async def options(request: Request):
        payload = runtime.models_dict()
        payload.update({
            "app": {
                "title": str(server_cfg.get("title", "Stage-2 Studio")),
                "subtitle": str(server_cfg.get("subtitle", "")),
                "single_client": bool(server_cfg.get("single_client", True)),
                "fixed_model": server_cfg.get("fixed_model"),
                "fixed_variant": server_cfg.get("fixed_variant"),
                "auth_required": bool(api_key),
            },
            "limits": {
                "max_text_chars": int(server_cfg.get("max_text_chars", 2000)),
                "max_queue": int(server_cfg.get("max_queue", 8)),
            },
            "session": session_of(request),
            "device": runtime.engine.device,
            "started": runtime.started,
        })
        return payload

    @app.get("/api/speakers")
    async def speakers(model: Optional[str] = None,
                       variant: Optional[str] = None):
        return {"speakers": runtime.engine.speakers(model, variant)}

    @app.post("/api/preview")
    async def preview(body: PreviewBody):
        return runtime.engine.preview(body.text)

    @app.post("/api/generate")
    async def generate(body: GenerateBody, request: Request):
        session = session_of(request)
        params = dict(runtime.defaults)
        params.update(body.model_dump(exclude_none=True, by_alias=True))
        text = str(params.pop("text", ""))
        model_id = params.get("model")
        variant_id = params.get("variant")
        model, variant = runtime.engine.variant(model_id, variant_id)
        job = Job(
            id=uuid.uuid4().hex[:16],
            session=session,
            kind="generate",
            model_id=model.id,
            variant_id=variant.id,
            request={**params, "text": text},
        )
        runtime.submit(job)
        return {"job_id": job.id, "status": job.status}

    @app.get("/api/jobs/{job_id}")
    async def job_status(job_id: str, request: Request):
        job = runtime.jobs.get(job_id, session_of(request))
        if job is None:
            raise HTTPException(404, "job not found (or expired)")
        return runtime.job_dict(job)

    @app.get("/api/jobs/{job_id}/audio.wav")
    async def job_audio(job_id: str, request: Request, download: int = 0):
        job = runtime.jobs.get(job_id, session_of(request))
        if job is None or job.wav is None:
            raise HTTPException(404, "audio not found (or expired)")
        label = job.sid_label or str(job.render.sid if job.render else "voice")
        safe = "".join(ch if (ch.isalnum() or ch in " ._-") else "_"
                       for ch in label).strip().replace(" ", "_") or "voice"
        name = f"stage2_{job.model_id}_{safe}_{job.id[:8]}.wav"
        headers = {"Content-Disposition":
                   f'{"attachment" if download else "inline"}; '
                   f'filename="{name}"'}
        return Response(content=job.wav, media_type="audio/wav",
                        headers=headers)

    @app.get("/api/jobs/{job_id}/codes.pt")
    async def job_codes(job_id: str, request: Request):
        job = runtime.jobs.get(job_id, session_of(request))
        if job is None or job.render is None:
            raise HTTPException(404, "codes not found (or expired)")
        return Response(
            content=job.render.codes_bytes(),
            media_type="application/octet-stream",
            headers={"Content-Disposition":
                     f'attachment; filename="codes_{job.id[:8]}.pt"'})

    @app.get("/api/health")
    async def health():
        status = runtime.engine.status()
        status.update({
            "uptime_seconds": round(time.time() - runtime.started, 1),
            "queue_size": runtime.queue.qsize(),
            "jobs": runtime.jobs.count(),
            "defaults": runtime.defaults,
        })
        return status

    @app.delete("/api/history")
    async def clear_history(request: Request):
        removed = runtime.jobs.clear_session(session_of(request))
        return {"removed": removed}

    return app


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Stage-2 model server + mobile web client.")
    parser.add_argument("--config", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "models.yaml"),
        help="model registry YAML (default: web_client/models.yaml)")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--device", default=None, help="cuda / cpu / cuda:1")
    parser.add_argument("--precision", default=None,
                        choices=("fp32", "bf16", "fp16"))
    parser.add_argument("--preload", dest="preload", action="store_true",
                        default=None, help="load the default model at startup")
    parser.add_argument("--no-preload", dest="preload", action="store_false")
    parser.add_argument("--model", default=None,
                        help="pin this model id (overrides server.fixed_model)")
    parser.add_argument("--variant", default=None,
                        help="pin this variant id (overrides server.fixed_variant)")
    parser.add_argument("--api-key", default=None,
                        help="require this key on /api (overrides server.api_key)")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    import uvicorn

    registry = load_registry(args.config)
    cfg = dict(registry["server"])
    host = args.host or str(cfg.get("host", "0.0.0.0"))
    port = int(args.port if args.port is not None else cfg.get("port", 8765))
    if args.device:
        cfg["device"] = args.device
    if args.precision:
        cfg["precision"] = args.precision
    if args.preload is not None:
        cfg["preload"] = bool(args.preload)
    if args.model:
        cfg["fixed_model"] = args.model
    if args.variant:
        cfg["fixed_variant"] = args.variant
    if args.api_key is not None:
        cfg["api_key"] = args.api_key or None

    # Write the effective settings back so the app sees the CLI overrides.
    registry["server"] = cfg
    app = create_app(registry=registry)

    print(f"Stage-2 web client on http://{host}:{port}  "
          f"(device {cfg.get('device') or 'auto'}, "
          f"precision {cfg.get('precision', 'fp32')})")
    uvicorn.run(app, host=host, port=port, log_level="warning",
                reload=False)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (KeyboardInterrupt, BrokenPipeError):
        sys.exit(0)
