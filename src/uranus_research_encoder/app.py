"""HTTP composition with injectable backend and failure-tolerant liveness."""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException
from starlette.responses import JSONResponse, Response

from .auth import Ingress
from .chunking import chunk_sections
from .config import Settings
from .contracts import (
    MAX_RESPONSE_BYTES,
    ChunkRequest,
    ChunkResponse,
    ChunkResult,
    EmbedMetrics,
    EmbedRequest,
    EmbedResponse,
)
from .errors import error, event
from .model import Backend, TorchBackend, runtime_name
from .onnx_backend import OnnxBackend
from .version import (
    CONTRACT_VERSION,
    DIMENSIONS,
    EMBEDDING_VERSION,
    MODEL,
    MODEL_REVISION,
    metadata,
)


def create_app(settings: Settings | None = None, backend: Backend | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            app.state.settings = settings or Settings.from_env()
        except Exception:
            event("configuration_invalid")
        if app.state.settings is not None:
            try:
                current_settings = app.state.settings
                app.state.backend = backend or (
                    OnnxBackend(
                        current_settings.model_root,
                        intra_op_threads=current_settings.onnx_intra_op_threads,
                        inter_op_threads=current_settings.onnx_inter_op_threads,
                    )
                    if current_settings.backend == "onnx"
                    else TorchBackend(current_settings.model_root)
                )
                await run_in_threadpool(app.state.backend.load)
                event("model_loaded")
            except Exception:
                app.state.backend = None
                event("model_load_failed")
        yield
        app.state.backend = None

    # Development documentation is an explicit opt-in and remains authenticated.
    docs = (
        settings.enable_docs
        if settings is not None
        else os.environ.get("ENCODER_ENABLE_DOCS") == "1"
    )
    app = FastAPI(
        lifespan=lifespan,
        redirect_slashes=False,
        docs_url="/docs" if docs else None,
        redoc_url=None,
        openapi_url="/openapi.json" if docs else None,
    )
    app.state.settings = None
    app.state.backend = None
    app.add_middleware(Ingress, state=app.state)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request, exc):
        return error(422, "invalid_request")

    @app.exception_handler(HTTPException)
    async def http_error(request, exc):
        return error(exc.status_code, "not_found" if exc.status_code == 404 else "invalid_request")

    @app.exception_handler(Exception)
    async def internal_error(request, exc):
        event("request_failed")
        return error(500, "internal_error")

    def ready() -> bool:
        current = app.state.backend
        return bool(
            app.state.settings
            and current
            and current.loaded
            and current.tokenizer_available
            and current.revision == MODEL_REVISION
        )

    def bounded_response(result) -> Response:
        serialized = result.model_dump_json().encode("utf-8")
        if len(serialized) > MAX_RESPONSE_BYTES:
            return error(422, "response_limit")
        return Response(serialized, media_type="application/json")

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/ready")
    async def readiness():
        if not ready():
            return error(503, "not_ready")
        return {
            "status": "ready",
            "model": MODEL,
            "dimensions": DIMENSIONS,
            "embedding_version": EMBEDDING_VERSION,
            "contract_version": CONTRACT_VERSION,
            "model_revision": MODEL_REVISION,
            "backend": app.state.backend.backend,
            "runtime": app.state.backend.runtime,
        }

    @app.get("/version")
    async def version():
        current = app.state.backend
        name = current.backend if current else app.state.settings.backend
        try:
            runtime = current.runtime if current else runtime_name(name)
        except Exception:
            runtime = "unavailable"
        return JSONResponse(metadata() | {"backend": name, "runtime": runtime})

    @app.post("/chunks", response_model=ChunkResponse)
    async def chunks(request: ChunkRequest):
        if not ready():
            return error(503, "not_ready")

        def process():
            try:
                result = ChunkResponse(
                    embedding_version=EMBEDDING_VERSION,
                    documents=[
                        ChunkResult(
                            entity_id=d.entity_id,
                            chunks=chunk_sections(
                                d.sections,
                                app.state.backend.count,
                            ),
                        )
                        for d in request.documents
                    ],
                )
            except ValueError:
                return error(422, "invalid_request")
            return bounded_response(result)

        return await run_in_threadpool(process)

    @app.post("/embed", response_model=EmbedResponse)
    async def embed(request: EmbedRequest):
        if not ready():
            return error(503, "not_ready")

        def process():
            current = app.state.backend
            counts = [current.count(text) for text in request.texts]
            if any(n < 1 or n > current.max_tokens for n in counts):
                return error(422, "invalid_request")
            vectors = current.embed(request.texts, request.kind)
            result = EmbedResponse(
                embedding_version=EMBEDDING_VERSION,
                vectors=vectors,
                metrics=EmbedMetrics(text_count=len(request.texts), token_count=sum(counts)),
            )
            return bounded_response(result)

        return await run_in_threadpool(process)

    return app


app = create_app()
