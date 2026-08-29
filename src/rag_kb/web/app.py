"""FastAPI 应用工厂。"""

from contextlib import asynccontextmanager
from time import perf_counter

from fastapi import FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from rag_kb.config import Settings, get_settings
from rag_kb.observability import (
    Observability,
    bind_request_id,
    clear_request_id,
    configure_json_logging,
    extract_trace_context,
    log_event,
    normalize_request_id,
)
from rag_kb.web.routers.health import router as health_router
from rag_kb.web.routers.knowledge import router as knowledge_router
from rag_kb.web.routers.documents import router as documents_router
from rag_kb.web.routers.tasks import router as tasks_router
from rag_kb.web.routers.approvals import router as approvals_router


def _route_template(request: Request) -> str:
    route = request.scope.get("route")
    template = getattr(route, "path_format", None) or getattr(route, "path", None)
    return template if isinstance(template, str) and template.startswith("/") else "/unknown"


def _status_class(status_code: int) -> str:
    return f"{status_code // 100}xx" if 100 <= status_code <= 599 else "unknown"


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    observability = Observability(
        enabled=settings.observability_enabled,
        metrics_enabled=settings.metrics_enabled,
        service_name=settings.otel_service_name,
        otlp_endpoint=settings.otel_exporter_otlp_endpoint,
        trace_sample_ratio=settings.otel_trace_sample_ratio,
    )
    configure_json_logging(service=settings.otel_service_name, level=settings.log_level)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        observability.start()
        try:
            yield
        finally:
            observability.shutdown()

    app = FastAPI(title="企业知识任务智能体", version="1.0.0", lifespan=lifespan)
    app.state.observability = observability

    @app.middleware("http")
    async def observe_http(request: Request, call_next) -> Response:
        request_id = normalize_request_id(request.headers.get("X-Request-ID"))
        request.state.request_id = request_id
        request_token = bind_request_id(request_id)
        response: Response | None = None
        started_at = perf_counter()
        try:
            parent_context = extract_trace_context(
                request.headers.get("traceparent"), request.headers.get("tracestate")
            )
            with observability.http_span(parent_context) as span:
                try:
                    response = await call_next(request)
                except Exception:
                    # 不能把异常文本、请求内容或 header 写入响应、Trace 或日志。
                    response = JSONResponse(status_code=500, content={"detail": "服务器内部错误"})
                finally:
                    if response is not None:
                        route = _route_template(request)
                        method = request.method
                        duration_seconds = perf_counter() - started_at
                        status_code = response.status_code
                        observability.finish_http_span(
                            span,
                            route=route,
                            method=method,
                            status_code=status_code,
                        )
                        observability.record_http(
                            route=route,
                            method=method,
                            status_class=_status_class(status_code),
                            duration_seconds=duration_seconds,
                        )
                        log_event(
                            "http.request.completed",
                            component="web",
                            operation="http.request",
                            outcome=_status_class(status_code),
                            duration_ms=duration_seconds * 1_000,
                        )
                        response.headers["X-Request-ID"] = request_id
            assert response is not None
            return response
        finally:
            clear_request_id(request_token)

    @app.exception_handler(RequestValidationError)
    async def validation_error(
        _request: Request, _exc: RequestValidationError
    ) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": "请求参数非法"})

    if settings.metrics_enabled:

        @app.get("/metrics", include_in_schema=False)
        def metrics() -> Response:
            return Response(
                content=observability.metrics_payload(),
                media_type=observability.metrics_content_type,
            )

    app.include_router(health_router)
    app.include_router(knowledge_router)
    app.include_router(documents_router)
    app.include_router(tasks_router)
    app.include_router(approvals_router)

    return app
