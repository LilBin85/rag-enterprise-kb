"""不触发模型加载的存活与就绪检查。"""

from typing import Annotated

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from rag_kb.web.dependencies import ReadinessFailure, ReadinessProbe, build_readiness_probe

router = APIRouter(tags=["health"])


@router.get("/health/live")
def live() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/health/ready", response_model=None)
def ready(
    probe: Annotated[ReadinessProbe, Depends(build_readiness_probe)],
) -> dict[str, object] | JSONResponse:
    try:
        dependencies = probe.check()
    except ReadinessFailure as exc:
        return JSONResponse(
            status_code=503,
            content={"status": "unavailable", "dependencies": {exc.component: "unavailable"}},
        )
    except Exception:
        return JSONResponse(
            status_code=503,
            content={"status": "unavailable", "dependencies": {"system": "unknown"}},
        )
    return {"status": "ok", "dependencies": dependencies}
