"""同步本机 Compose 演示令牌对应的 PostgreSQL 主体。"""

from __future__ import annotations

import logging
from collections.abc import Mapping

from sqlalchemy import select

from rag_kb.config import get_settings
from rag_kb.infrastructure.auth.providers import demo_token_principals
from rag_kb.infrastructure.database import build_database
from rag_kb.infrastructure.database.models import PrincipalRow, TenantRow
from rag_kb.infrastructure.database.session import Database


logger = logging.getLogger(__name__)


def sync_demo_principals(
    database: Database,
    role_tokens: Mapping[str, str],
    *,
    tenant_id: str = "local-demo",
) -> tuple[str, ...]:
    """创建或更新当前演示令牌实际能解析到的本地主体。"""
    principals = tuple(demo_token_principals(role_tokens, tenant_id=tenant_id).values())
    if not principals:
        raise ValueError("demo_role_tokens_required")
    with database.sessions.begin() as session:
        tenant = session.get(TenantRow, tenant_id)
        if tenant is None:
            session.add(TenantRow(tenant_id=tenant_id, name="本地演示"))
            session.flush()
        else:
            tenant.active = True
        for principal in principals:
            row = session.scalar(
                select(PrincipalRow)
                .where(
                    PrincipalRow.tenant_id == principal.tenant_id,
                    PrincipalRow.user_id == principal.user_id,
                )
                .with_for_update()
            )
            if row is None:
                session.add(
                    PrincipalRow(
                        tenant_id=principal.tenant_id,
                        user_id=principal.user_id,
                        roles=sorted(principal.roles),
                        attributes=principal.attributes,
                    )
                )
            else:
                row.roles = sorted(principal.roles)
                row.attributes = principal.attributes
                row.active = True
    return tuple(principal.user_id for principal in principals)


def run() -> int:
    settings = get_settings()
    synced = sync_demo_principals(
        build_database(settings.database_url), settings.demo_role_tokens
    )
    logger.info("本地演示身份已同步 count=%s", len(synced))
    return 0


def main() -> int:
    logging.basicConfig(level=logging.INFO)
    try:
        return run()
    except ValueError as exc:
        logger.error("本地演示身份同步失败 error_code=%s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
