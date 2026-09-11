"""对现有数据执行一次只读企业检索诊断，不输出令牌、查询正文或证据正文。"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Sequence

from rag_kb.config import get_settings
from rag_kb.infrastructure.auth.providers import DemoTokenIdentityProvider
from rag_kb.infrastructure.database import build_database
from rag_kb.infrastructure.database.acl_repository import DatabaseAuthorizationPort
from rag_kb.infrastructure.retrieval.factory import build_enterprise_retriever
from rag_kb.observability import stable_error_code


class DiagnosticError(RuntimeError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", required=True, help="待诊断查询；不会写入诊断输出")
    parser.add_argument("--role", required=True, help="DEMO_ROLE_TOKENS 中的角色名")
    parser.add_argument("--document-id", help="可选的单文档硬范围")
    return parser


def run(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    settings = get_settings()
    token = settings.demo_role_tokens.get(args.role)
    if not token:
        raise DiagnosticError("diagnostic_demo_role_unavailable")
    auth = DemoTokenIdentityProvider.from_role_tokens(
        settings.demo_role_tokens
    ).authenticate(token, request_id="retrieval-diagnostic")
    if auth is None:
        raise DiagnosticError("diagnostic_authentication_failed")

    database = build_database(settings.database_url)
    authorized = DatabaseAuthorizationPort(database).authorized_active_versions(auth)
    result = build_enterprise_retriever(
        settings, database, consumer="fast"
    ).search(args.query, auth, scope_document_id=args.document_id)
    print(
        json.dumps(
            {
                "tenant_id": auth.tenant_id,
                "user_id": auth.user_id,
                "scope_document_id": args.document_id,
                "authorized_version_count": len(authorized.version_ids),
                "query_fingerprint": result.query_fingerprint,
                "sufficient": result.sufficient,
                "insufficiency_code": result.insufficiency_code,
                "context_token_estimate": result.context_token_estimate,
                "candidates": [
                    {
                        "document_id": item.document_id,
                        "version_id": item.version_id,
                        "document_version": item.document_version,
                        "projection_generation": item.projection_generation,
                        "chunk_id": item.chunk_id,
                        "dense_score": item.dense_score,
                        "sparse_score": item.sparse_score,
                        "rerank_score": item.rerank_score,
                    }
                    for item in result.candidates
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return run(argv)
    except Exception as exc:
        print(
            json.dumps(
                {"status": "error", "error_code": stable_error_code(exc)},
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
