"""DeepSeek thinking 与 tool_choice 组合的裸 HTTP 对照实验(不入库诊断工具)。

手动运行: python scripts/diagnose_thinking.py
"""

import json

import httpx

from rag_kb.config import get_settings

TOOL = {
    "type": "function",
    "function": {
        "name": "ModelVerdict",
        "description": "结构化回答",
        "parameters": {
            "type": "object",
            "properties": {
                "verdict": {"type": "string", "enum": ["answer", "refusal"]},
                "answer": {"type": "string"},
                "used_chunk_ids": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["verdict", "answer", "used_chunk_ids"],
        },
    },
}

PAYLOADS = {
    "A_disabled+tool_choice": {
        "thinking": {"type": "disabled"},
        "tools": [TOOL],
        "tool_choice": {"type": "function", "function": {"name": "ModelVerdict"}},
    },
    "B_no_thinking+tool_choice": {
        "tools": [TOOL],
        "tool_choice": {"type": "function", "function": {"name": "ModelVerdict"}},
    },
    "C_disabled_no_tools": {
        "thinking": {"type": "disabled"},
    },
}


def main() -> None:
    settings = get_settings()
    headers = {"Authorization": f"Bearer {settings.deepseek_api_key}"}
    for name, extra in PAYLOADS.items():
        payload = {
            "model": settings.deepseek_model,
            "messages": [{"role": "user", "content": "只回复:好"}],
            **extra,
        }
        try:
            resp = httpx.post(
                f"{settings.deepseek_base_url}/chat/completions",
                headers=headers, json=payload, timeout=60,
            )
            body = resp.json()
            if resp.status_code == 200:
                msg = body["choices"][0]["message"]
                print(f"{name}: 200 | content={str(msg.get('content'))[:30]!r} tool_calls={bool(msg.get('tool_calls'))}")
            else:
                print(f"{name}: {resp.status_code} | {body.get('error', {}).get('message', '')[:80]}")
        except Exception as exc:
            print(f"{name}: 异常 {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
