"""真实浏览器与三个生产 Worker 的隔离验收；只管理本次拥有的资源。"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import re
import secrets
import shutil
import socket
import subprocess
import sys
import time
from urllib.parse import urlsplit

import httpx
import yaml


REPO = Path(__file__).resolve().parents[2]
EVIDENCE_ROOT = REPO / "data" / "local" / "e2e"
OWNER_LABEL = "rag-kb.e2e.owner"
RUN_LABEL = "rag-kb.e2e.run"
SERVICES = ("postgres", "qdrant", "migrate", "qdrant-init", "api",
            "task-worker", "ingestion-worker", "projection-worker")
WORKERS = ("task-worker", "ingestion-worker", "projection-worker")
APP_SERVICES = ("api", "migrate", "qdrant-init", *WORKERS)
METRICS_PORTS = {"task-worker": 9464, "ingestion-worker": 9465, "projection-worker": 9466}
SQL_QUERIES = {
    "document_projection": ("SELECT tenant_id, document_id, applied_projection_version_id, applied_projection_generation, projection_reconcile_after, projection_reconcile_owner, projection_verified_at, projection_reconcile_lease_generation FROM documents WHERE document_id=:id", frozenset({"id"})),
    "document_version_projection": ("SELECT lifecycle_status, applied_projection_generation FROM document_versions WHERE version_id=:id", frozenset({"id"})),
    "ingestion_receipt": ("SELECT indexed_chunk_count, index_manifest_digest, index_completed_at FROM ingestion_jobs WHERE job_id=:id", frozenset({"id"})),
    "chunks_by_version": ("SELECT chunk_id, content_hash, source_locator FROM chunk_metadata WHERE version_id=:id", frozenset({"id"})),
    "task_capability": ("SELECT worker_id, capability_name, protocol_version FROM agent_capability_leases WHERE capability_name='task-agent' AND protocol_version='m2-fenced-v1' AND lease_expires_at > CURRENT_TIMESTAMP", frozenset()),
    "task_evidence": ("SELECT task_id, status, lease_generation FROM agent_tasks WHERE task_id=:id", frozenset({"id"})),
    "task_checkpoints": ("SELECT count(*) AS count FROM checkpoints WHERE thread_id=:id", frozenset({"id"})),
    "resume_commands_by_task": ("SELECT command_id, status, attempt_count FROM task_resume_commands WHERE task_id=:id", frozenset({"id"})),
    "artifact_ids_by_task": ("SELECT artifact_id FROM artifacts WHERE task_id=:id", frozenset({"id"})),
    "resume_command": ("SELECT command_id, status, attempt_count, applied_checkpoint_id FROM task_resume_commands WHERE command_id=:id", frozenset({"id"})),
    "artifacts_by_task": ("SELECT artifact_id, tool_call_id, content_hash, status FROM artifacts WHERE task_id=:id", frozenset({"id"})),
    "write_tool_calls_by_task": ("SELECT tool_call_id, approval_id, status, verified_at FROM tool_calls WHERE task_id=:id AND mode='write'", frozenset({"id"})),
}


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def evidence_directory(run_id: str) -> Path:
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{5,47}", run_id):
        raise ValueError("invalid_run_id")
    root = EVIDENCE_ROOT.resolve()
    target = (root / run_id).resolve()
    if target.parent != root:
        raise ValueError("evidence_path_outside_root")
    return target


def find_docker(explicit: str | None) -> str:
    candidates = [explicit] if explicit else [
        shutil.which("docker"),
        str(Path(os.environ.get("LOCALAPPDATA", "")) / "Programs/DockerDesktop/resources/bin/docker.exe"),
        "C:/Program Files/Docker/Docker/resources/bin/docker.exe",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return str(Path(candidate).resolve())
    raise RuntimeError("docker_client_not_found_use_docker_executable")


def find_browser_executable(channel: str | None) -> str | None:
    if channel != "msedge" or os.name != "nt":
        return None
    for candidate in (
        Path(os.environ.get("PROGRAMFILES(X86)", "C:/Program Files (x86)")) / "Microsoft/Edge/Application/msedge.exe",
        Path(os.environ.get("PROGRAMFILES", "C:/Program Files")) / "Microsoft/Edge/Application/msedge.exe",
        Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft/Edge/Application/msedge.exe",
    ):
        if candidate.is_file():
            return str(candidate.resolve())
    raise RuntimeError("edge_executable_not_found")


def isolated_compose(source: dict, *, run_id: str, owner: str, model_cache: Path) -> dict:
    """重新构造宿主绑定，避免 Compose 合并保留业务端口/共享卷。"""
    labels = {OWNER_LABEL: owner, RUN_LABEL: run_id}
    project = f"rag-kb-e2e-{run_id}"
    result = {"name": project, "services": {},
              "networks": {"default": {"labels": labels.copy()}},
              "volumes": {name: {"labels": labels.copy()} for name in
                          ("postgres_data", "qdrant_storage", "object_storage")}}
    for name in SERVICES:
        original = source["services"][name]
        service = {key: deepcopy(original[key]) for key in
                   ("command", "healthcheck", "depends_on", "expose") if key in original}
        service.update(labels=labels.copy(), restart="no")
        if name in APP_SERVICES:
            service["image"] = f"{project}:latest"
            if name == "api":
                service["build"] = {"context": str((REPO / "backend").resolve()), "labels": labels.copy()}
            env = {"DATABASE_URL": "postgresql+psycopg://rag_kb:${E2E_POSTGRES_PASSWORD:?required}@postgres:5432/rag_kb",
                   "QDRANT_URL": "http://qdrant:6333",
                   "QDRANT_V2_COLLECTION": f"e2e_{run_id.replace('-', '_')}",
                   "OBJECT_STORE_ROOT": "/app/data/local/objects",
                   "RAG_KB_DISABLE_ENV_FILE": "1", "LANGGRAPH_STRICT_MSGPACK": "true",
                   "HF_HOME": "/home/ragkb/.cache/huggingface", "HF_HUB_OFFLINE": "1",
                   "HF_HUB_DISABLE_XET": "1", "PYTHONUTF8": "1",
                   "OBSERVABILITY_ENABLED": "true", "METRICS_ENABLED": "true",
                   "OTEL_EXPORTER_OTLP_ENDPOINT": "", "OTEL_SERVICE_NAME": f"e2e-{name}",
                   "EMBEDDING_DIM": "${E2E_EMBEDDING_DIM:?required}"}
            # 继承已声明的非连接运行参数；只使用 Compose 默认值，不继承宿主同名变量。
            for key, value in original.get("environment", {}).items():
                if key.startswith(("RETRIEVAL_", "AGENT_", "TASK_AGENT_", "TOOL_RESULT_",
                                   "TASK_RESULT_", "RESULT_OBJECT_", "ENABLE_")):
                    if isinstance(value, str) and value.startswith("${"):
                        match = re.fullmatch(r"\$\{[^:}]+:-([^}]*)\}", value)
                        if not match:
                            raise ValueError("unsupported_runtime_default")
                        value = match.group(1)
                    env[key] = str(value)
            if name in ("api", "task-worker"):
                env.update(DEEPSEEK_API_KEY="${E2E_DEEPSEEK_API_KEY:?required}",
                           DEEPSEEK_MODEL="${E2E_DEEPSEEK_MODEL:?required}",
                           DEEPSEEK_BASE_URL="${E2E_DEEPSEEK_BASE_URL:?required}")
            if name == "api":
                env["DEMO_ROLE_TOKENS"] = "${E2E_DEMO_ROLE_TOKENS:?required}"
                service["ports"] = [{"target": 8000, "host_ip": "127.0.0.1", "protocol": "tcp"}]
            if name in WORKERS:
                env["WORKER_METRICS_PORT"] = str(METRICS_PORTS[name])
            service["environment"] = env
            if name in ("api", *WORKERS):
                service["volumes"] = ["object_storage:/app/data/local/objects",
                    {"type": "bind", "source": str(model_cache.resolve()),
                     "target": "/home/ragkb/.cache/huggingface", "read_only": True}]
        else:
            service["image"] = original["image"]
            if name == "postgres":
                service["environment"] = {"POSTGRES_DB": "rag_kb", "POSTGRES_USER": "rag_kb",
                                          "POSTGRES_PASSWORD": "${E2E_POSTGRES_PASSWORD:?required}"}
                service["volumes"] = ["postgres_data:/var/lib/postgresql"]
            else:
                service["volumes"] = ["qdrant_storage:/qdrant/storage"]
        result["services"][name] = service
    return result


def verify_isolated_compose(config: dict, *, run_id: str, owner: str, model_cache: Path) -> None:
    """在 Docker 创建任何资源前拒绝共享绑定或非本次配置。"""
    project = f"rag-kb-e2e-{run_id}"
    labels = {OWNER_LABEL: owner, RUN_LABEL: run_id}
    services = config.get("services")
    if not isinstance(services, dict) or set(services) != set(SERVICES):
        raise ValueError("isolated_service_set_invalid")
    if config.get("name") != project:
        raise ValueError("isolated_project_name_invalid")
    for name, service in services.items():
        if not isinstance(service, dict) or any(service.get("labels", {}).get(key) != value for key, value in labels.items()):
            raise ValueError("isolated_service_labels_invalid")
        if service.get("container_name") or service.get("restart") != "no":
            raise ValueError("isolated_service_lifecycle_invalid")
        ports = service.get("ports", [])
        if name == "api":
            if len(ports) != 1 or not isinstance(ports[0], dict) or ports[0].get("target") != 8000 or ports[0].get("host_ip") != "127.0.0.1" or ports[0].get("published") not in (None, "", 0, "0"):
                raise ValueError("isolated_api_port_invalid")
        elif ports:
            raise ValueError("isolated_shared_port_invalid")
        if name not in APP_SERVICES:
            expected_mount = "postgres_data:/var/lib/postgresql" if name == "postgres" else "qdrant_storage:/qdrant/storage"
            if service.get("volumes") != [expected_mount]:
                raise ValueError("isolated_storage_volume_invalid")
            continue
        if service.get("image") != f"{project}:latest":
            raise ValueError("isolated_app_image_invalid")
        env = service.get("environment", {})
        if not isinstance(env, dict) or env.get("QDRANT_URL") != "http://qdrant:6333" or env.get("QDRANT_V2_COLLECTION") != f"e2e_{run_id.replace('-', '_')}" or env.get("OBJECT_STORE_ROOT") != "/app/data/local/objects" or env.get("RAG_KB_DISABLE_ENV_FILE") != "1":
            raise ValueError("isolated_runtime_binding_invalid")
        database_url = str(env.get("DATABASE_URL", ""))
        if "@postgres:5432/rag_kb" not in database_url or "localhost" in database_url:
            raise ValueError("isolated_database_binding_invalid")
        if name in ("api", *WORKERS):
            mounts = service.get("volumes", [])
            if len(mounts) != 2 or not any(mount == "object_storage:/app/data/local/objects" for mount in mounts):
                raise ValueError("isolated_object_volume_invalid")
            if not any(isinstance(mount, dict) and mount.get("type") == "bind" and mount.get("source") == str(model_cache.resolve()) and mount.get("target") == "/home/ragkb/.cache/huggingface" and mount.get("read_only") is True for mount in mounts):
                raise ValueError("isolated_model_mount_invalid")
        elif service.get("volumes"):
            raise ValueError("isolated_unneeded_volume_invalid")
    build = services["api"].get("build")
    if not isinstance(build, dict) or build.get("context") != str((REPO / "backend").resolve()) or any(build.get("labels", {}).get(key) != value for key, value in labels.items()):
        raise ValueError("isolated_build_labels_invalid")
    volumes = config.get("volumes", {})
    if not isinstance(volumes, dict) or set(volumes) != {"postgres_data", "qdrant_storage", "object_storage"}:
        raise ValueError("isolated_volume_set_invalid")
    for name, volume in volumes.items():
        if volume.get("external") or volume.get("name") or any(volume.get("labels", {}).get(key) != value for key, value in labels.items()):
            raise ValueError("isolated_volume_labels_invalid")
    networks = config.get("networks", {})
    if not isinstance(networks, dict) or set(networks) != {"default"}:
        raise ValueError("isolated_network_set_invalid")
    network = networks["default"]
    if network.get("external") or network.get("name") or any(network.get("labels", {}).get(key) != value for key, value in labels.items()):
        raise ValueError("isolated_network_labels_invalid")


class Redactor:
    def __init__(self, values=()):
        self.values = sorted({str(value) for value in values if value}, key=len, reverse=True)

    def text(self, value: str) -> str:
        for secret in self.values:
            value = value.replace(secret, "[REDACTED]")
        value = re.sub(r"(?i)(bearer\s+)[^\s\"'<>]+", r"\1[REDACTED]", value)
        value = re.sub(r"(://[^\s/:]+:)[^@\s]+@", r"\1[REDACTED]@", value)
        value = re.sub(r"\be2e_token_[A-Za-z0-9_-]+", "[REDACTED]", value)
        return value

    def data(self, value):
        return json.loads(self.text(json.dumps(value, ensure_ascii=False, default=str)))


def process_identity(pid: int) -> dict:
    """保存内核创建时间，防止异常清理时误杀已复用的 PID。"""
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.GetProcessTimes.argtypes = [wintypes.HANDLE, *([ctypes.POINTER(wintypes.FILETIME)] * 4)]
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            raise ProcessLookupError(pid)
        try:
            times = [wintypes.FILETIME() for _ in range(4)]
            if not kernel.GetProcessTimes(handle, *(ctypes.byref(item) for item in times)):
                raise ProcessLookupError(pid)
            return {"pid": pid, "created": (times[0].dwHighDateTime << 32) | times[0].dwLowDateTime}
        finally:
            kernel.CloseHandle(handle)
    stat = Path(f"/proc/{pid}/stat").read_text()
    return {"pid": pid, "created": stat[stat.rfind(")") + 2:].split()[19]}


class Harness:
    def __init__(self, run_id: str, directory: Path, docker: str | None = None):
        self.run_id, self.directory, self.docker = run_id, directory, docker
        self.project = f"rag-kb-e2e-{run_id}"
        self.collection = f"e2e_{run_id.replace('-', '_')}"
        self.owner = secrets.token_hex(16)
        self.env = os.environ.copy()
        for key in ("COMPOSE_FILE", "COMPOSE_PROJECT_NAME", "COMPOSE_PROFILES", "COMPOSE_ENV_FILES"):
            self.env.pop(key, None)
        self.redactor = Redactor()
        self.report = {"run_id": run_id, "started_at": utc_now(), "status": "running", "scenarios": []}
        self.manifest = {"run_id": run_id, "project": self.project, "owner": self.owner,
                         "containers": {}, "networks": {}, "volumes": {}, "images": {}}
        self.api_url = ""
        self.tokens = {}
        self.deadline = None
        self.frontend = None
        self.frontend_log = None
        self.browser_channel = None
        self.browser_executable = None

    def save(self):
        self.record("report", self.report)
        self.record("resources", self.manifest)

    def record(self, name: str, data):
        if not re.fullmatch(r"[a-zA-Z0-9_.-]+", name):
            raise ValueError("invalid_evidence_name")
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f"{name}.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.redactor.data(data), ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)

    def event(self, message: str):
        print(json.dumps({"time": utc_now(), "event": self.redactor.text(message)}, ensure_ascii=False), flush=True)

    @staticmethod
    def check(condition, code: str):
        if not condition:
            raise RuntimeError(code)

    @contextmanager
    def scenario(self, name: str):
        entry = {"name": name, "started_at": utc_now(), "status": "running"}
        self.report["scenarios"].append(entry)
        self.save()
        self.event(f"开始：{name}")
        try:
            yield
        except BaseException as exc:
            entry.update(status="failed", error=self.redactor.text(str(exc))[:1600])
            raise
        else:
            entry["status"] = "passed"
        finally:
            entry["finished_at"] = utc_now()
            self.save()
            self.event(f"{name}：{entry['status']}")

    @staticmethod
    def _popen_options():
        if os.name == "nt":
            return {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0) |
                    getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)}
        return {"start_new_session": True}

    def _wait_for_process_exit(self, expected, timeout):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if not self._process_running(expected["pid"]):
                return True
            try:
                actual = process_identity(expected["pid"])
            except (ProcessLookupError, FileNotFoundError):
                return True
            self.check(actual == expected, "managed_process_pid_reused")
            time.sleep(0.1)
        return False

    @staticmethod
    def _process_running(pid):
        if os.name != "nt":
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                return False
            return True
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.GetExitCodeProcess.restype = wintypes.BOOL
        kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            return False
        try:
            code = wintypes.DWORD()
            if not kernel.GetExitCodeProcess(handle, ctypes.byref(code)):
                raise RuntimeError("managed_process_state_unavailable")
            return code.value == 259
        finally:
            kernel.CloseHandle(handle)

    def _terminate_process_tree(self, expected, *, timeout=20):
        """只终止已记录且创建时间匹配的本次进程树。"""
        if not self._process_running(expected["pid"]):
            return
        try:
            actual = process_identity(expected["pid"])
        except (ProcessLookupError, FileNotFoundError):
            return
        self.check(actual == expected, "managed_process_pid_reused")
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(expected["pid"]), "/T"], capture_output=True,
                           text=True, encoding="utf-8", errors="replace", timeout=timeout)
            if not self._wait_for_process_exit(expected, timeout):
                self.check(process_identity(expected["pid"]) == expected, "managed_process_pid_reused")
                subprocess.run(["taskkill", "/PID", str(expected["pid"]), "/T", "/F"], capture_output=True,
                               text=True, encoding="utf-8", errors="replace", timeout=timeout)
                self.check(self._wait_for_process_exit(expected, timeout), "managed_process_termination_failed")
            return
        import signal
        os.killpg(expected["pid"], signal.SIGTERM)
        if not self._wait_for_process_exit(expected, timeout):
            self.check(process_identity(expected["pid"]) == expected, "managed_process_pid_reused")
            os.killpg(expected["pid"], signal.SIGKILL)
            self.check(self._wait_for_process_exit(expected, timeout), "managed_process_termination_failed")

    def _run(self, args, *, timeout=120, stdin=None, env=None, check=True, cwd=None):
        if self.deadline:
            timeout = min(timeout, max(1, self.deadline - time.monotonic()))
            self.check(time.monotonic() < self.deadline, "overall_deadline_exceeded")
        process = subprocess.Popen(args, stdin=subprocess.PIPE if stdin is not None else None,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                   encoding="utf-8", errors="replace", env=env or self.env,
                                   cwd=cwd or REPO, **self._popen_options())
        identity = process_identity(process.pid)
        try:
            stdout, stderr = process.communicate(input=stdin, timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            self._terminate_process_tree(identity)
            output = (exc.stdout or "") + (exc.stderr or "")
            if isinstance(output, bytes):
                output = output.decode("utf-8", "replace")
            self.record("last-command-error", {"executable": Path(args[0]).name,
                                                 "code": "timeout", "output": output[-16000:]})
            raise RuntimeError("subprocess_timeout") from None
        result = subprocess.CompletedProcess(args, process.returncode, stdout, stderr)
        if check and result.returncode:
            self.record("last-command-error", {"executable": Path(args[0]).name,
                        "code": result.returncode, "output": (result.stdout + result.stderr)[-16000:]})
            raise RuntimeError(f"subprocess_failed_{Path(args[0]).name}_{result.returncode}")
        return result

    def dc(self, *args, **kwargs):
        return self._run([self.docker, "compose", "--project-name", self.project,
                          "--env-file", str(self.directory / "compose.env"),
                          "-f", str(self.directory / "compose.yaml"), *args], **kwargs)

    def docker_json(self, *args):
        return json.loads(self._run([self.docker, *args]).stdout)

    def poll(self, label, fn, timeout=120, interval=2):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if self.deadline:
                self.check(time.monotonic() < self.deadline, "overall_deadline_exceeded")
            value = fn()
            if value:
                return value
            time.sleep(min(interval, max(0, end - time.monotonic())))
        raise RuntimeError(f"timeout_{label}")

    def inspect_resource(self, kind, identifier):
        if kind == "containers":
            template = '{"id":{{json .Id}},"name":{{json .Name}},"labels":{{json .Config.Labels}},"created":{{json .Created}},"state_detail":{{json .State}},"image":{{json .Image}}}'
            args = ["inspect", "--type", "container", "--format", template, identifier]
            item = self.docker_json(*args)
            state = item.pop("state_detail")
            item.update(state=state["Status"], started=state["StartedAt"],
                        health=(state.get("Health") or {}).get("Status"), exit_code=state["ExitCode"])
            return item
        elif kind == "networks":
            args = ["network", "inspect", "--format", '{"id":{{json .Id}},"name":{{json .Name}},"labels":{{json .Labels}},"created":{{json .Created}}}', identifier]
        else:
            args = ["volume", "inspect", "--format", '{"id":{{json .Name}},"name":{{json .Name}},"labels":{{json .Labels}},"created":{{json .CreatedAt}}}', identifier]
        return self.docker_json(*args)

    def inspect_image(self, identifier):
        return self.docker_json("image", "inspect", "--format",
                                '{"id":{{json .Id}},"labels":{{json .Config.Labels}},"created":{{json .Created}}}', identifier)

    def owned(self, item):
        labels = item.get("labels") or {}
        return (labels.get("com.docker.compose.project") == self.project
                and labels.get(OWNER_LABEL) == self.owner and labels.get(RUN_LABEL) == self.run_id)

    def owned_image(self, item):
        labels = item.get("labels") or {}
        return labels.get(OWNER_LABEL) == self.owner and labels.get(RUN_LABEL) == self.run_id

    def discover(self):
        """启动部分失败时也按双标签补登记，不依赖成功返回的 compose up。"""
        for kind, command in (("containers", ["ps", "-a"]), ("networks", ["network", "ls"]), ("volumes", ["volume", "ls"])):
            ids = self._run([self.docker, *command, "--filter", f"label=com.docker.compose.project={self.project}",
                            "--filter", f"label={OWNER_LABEL}={self.owner}", "--quiet"]).stdout.split()
            for identifier in ids:
                item = self.inspect_resource(kind, identifier)
                self.check(self.owned(item), "resource_ownership_mismatch")
                old = self.manifest[kind].get(item["id"])
                if old:
                    self.check(old["created"] == item["created"], "resource_identity_changed")
                self.manifest[kind][item["id"]] = item
        images = self.manifest.setdefault("images", {})
        ids = self._run([self.docker, "image", "ls", "--filter", f"label={OWNER_LABEL}={self.owner}",
                         "--filter", f"label={RUN_LABEL}={self.run_id}", "--quiet", "--no-trunc"]).stdout.split()
        for identifier in ids:
            item = self.inspect_image(identifier)
            self.check(self.owned_image(item), "image_ownership_mismatch")
            old = images.get(item["id"])
            if old:
                self.check(old["created"] == item["created"], "image_identity_changed")
            images[item["id"]] = item
        self.save()

    def container_identity(self, service):
        self.check(service in SERVICES, "unknown_service")
        self.discover()
        matches = [item for item in self.manifest["containers"].values()
                   if item["labels"].get("com.docker.compose.service") == service
                   and item["labels"].get("com.docker.compose.oneoff", "False").lower() == "false"]
        self.check(len(matches) == 1, f"service_identity_ambiguous_{service}")
        item = self.inspect_resource("containers", matches[0]["id"])
        self.check(self.owned(item) and item["created"] == matches[0]["created"], "container_identity_mismatch")
        return item

    def runtime_binding(self, service):
        self.check(service in APP_SERVICES, "runtime_binding_service_invalid")
        item = self.container_identity(service)
        entries = self.docker_json("inspect", "--type", "container", "--format", "{{json .Config.Env}}", item["id"])
        env = dict(entry.split("=", 1) for entry in entries if "=" in entry)
        database_url = env.get("DATABASE_URL", "")
        self.check(re.fullmatch(r"postgresql\+psycopg://rag_kb:[^@]+@postgres:5432/rag_kb", database_url) is not None,
                   "runtime_database_binding_invalid")
        self.check(env.get("QDRANT_URL") == "http://qdrant:6333" and env.get("QDRANT_V2_COLLECTION") == self.collection
                   and env.get("OBJECT_STORE_ROOT") == "/app/data/local/objects" and env.get("RAG_KB_DISABLE_ENV_FILE") == "1",
                   "runtime_binding_mismatch")
        if service in ("api", "task-worker"):
            self.check(env.get("DEEPSEEK_API_KEY") == self.env.get("E2E_DEEPSEEK_API_KEY")
                       and env.get("DEEPSEEK_MODEL") == self.env.get("E2E_DEEPSEEK_MODEL")
                       and env.get("DEEPSEEK_BASE_URL") == self.env.get("E2E_DEEPSEEK_BASE_URL"),
                       "runtime_model_binding_mismatch")
        else:
            self.check("DEEPSEEK_API_KEY" not in env, "runtime_unneeded_model_key")
        if service == "api":
            self.check(json.loads(env.get("DEMO_ROLE_TOKENS", "")) == self.tokens, "runtime_demo_tokens_mismatch")
        if service in WORKERS:
            self.check(env.get("WORKER_METRICS_PORT") == str(METRICS_PORTS[service]), "runtime_metrics_binding_mismatch")
        return {"container_id": item["id"], "database_host": "postgres", "database": "rag_kb",
                "qdrant": env["QDRANT_URL"], "collection": env["QDRANT_V2_COLLECTION"],
                "object_root": env["OBJECT_STORE_ROOT"]}

    def verify_runtime_bindings(self):
        bindings = {service: self.runtime_binding(service) for service in APP_SERVICES}
        self.record("runtime-binding", bindings)

    def wait_healthy(self, service):
        def probe():
            item = self.container_identity(service)
            return item if item["state"] == "running" and item["health"] == "healthy" else None
        return self.poll("health:" + service, probe, timeout=120)

    def wait_completed(self, service):
        def probe():
            item = self.container_identity(service)
            self.check(not (item["state"] == "exited" and item["exit_code"] != 0), f"service_failed_{service}")
            return item if item["state"] == "exited" and item["exit_code"] == 0 else None
        return self.poll("completion:" + service, probe, timeout=180)

    def start(self, service):
        self.check(service in WORKERS, "only_owned_workers_can_start")
        self.runtime_binding(service)
        self.dc("start", service, timeout=300)
        self.discover()
        self.check(self.container_identity(service)["state"] == "running", "worker_not_running")

    def stop(self, service):
        self.check(service in WORKERS, "only_owned_workers_can_stop")
        item = self.container_identity(service)
        self._run([self.docker, "stop", "--time", "30", item["id"]], timeout=45)
        self.check(self.container_identity(service)["state"] == "exited", "worker_not_stopped")

    def exec_python(self, code, payload=None, *, timeout=60):
        item = self.container_identity("api")
        return self._run([self.docker, "exec", "-i", item["id"], "python", "-c", code],
                         stdin=json.dumps(payload or {}), timeout=timeout).stdout

    def sql(self, query_name, params=None):
        self.check(query_name in SQL_QUERIES, "sql_query_not_allowlisted")
        query, expected_params = SQL_QUERIES[query_name]
        params = params or {}
        self.check(set(params) == expected_params, "sql_parameters_invalid")
        code = """import json,sys,os
from sqlalchemy import create_engine,text
p=json.load(sys.stdin)
engine=create_engine(os.environ['DATABASE_URL'])
with engine.connect() as c:
 c.exec_driver_sql('SET TRANSACTION READ ONLY')
 rows=c.execute(text(p['query']),p['params']).mappings().all()
 print(json.dumps([dict(r) for r in rows],default=str))
engine.dispose()
"""
        return json.loads(self.exec_python(code, {"query": query, "params": params}))

    def internal_http(self, method, url, payload=None):
        code = """import json,sys,urllib.request
p=json.load(sys.stdin)
data=json.dumps(p['payload']).encode() if p['payload'] is not None else None
req=urllib.request.Request(p['url'],data=data,method=p['method'],headers={'Content-Type':'application/json'})
with urllib.request.urlopen(req,timeout=30) as r: print(r.read().decode())
"""
        return self.exec_python(code, {"method": method, "url": url, "payload": payload})

    def qdrant(self, method, path, json=None, *, expected_binding=None):
        prefix = f"/collections/{self.collection}"
        self.check(path == prefix or path.startswith(prefix + "/"), "qdrant_collection_outside_run")
        self.check(".." not in path and "%" not in path and "#" not in path, "invalid_qdrant_path")
        self.container_identity("qdrant")
        allowed = {prefix, prefix + "/points/scroll", prefix + "/points/count", prefix + "/points"}
        deleting = path == prefix + "/points/delete?wait=true"
        self.check((method == "GET" and path == prefix) or
                   (method == "POST" and path in allowed - {prefix}) or
                   (method == "POST" and deleting), "qdrant_operation_not_allowed")
        if deleting:
            self.check(isinstance(json, dict) and set(json) == {"points"}
                        and isinstance(json["points"], list) and len(json["points"]) == 1,
                        "fault_requires_one_exact_point")
            required = {"point_id", "tenant_id", "document_id", "version_id", "projection_generation",
                        "projection_key", "current", "lifecycle_status"}
            self.check(isinstance(expected_binding, dict) and set(expected_binding) == required,
                       "fault_binding_required")
            self.check(json["points"][0] == expected_binding["point_id"], "fault_point_id_mismatch")
            points = self.qdrant("POST", prefix + "/points", {"ids": json["points"], "with_payload": True})["result"]
            self.check(len(points) == 1, "fault_point_not_found")
            payload = points[0]["payload"]
            self.check(points[0]["id"] == expected_binding["point_id"]
                       and all(payload.get(key) == expected_binding[key] for key in
                               ("tenant_id", "document_id", "version_id", "projection_generation", "projection_key",
                                "current", "lifecycle_status"))
                       and payload.get("current") is True and payload.get("lifecycle_status") == "active",
                       "fault_point_binding_invalid")
            self.record("fault-point-binding", {"id": points[0]["id"], "binding": expected_binding})
        return __import__("json").loads(self.internal_http(method, "http://qdrant:6333" + path, json))

    def metrics(self, service):
        self.check(service in WORKERS, "invalid_metrics_service")
        self.container_identity(service)
        return self.internal_http("GET", f"http://{service}:{METRICS_PORTS[service]}/metrics")

    def request(self, method, path, role="admin", expected=200, **kwargs):
        self.check(path.startswith("/v1/") or path.startswith("/health/"), "invalid_api_path")
        headers = dict(kwargs.pop("headers", {}))
        headers["Authorization"] = f"Bearer {self.tokens[role]}"
        response = httpx.request(method, self.api_url + path, headers=headers,
                                 timeout=kwargs.pop("timeout", 60), **kwargs)
        self.check(response.status_code == expected, f"http_{method}_{path.split('?')[0]}_{response.status_code}_expected_{expected}")
        return response

    def preflight(self, browser_channel):
        from rag_kb.config import Settings
        settings = Settings()
        self.check(bool(settings.deepseek_api_key), "deepseek_key_required")
        self.check(settings._bge_m3_fully_cached(), "frozen_embedding_model_required")
        self.check(urlsplit(settings.deepseek_base_url).scheme == "https", "https_model_endpoint_required")
        self.check(shutil.which("node") is not None, "node_required")
        self.check((REPO / "frontend/node_modules/@playwright/test/cli.js").is_file(), "existing_playwright_required")
        self.browser_channel = browser_channel
        self.browser_executable = find_browser_executable(browser_channel)
        # 启动/关闭一个独立空浏览器，仅验证现有安装，不下载。
        code = "const {chromium}=require('@playwright/test');chromium.launch({executablePath:process.argv[1]||undefined,channel:process.argv[2]||undefined}).then(b=>b.close()).catch(()=>{console.error('browser_preflight_failed');process.exit(1)});"
        self._run([shutil.which("node"), "-e", code, self.browser_executable or "",
                   "" if self.browser_executable else browser_channel or ""], cwd=REPO / "frontend")
        daemon = self.docker_json("info", "--format", '{{json .ID}}')
        self.check(bool(daemon), "docker_engine_required")
        self.manifest["daemon_id"] = daemon
        self._run([self.docker, "compose", "version", "--short"])
        existing = self._run([self.docker, "ps", "-aq", "--filter", f"label=com.docker.compose.project={self.project}"]).stdout.strip()
        self.check(not existing, "project_already_exists")
        self.manifest["existing_container_ids"] = self._run([self.docker, "ps", "-q", "--no-trunc"]).stdout.split()
        self.tokens = {role: "e2e_token_" + secrets.token_urlsafe(24) for role in ("admin", "approver")}
        password = secrets.token_hex(24)
        self.redactor = Redactor([settings.deepseek_api_key, password, *self.tokens.values()])
        self.env.update(E2E_POSTGRES_PASSWORD=password, E2E_DEEPSEEK_API_KEY=settings.deepseek_api_key,
                        E2E_DEEPSEEK_BASE_URL=settings.deepseek_base_url, E2E_DEEPSEEK_MODEL=settings.deepseek_model,
                        E2E_EMBEDDING_DIM=str(settings.embedding_dim), E2E_DEMO_ROLE_TOKENS=json.dumps(self.tokens))
        cache = settings.hf_home or Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface"))
        config = isolated_compose(yaml.safe_load((REPO / "docker-compose.yml").read_text(encoding="utf-8")),
                                   run_id=self.run_id, owner=self.owner, model_cache=cache)
        verify_isolated_compose(config, run_id=self.run_id, owner=self.owner, model_cache=cache)
        (self.directory / "compose.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        (self.directory / "compose.env").write_text("# 本次隔离执行使用进程环境；禁止自动加载仓库 .env。\n", encoding="utf-8")
        self.dc("config", "--quiet")
        from scripts.build_project_acceptance_report import git_worktree_fingerprint
        self.record("execution", {"python": sys.executable, "source_fingerprint": git_worktree_fingerprint(),
                    "model": settings.deepseek_model, "browser_channel": browser_channel,
                    "versions": self._run([sys.executable, "-m", "pip", "freeze"]).stdout.splitlines(),
                    "scenarios": ["ingestion", "research", "approval-recovery", "browser", "projection-repair"]})
        self.save()

    def initialize(self):
        self.event("构建本次后端镜像")
        result = self.dc("build", "api", timeout=1800)
        (self.directory / "build.log").write_text(self.redactor.text(result.stdout + result.stderr), encoding="utf-8")
        self.deadline = time.monotonic() + 3600
        self.discover()
        image = self.inspect_image(f"{self.project}:latest")
        self.check(self.owned_image(image), "image_ownership_mismatch")
        self.manifest["image_id"] = image["id"]
        self.manifest["images"][image["id"]] = image
        self.save()
        self.dc("create", *SERVICES, timeout=300)
        self.discover()
        for service in APP_SERVICES:
            self.check(self.container_identity(service)["image"] == image["id"], "service_image_mismatch")
        # 所有实际容器在启动迁移、初始化或 Worker 前完成隔离环境核对。
        self.verify_runtime_bindings()
        self.dc("start", "postgres", "qdrant", timeout=300)
        self.wait_healthy("postgres")
        self.wait_healthy("qdrant")
        self.dc("start", "migrate", "qdrant-init", timeout=300)
        self.wait_completed("migrate")
        self.wait_completed("qdrant-init")
        self.dc("start", "api", timeout=300)
        mapped = self.dc("port", "api", "8000").stdout.strip()
        self.check(bool(re.fullmatch(r"127\.0\.0\.1:\d+", mapped)), "api_binding_not_loopback")
        self.api_url = "http://" + mapped
        self.manifest["api_url"] = self.api_url
        def ready():
            try:
                return httpx.get(self.api_url + "/health/ready", timeout=5).status_code == 200
            except httpx.HTTPError:
                return False
        self.poll("api_readiness", ready, timeout=120)
        item = self.container_identity("api")
        self._run([self.docker, "exec", item["id"], "python", "scripts/init_demo_principals.py"])
        self.save()

    def frontend_environment(self, *, browser=False):
        # 前端不继承宿主模型密钥、数据库连接或 Compose 凭据。
        allowed = {"PATH", "SYSTEMROOT", "WINDIR", "TEMP", "TMP", "HOME", "USERPROFILE", "LOCALAPPDATA",
                   "APPDATA", "COMSPEC", "PATHEXT", "PLAYWRIGHT_BROWSERS_PATH"}
        env = {key: value for key, value in os.environ.items() if key.upper() in allowed}
        env.update(RAG_KB_E2E_RUN_ID=self.run_id, RAG_KB_E2E_API_URL=self.api_url,
                   RAG_KB_E2E_BASE_URL=self.manifest["frontend_url"],
                   RAG_KB_E2E_OUTPUT_DIR=str(self.directory / "browser"), NO_COLOR="1")
        if browser:
            env.update(RAG_KB_E2E_ADMIN_TOKEN=self.tokens["admin"], RAG_KB_E2E_APPROVER_TOKEN=self.tokens["approver"])
            if self.browser_executable:
                env["PLAYWRIGHT_EXECUTABLE_PATH"] = self.browser_executable
            elif self.browser_channel:
                env["PLAYWRIGHT_CHANNEL"] = self.browser_channel
        return env

    def verify_browser_report(self):
        path = self.directory / "browser" / "browser-report.json"
        self.check(path.is_file(), "browser_report_missing")
        report = json.loads(path.read_text(encoding="utf-8"))
        stats = report.get("stats", {})
        self.check(stats.get("expected") == 1 and stats.get("skipped") == 0
                   and stats.get("unexpected") == 0 and stats.get("flaky") == 0
                   and not report.get("errors"), "browser_report_not_clean")
        tests = []

        def collect(suites):
            for suite in suites:
                for spec in suite.get("specs", []):
                    tests.extend(spec.get("tests", []))
                collect(suite.get("suites", []))

        collect(report.get("suites", []))
        self.check(len(tests) == 1 and tests[0].get("status") == "expected"
                   and len(tests[0].get("results", [])) == 1
                   and tests[0]["results"][0].get("status") == "passed", "browser_report_test_missing")
        path.write_text(json.dumps(self.redactor.data(report), ensure_ascii=False, indent=2), encoding="utf-8")
        return {"expected": stats["expected"], "skipped": stats["skipped"],
                "unexpected": stats["unexpected"], "flaky": stats["flaky"], "test_count": len(tests)}

    def run_browser(self):
        with self.scenario("frontend-live"):
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
            self.manifest["frontend_url"] = f"http://127.0.0.1:{port}"
            args = [shutil.which("node"), str(REPO / "frontend/node_modules/vite/bin/vite.js"),
                    "--host", "127.0.0.1", "--port", str(port), "--strictPort", "--mode", f"e2e_{self.run_id}"]
            self.frontend_log = (self.directory / "vite.log").open("w", encoding="utf-8")
            self.frontend = subprocess.Popen(args, cwd=REPO / "frontend", env=self.frontend_environment(),
                    stdout=self.frontend_log, stderr=subprocess.STDOUT,
                    **self._popen_options())
            self.manifest["frontend_process"] = process_identity(self.frontend.pid)
            self.save()
            def ready():
                self.check(self.frontend.poll() is None, "frontend_exited_during_startup")
                try:
                    return httpx.get(self.manifest["frontend_url"], timeout=3).status_code == 200
                except httpx.HTTPError:
                    return False
            self.poll("frontend_readiness", ready, timeout=60)
            result = self._run([shutil.which("node"), "node_modules/@playwright/test/cli.js", "test", "--config", "playwright.live.config.ts"],
                                cwd=REPO / "frontend", env=self.frontend_environment(browser=True), timeout=1800, check=False)
            (self.directory / "browser.log").write_text(self.redactor.text(result.stdout + result.stderr), encoding="utf-8")
            self.record("browser-report-summary", self.verify_browser_report())
            self.check(result.returncode == 0, "frontend_live_tests_failed")

    def diagnostics(self):
        if not self.docker or not self.manifest.get("daemon_id"):
            return
        try:
            self.discover()
            for item in self.manifest["containers"].values():
                self.check(self.owned(self.inspect_resource("containers", item["id"])), "diagnostics_ownership_mismatch")
                result = self._run([self.docker, "logs", "--tail", "200", item["id"]], check=False)
                name = item["labels"].get("com.docker.compose.service", "container")
                (self.directory / f"{name}.log").write_text(self.redactor.text(result.stdout + result.stderr), encoding="utf-8")
        except Exception as exc:
            self.report["diagnostics_error"] = self.redactor.text(str(exc))

    def cleanup(self):
        """不执行全局/down清理；重新核对每个已登记对象的所有权与身份。"""
        self.deadline = None
        self.manifest.setdefault("images", {})
        errors = []
        try:
            if self.frontend:
                self.terminate_recorded_frontend()
            elif self.manifest.get("frontend_process"):
                self.terminate_recorded_frontend()
        except Exception as exc:
            errors.append(str(exc))
        if self.frontend_log:
            self.frontend_log.close()
        if self.docker and self.manifest.get("daemon_id"):
            try:
                self.check(self.docker_json("info", "--format", '{{json .ID}}') == self.manifest["daemon_id"], "docker_daemon_changed")
                self.discover()
                for kind, command in (("containers", ["rm", "-f"]), ("networks", ["network", "rm"]), ("volumes", ["volume", "rm"])):
                    for identifier, expected in list(self.manifest[kind].items()):
                        try:
                            current = self.inspect_resource(kind, identifier)
                            self.check(self.owned(current) and current["id"] == expected["id"] and current["created"] == expected["created"], "cleanup_identity_mismatch")
                            self._run([self.docker, *command, identifier], timeout=60)
                            del self.manifest[kind][identifier]
                            self.save()
                        except Exception as exc:
                            errors.append(f"{kind}:{identifier}:{exc}")
                for identifier, expected in list(self.manifest.get("images", {}).items()):
                    try:
                        current = self.inspect_image(identifier)
                        self.check(self.owned_image(current) and current["id"] == expected["id"]
                                   and current["created"] == expected["created"], "cleanup_image_identity_mismatch")
                        self._run([self.docker, "image", "rm", identifier], timeout=60)
                        del self.manifest["images"][identifier]
                        self.save()
                    except Exception as exc:
                        errors.append(f"images:{identifier}:{exc}")
                self.discover()
                for kind in ("containers", "networks", "volumes", "images"):
                    if self.manifest[kind]:
                        errors.append(f"residual_{kind}")
                running = set(self._run([self.docker, "ps", "-q", "--no-trunc"]).stdout.split())
                self.check(set(self.manifest.get("existing_container_ids", [])) <= running, "previously_running_container_missing")
            except Exception as exc:
                errors.append(str(exc))
        self.report["cleanup"] = {"status": "failed" if errors else "passed", "errors": errors, "finished_at": utc_now()}
        self.save()
        return not errors

    def terminate_recorded_frontend(self):
        expected = self.manifest["frontend_process"]
        self._terminate_process_tree(expected)
        self.manifest.pop("frontend_process", None)


def main(argv=None):
    if str(REPO / "backend") not in sys.path:
        sys.path.insert(0, str(REPO / "backend"))
    if str(REPO / "backend" / "src") not in sys.path:
        sys.path.insert(0, str(REPO / "backend" / "src"))
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--allow-live", action="store_true")
    run.add_argument("--docker-executable")
    run.add_argument("--browser-channel", default="msedge" if os.name == "nt" else None)
    cleanup = sub.add_parser("cleanup")
    cleanup.add_argument("--run-id", required=True)
    cleanup.add_argument("--docker-executable")
    args = parser.parse_args(argv)
    if args.command == "run" and not args.allow_live:
        parser.error("真实模型验收必须显式传入 --allow-live")
    run_id = args.run_id if args.command == "cleanup" else datetime.now(UTC).strftime("%Y%m%d-%H%M%S-") + secrets.token_hex(3)
    directory = evidence_directory(run_id)
    if args.command == "run":
        directory.mkdir(parents=True, exist_ok=False)
    harness = Harness(run_id, directory)
    try:
        harness.docker = find_docker(args.docker_executable)
        if args.command == "cleanup":
            harness.manifest = json.loads((directory / "resources.json").read_text(encoding="utf-8"))
            harness.check(harness.manifest["run_id"] == run_id and harness.manifest["project"] == harness.project, "manifest_run_mismatch")
            harness.owner = harness.manifest["owner"]
            harness.report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
            cleaned = harness.cleanup()
            harness.report.update(status="aborted_cleaned" if cleaned else "cleanup_failed", finished_at=utc_now())
            harness.save()
            return 0 if cleaned else 1
        harness.save()
        with harness.scenario("preflight"):
            harness.preflight(args.browser_channel)
        with harness.scenario("initialize"):
            harness.initialize()
        from scripts.isolated_e2e_scenarios import run_projection_scenario, run_worker_scenarios
        run_worker_scenarios(harness)
        harness.run_browser()
        run_projection_scenario(harness)
        harness.report["status"] = "passed"
    except BaseException as exc:
        harness.report.update(status="failed", error=harness.redactor.text(str(exc))[:1600])
        harness.event(f"验收失败：{harness.report['error']}")
    finally:
        if args.command == "run":
            harness.deadline = None
            harness.diagnostics()
            if not harness.cleanup():
                harness.report["status"] = "cleanup_failed"
            harness.report["finished_at"] = utc_now()
            harness.save()
            harness.event(f"结果：{harness.report['status']}；证据：{directory}")
    return 0 if harness.report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
