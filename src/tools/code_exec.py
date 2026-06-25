from __future__ import annotations

import atexit
import html
import importlib.util
import os
import re
import sys
import threading
from typing import Any

DEFAULT_SANDBOX_TIMEOUT = 600
DEFAULT_E2B_TEMPLATE_ID = "1av7fdjfvcparqo8efq6"
MAX_RESULT_LEN = 20_000
MAX_ERROR_LEN = 4_000

E2B_IMPORT_NAME = "e2b_code_interpreter"
E2B_PACKAGE_NAME = "e2b-code-interpreter"
E2B_MISSING_MESSAGE = (
    f"Error: {E2B_PACKAGE_NAME} package is not installed in the OpenSeeker runtime. "
    f"Install it with: pip install {E2B_PACKAGE_NAME}"
)

DEFAULT_CODE_EXEC_DESCRIPTION = (
    "Execute a shell command in the shared execution environment for this task attempt.\n\n"
    "All calls in the same attempt reuse one sandbox, so files, installed packages, "
    "and working-directory state persist across calls. Do not create or pass sandbox ids. "
    "Use Python by running python commands or scripts, for example `python - <<'PY' ... PY`. "
    "Use this for calculations, symbolic math (sympy), numeric work (numpy/scipy), data "
    "processing, installing lightweight packages, writing/reading files, and verifying "
    "intermediate results.\n\n"
    "Args:\n"
    "    cmd: The shell command to execute in the shared sandbox.\n\n"
    "Returns:\n"
    "    XML containing exit_code, stdout, and stderr."
)

CODE_EXEC_TOOL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "code_exec",
        "description": DEFAULT_CODE_EXEC_DESCRIPTION,
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "cmd": {
                    "type": "string",
                    "description": "Shell command to execute in the shared sandbox.",
                },
            },
            "required": ["cmd"],
        },
    },
}

_STATUS_PREFIX_RE = re.compile(r"^\s*(\d{3})\b")
_LOCK = threading.Lock()
_SANDBOX: Any | None = None
_SANDBOX_ID: str | None = None


def is_e2b_available() -> bool:
    if E2B_IMPORT_NAME in sys.modules:
        return True
    return importlib.util.find_spec(E2B_IMPORT_NAME) is not None


def validate_code_exec_environment() -> list[str]:
    problems: list[str] = []
    if not os.environ.get("E2B_API_KEY"):
        problems.append("E2B_API_KEY is not set")
    if not is_e2b_available():
        problems.append(f"{E2B_PACKAGE_NAME} is not installed")
    return problems


def _truncate(text: str, limit: int) -> str:
    if len(text) > limit:
        return text[:limit] + " [truncated due to length limit]"
    return text


def _format_shell_result(result: Any) -> str:
    exit_code = getattr(result, "exit_code", None)
    if exit_code is None:
        exit_code = 0
    stdout = getattr(result, "stdout", "")
    stderr = getattr(result, "stderr", "")
    return (
        "<shell_results>\n"
        f"<exit_code>{html.escape(str(exit_code))}</exit_code>\n"
        f"<stdout>{html.escape(_truncate(str(stdout or ''), MAX_RESULT_LEN))}</stdout>\n"
        f"<stderr>{html.escape(_truncate(str(stderr or ''), MAX_ERROR_LEN))}</stderr>\n"
        "</shell_results>"
    )


def _load_sandbox_class() -> Any:
    try:
        from e2b_code_interpreter import Sandbox  # type: ignore[import-untyped]
    except ImportError as exc:
        raise RuntimeError(E2B_MISSING_MESSAGE) from exc
    return Sandbox


def _get_sandbox_id(sandbox: Any) -> str | None:
    sandbox_id = getattr(sandbox, "sandbox_id", None)
    if sandbox_id:
        return str(sandbox_id)
    try:
        info = sandbox.get_info()
    except Exception:
        return None
    sandbox_id = getattr(info, "sandbox_id", None) or getattr(info, "id", None)
    return str(sandbox_id) if sandbox_id else None


def _new_sandbox() -> Any:
    api_key = os.environ.get("E2B_API_KEY")
    if not api_key:
        raise RuntimeError("E2B_API_KEY is not set")
    Sandbox = _load_sandbox_class()
    create = getattr(Sandbox, "create", None)
    kwargs: dict[str, Any] = {"timeout": DEFAULT_SANDBOX_TIMEOUT, "api_key": api_key}
    if DEFAULT_E2B_TEMPLATE_ID:
        kwargs["template"] = DEFAULT_E2B_TEMPLATE_ID
    if callable(create):
        try:
            return create(**kwargs)
        except TypeError as exc:
            if "template" not in str(exc):
                raise
            kwargs.pop("template", None)
            return create(**kwargs)
    return Sandbox(**kwargs)


def _is_sandbox_not_found(exc: BaseException) -> bool:
    lowered = str(exc).lower()
    return "sandbox was not found" in lowered or "sandbox not found" in lowered


def _error_status(exc: BaseException) -> int | None:
    try:
        from e2b.exceptions import RateLimitException  # type: ignore[import-untyped]

        if isinstance(exc, RateLimitException):
            return 429
    except Exception:
        pass
    match = _STATUS_PREFIX_RE.match(str(exc))
    return int(match.group(1)) if match else None


def _get_or_create_sandbox() -> Any:
    global _SANDBOX, _SANDBOX_ID
    with _LOCK:
        if _SANDBOX is None:
            _SANDBOX = _new_sandbox()
            _SANDBOX_ID = _get_sandbox_id(_SANDBOX)
        try:
            _SANDBOX.set_timeout(DEFAULT_SANDBOX_TIMEOUT)
        except Exception:
            pass
        return _SANDBOX


def reset_code_exec_sandbox() -> None:
    global _SANDBOX, _SANDBOX_ID
    with _LOCK:
        sandbox = _SANDBOX
        _SANDBOX = None
        _SANDBOX_ID = None
    if sandbox is not None:
        try:
            sandbox.kill()
        except Exception:
            pass


atexit.register(reset_code_exec_sandbox)


def code_exec(cmd: str) -> str:
    if not isinstance(cmd, str) or not cmd.strip():
        return "[ERROR]: cmd must be a non-empty string."

    last_error: BaseException | None = None
    for attempt in range(1, 4):
        try:
            sandbox = _get_or_create_sandbox()
            result = sandbox.commands.run(cmd)
            return _format_shell_result(result)
        except Exception as exc:
            last_error = exc
            status = _error_status(exc)
            if _is_sandbox_not_found(exc):
                reset_code_exec_sandbox()
                continue
            if status in {408, 429} or (status is not None and status >= 500):
                continue
            break

    detail = _truncate(str(last_error or "unknown error"), MAX_ERROR_LEN)
    return (
        "<shell_results>\n"
        "<exit_code>1</exit_code>\n"
        "<stdout></stdout>\n"
        f"<stderr>{html.escape(detail)}</stderr>\n"
        "</shell_results>"
    )
