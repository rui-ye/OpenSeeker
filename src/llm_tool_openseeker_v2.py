

from __future__ import annotations

import argparse
import asyncio
import datetime
import json
import os
import random
import re
import sys
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

import jinja2
import requests
from pydantic import ConfigDict, ValidationError, validate_call
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)

from tools.search import Search
from tools.visit import Visit
from tools.e2b_sandbox_tools import (
    create_sandbox,
    download_file_from_internet_to_sandbox,
    run_command,
    run_python_code,
    upload_file_from_local_to_sandbox,
)


def print_colored(text: str, color: int) -> None:
    print(f"\033[{color}m{text}\033[0m", end="", flush=True)


def _tool_color(tool_name: str) -> int:
    if tool_name == "search":
        return 34  # blue
    if tool_name == "visit":
        return 33  # yellow
    if tool_name in (
        "create_sandbox",
        "run_command",
        "run_python_code",
        "upload_file_from_local_to_sandbox",
        "download_file_from_internet_to_sandbox",
    ):
        return 36  # cyan
    return 35  # magenta


def _truncate_text(s: Any, max_chars: int) -> str:
    ss = "" if s is None else str(s)
    if max_chars <= 0 or len(ss) <= max_chars:
        return ss
    head = max(0, int(max_chars * 0.7))
    tail = max_chars - head
    return f"{ss[:head]}...<truncated {len(ss)-max_chars} chars>...{ss[-tail:] if tail > 0 else ''}"


def _print_tool_call(tool_name: str, tool_args: Any, tool_response: str) -> None:
    TOOL_LOG_MAX_CHARS = 800
    max_chars = TOOL_LOG_MAX_CHARS
    c = _tool_color(tool_name)
    try:
        args_str = json.dumps(tool_args, ensure_ascii=False, sort_keys=True)
    except Exception:
        args_str = str(tool_args)
    args_str = _truncate_text(args_str, max_chars)
    resp_str = _truncate_text(tool_response, max_chars)
    print_colored(f"\n[{tool_name}] args={args_str}\n", c)
    print_colored(f"[{tool_name}] response={resp_str}\n", c)



template_file_path = os.path.join(_THIS_DIR, "config/chat_template.jinja")


def strftime_now_function(fmt: str) -> str:
    return datetime.datetime.now().strftime(fmt)


def _tojson(obj: Any) -> str:
    # Keep consistent encoding (no ascii escaping).
    return json.dumps(obj, ensure_ascii=False)


with open(template_file_path, "r", encoding="utf-8") as f:
    _template_string = f.read()

_env = jinja2.Environment()
_env.filters["tojson"] = _tojson
_env.globals["strftime_now"] = strftime_now_function
template = _env.from_string(_template_string)


# ----------------------------
# Tool schemas (OpenAI "tools" format)
# ----------------------------
developer_prompt = (
    "You are a tool-augmented QA agent. Cleverly leverage appropriate tools to answer the user's question."
)

search_description = (
    "Performs batched web searches: supply an array 'query'; the tool retrieves the top 10 results for each query in one call."
)

visit_description = "Parse webpage(s) and return the summary of the content according to the goal."


tools_visit: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": search_description,
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Array of query strings. Include multiple complementary search queries in a single call.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "visit",
            "description": visit_description,
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": ["string", "array"],
                        "items": {"type": "string"},
                        "minItems": 1,
                        "description": "The URL(s) of the webpage(s) to visit. Can be a single URL or an array of URLs.",
                    },
                    "goal": {"type": "string", "description": "The goal of the visit for webpage(s)."},
                },
                "required": ["url", "goal"],
            },
        },
    },
]

create_sandbox_description = (
    "Create a linux sandbox.\n\n"
    "Args:\n"
    "    timeout: Time in seconds before the sandbox is automatically shutdown. "
    "The default is 600 seconds.\n\n"
    "Returns:\n"
    "    The sandbox_id of the newly created sandbox. You should use this sandbox_id "
    "to run other tools in the sandbox."
)

run_command_description = (
    "Execute a lightweight shell command in the linux sandbox "
    "(no long-running, blocking, or resource-heavy processes).\n\n"
    "Args:\n"
    "    command: The command to execute.\n"
    "    sandbox_id: The id of the sandbox to execute the command in. "
    "To create a new sandbox, use tool `create_sandbox`.\n\n"
    "Returns:\n"
    "    A CommandResult object containing the result of the command execution, "
    "format like CommandResult(stderr=..., stdout=..., exit_code=..., error=...)"
)

run_python_code_description = (
    "Run short, safe python code in a sandbox and return the execution result "
    "(avoid long loops or heavy tasks; must finish quickly).\n\n"
    "Args:\n"
    "    code_block: The python code to run.\n"
    "    sandbox_id: The id of the sandbox to run the code in. "
    "Reuse existing sandboxes whenever possible. To create a new sandbox, use tool `create_sandbox`.\n\n"
    "Returns:\n"
    "    An Execution object containing the result of the Python code execution, "
    "format like Execution(Results: ..., Logs: Logs(stdout: ..., stderr: ...), Error: ...)"
)

upload_file_description = (
    "Upload a local file to the `/home/user` dir of the remote python interpreter.\n\n"
    "Args:\n"
    "    sandbox_id: The id of the sandbox to run the code in. "
    "Reuse existing sandboxes whenever possible. To create a new sandbox, use tool `create_sandbox`.\n"
    "    local_file_path: The path of the file on local machine to upload.\n"
    "    sandbox_file_path: The path of directory to upload the file to in the sandbox. "
    "Default is `/home/user/`.\n\n"
    "Returns:\n"
    "    The path of the uploaded file in the remote python interpreter if the upload is successful."
)

download_file_description = (
    "Download a file from the internet to the `/home/user` dir of the sandbox "
    "(avoid large or slow URLs).\n\n"
    "Args:\n"
    "    sandbox_id: The id of the sandbox to run the code in. "
    "Reuse existing sandboxes whenever possible. To create a new sandbox, use tool `create_sandbox`.\n"
    "    url: The URL of the file to download.\n"
    "    sandbox_file_path: The path of directory to download the file to in the sandbox. "
    "Default is `/home/user/`.\n\n"
    "Returns:\n"
    "    The path of the downloaded file in the sandbox if the download is successful."
)

tools_e2b: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "create_sandbox",
            "description": create_sandbox_description,
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "timeout": {
                        "type": "integer",
                        "default": 600,
                        "description": (
                            "Time in seconds before the sandbox is automatically shutdown. "
                            "The default is 600 seconds."
                        ),
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": run_command_description,
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The command to execute.",
                    },
                    "sandbox_id": {
                        "type": "string",
                        "description": (
                            "The id of the sandbox to execute the command in. "
                            "To create a new sandbox, use tool `create_sandbox`."
                        ),
                    },
                },
                "required": ["command", "sandbox_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_python_code",
            "description": run_python_code_description,
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "code_block": {
                        "type": "string",
                        "description": "The python code to run.",
                    },
                    "sandbox_id": {
                        "type": "string",
                        "description": (
                            "The id of the sandbox to run the code in. "
                            "Reuse existing sandboxes whenever possible. "
                            "To create a new sandbox, use tool `create_sandbox`."
                        ),
                    },
                },
                "required": ["code_block", "sandbox_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "upload_file_from_local_to_sandbox",
            "description": upload_file_description,
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "sandbox_id": {
                        "type": "string",
                        "description": (
                            "The id of the sandbox to run the code in. "
                            "Reuse existing sandboxes whenever possible. "
                            "To create a new sandbox, use tool `create_sandbox`."
                        ),
                    },
                    "local_file_path": {
                        "type": "string",
                        "description": "The path of the file on local machine to upload.",
                    },
                    "sandbox_file_path": {
                        "type": "string",
                        "default": "/home/user",
                        "description": (
                            "The path of directory to upload the file to in the sandbox. "
                            "Default is `/home/user/`."
                        ),
                    },
                },
                "required": ["sandbox_id", "local_file_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "download_file_from_internet_to_sandbox",
            "description": download_file_description,
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "sandbox_id": {
                        "type": "string",
                        "description": (
                            "The id of the sandbox to run the code in. "
                            "Reuse existing sandboxes whenever possible. "
                            "To create a new sandbox, use tool `create_sandbox`."
                        ),
                    },
                    "url": {
                        "type": "string",
                        "description": "The URL of the file to download.",
                    },
                    "sandbox_file_path": {
                        "type": "string",
                        "default": "/home/user",
                        "description": (
                            "The path of directory to download the file to in the sandbox. "
                            "Default is `/home/user/`."
                        ),
                    },
                },
                "required": ["sandbox_id", "url"],
            },
        },
    },
]

tools_all: List[Dict[str, Any]] = tools_visit + tools_e2b



# ----------------------------
# Streaming completion (OpenAI-compatible /v1/completions)
# ----------------------------
def _normalize_completions_url(base_or_full: str) -> str:
    s = (base_or_full or "").strip()
    if not s:
        raise ValueError("Empty base_url / completions_url")
    if s.endswith("/completions"):
        return s
    if s.endswith("/v1"):
        return s + "/completions"
    if s.endswith("/v1/"):
        return s + "completions"
    if s.endswith("/"):
        return s + "v1/completions"
    return s + "/v1/completions"


def get_stream_response(
    completions_url: str,
    payload: Dict[str, Any],
    print_stream: bool = True,
    *,
    max_retries: int = 3,
    connect_timeout: int = 10,
    read_timeout: int = 60,
    max_total_seconds: int = 1200,
    max_idle_seconds: int = 45,
) -> Tuple[str, bool]:
    """
    Stream text from an OpenAI-compatible /v1/completions SSE endpoint.

    Returns:
      (full_text, too_many_tokens_error)
    """
    retry_backoff = 1.8
    retry_jitter = 0.4
    too_many_tokens_error = False
    result = ""

    for attempt in range(int(max_retries) + 1):
        start = time.monotonic()
        last = start
        chunks: List[str] = []
        got_done = False
        failed = False

        try:
            with requests.post(
                completions_url,
                json=payload,
                stream=True,
                timeout=(int(connect_timeout), int(read_timeout)),
            ) as response:
                response.raise_for_status()

                full_text_so_far = ""
                for raw_line in response.iter_lines(decode_unicode=False):
                    now = time.monotonic()
                    if now - start > float(max_total_seconds):
                        failed = True
                        break
                    if now - last > float(max_idle_seconds):
                        failed = True
                        break

                    if not raw_line:
                        continue
                    line = raw_line.decode("utf-8", errors="replace")
                    if not line.startswith("data:"):
                        continue

                    last = now
                    json_str = line.split("data:", 1)[1].strip()
                    if not json_str:
                        continue
                    if json_str == "[DONE]":
                        got_done = not failed
                        break

                    try:
                        data = json.loads(json_str)
                    except json.JSONDecodeError:
                        continue

                    if isinstance(data, dict) and "error" in data:
                        err = data.get("error") or {}
                        message = (err.get("message") or "").strip()
                        if "maximum context length" in message or "context length" in message:
                            too_many_tokens_error = True
                        if message:
                            print(f"[LLM ERROR] {message}")
                        failed = True
                        break

                    try:
                        choice0 = (data.get("choices") or [{}])[0] or {}
                    except Exception:
                        choice0 = {}

                    text_piece = choice0.get("text") or ""
                    if isinstance(text_piece, str) and text_piece:
                        if text_piece.startswith(full_text_so_far):
                            delta = text_piece[len(full_text_so_far) :]
                            full_text_so_far = text_piece
                        else:
                            delta = text_piece
                            full_text_so_far += text_piece
                        if delta:
                            chunks.append(delta)
                            if print_stream:
                                print(delta, end="", flush=True)

                    if choice0.get("finish_reason") or choice0.get("matched_stop"):
                        got_done = True
                        break

        except requests.exceptions.RequestException as e:
            print(f"[LLM ERROR] Request failed: {type(e).__name__}: {e}")
            failed = True

        result = "".join(chunks)
        if got_done and not failed:
            return result, too_many_tokens_error
        if too_many_tokens_error:
            return result, too_many_tokens_error
        if attempt < int(max_retries):
            delay = min(10.0, (retry_backoff**attempt)) + random.uniform(0, retry_jitter)
            time.sleep(delay)
            continue
        return result, too_many_tokens_error

    return result, too_many_tokens_error


# ----------------------------
# Tool-call parsing (Qwen3 template contract)
# ----------------------------
_TOOL_CALLS_BLOCK_RE = re.compile(r"<tool_calls_begin>\s*(.*?)\s*</tool_calls_end>", re.DOTALL)
_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)



def _try_fix_incomplete_json(json_str: str) -> str:
    if not json_str or not str(json_str).strip():
        return json_str

    s = str(json_str).strip()

    s = re.sub(r'"\s+"', '", "', s)

    s = re.sub(r'"\s*\[', '", [', s)
    s = re.sub(r'\]\s*"', '], "', s)
    s = re.sub(r'\}\s*\{', '}, {', s)

    s = re.sub(r',\s*([\]}])', r'\1', s)

    s = s.rstrip()
    missing_brackets = s.count("[") - s.count("]")
    missing_braces = s.count("{") - s.count("}")

    if missing_brackets > 0:
        s += "]" * missing_brackets
    if missing_braces > 0:
        s += "}" * missing_braces

    return s



def _parse_tool_calls_from_text(text: str) -> Tuple[str, List[Dict[str, Any]], Optional[str]]:
    if not text:
        error_msg = "ERROR: No text to parse tool calls"
        print(f"\033[91m{error_msg}\033[0m")
        return "", [], error_msg

    tool_calls: List[Dict[str, Any]] = []
    errors: List[str] = []
    
    blocks = [m.group(1) for m in _TOOL_CALLS_BLOCK_RE.finditer(text)]
    if blocks and all(not _TOOL_CALL_RE.search(b) for b in blocks):
        errors.append(
            "ERROR: Found <tool_calls_begin> block but no <tool_call> inside. "
            "I must either call a tool with <tool_call>...</tool_call> or "
            "provide my final answer directly as plain text WITHOUT any tool_calls tags."
        )
    scan_targets = blocks if blocks else [text]

    for chunk in scan_targets:
        for m in _TOOL_CALL_RE.finditer(chunk):
            inner = (m.group(1) or "").strip()
            if not inner:
                errors.append("ERROR: Found empty <tool_call> tag (inner content is empty)")
                continue
            try:
                obj = json.loads(inner)
            except json.JSONDecodeError as e:
                fixed = _try_fix_incomplete_json(inner)
                try:
                    obj = json.loads(fixed)
                except Exception as e2:
                    error_msg = "ERROR: JSON decode failed even after fix attempt"
                    print(f"\033[91m{error_msg},{e2}\033[0m")
                    errors.append(error_msg)
                    continue
            except Exception as e:
                error_msg = f"ERROR: Unexpected exception during JSON parsing"
                print(f"\033[91m{error_msg},{e}\033[0m")
                errors.append(error_msg)
                continue

            def _append_one_tool_call(obj_dict: Dict[str, Any]) -> None:
                name = obj_dict.get("tool_name") or obj_dict.get("name")
                arguments = obj_dict.get("tool_args") or obj_dict.get("arguments")
                if not isinstance(name, str) or not name.strip():
                    errors.append("ERROR: Tool call missing or empty name field")
                    return
                name = name.strip()
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except Exception as e:
                        errors.append("ERROR: Failed to parse arguments as JSON string")
                        arguments = {}
                if not isinstance(arguments, dict):
                    errors.append("ERROR: Arguments is not a dict")
                    arguments = {}
                tool_calls.append({"function": {"name": name, "arguments": arguments}})

            if isinstance(obj, dict):
                _append_one_tool_call(obj)
            elif isinstance(obj, list):
                for item in obj:
                    if isinstance(item, dict):
                        _append_one_tool_call(item)
                    else:
                        errors.append("ERROR: List item is not a dict")
            else:
                error_msg = "ERROR: Parsed object is neither dict nor list"
                print(f"\033[91m{error_msg}\033[0m")
                errors.append(error_msg)
                continue

    cleaned = _TOOL_CALLS_BLOCK_RE.sub("", text)
    
    return cleaned, tool_calls, "\n".join(errors)




def _has_answer_tag(text: str) -> bool:
    if not text:
        return False
    return "</answer>" in text


def _split_think_and_content(completion_text: str) -> Tuple[str, str]:
    s = (completion_text or "")
    if not s:
        return "", ""

    s2 = s.lstrip()
    if s2.startswith("<think>"):
        s2 = s2[len("<think>") :]
        s = s2

    if "</think>" in s:
        reasoning, rest = s.split("</think>", 1)
        return reasoning.strip(), rest.lstrip("\n").lstrip()
    return "", s.strip()


@validate_call(config=ConfigDict(extra="forbid"))
def _validate_create_sandbox_args(timeout: int = 600) -> None:
    return None


@validate_call(config=ConfigDict(extra="forbid"))
def _validate_run_command_args(command: str, sandbox_id: str) -> None:
    return None


@validate_call(config=ConfigDict(extra="forbid"))
def _validate_run_python_code_args(code_block: str, sandbox_id: str) -> None:
    return None


@validate_call(config=ConfigDict(extra="forbid"))
def _validate_upload_file_args(
    sandbox_id: str,
    local_file_path: str,
    sandbox_file_path: str = "/home/user",
) -> None:
    return None


@validate_call(config=ConfigDict(extra="forbid"))
def _validate_download_file_args(
    sandbox_id: str,
    url: str,
    sandbox_file_path: str = "/home/user",
) -> None:
    return None


_E2B_ARG_VALIDATORS = {
    "create_sandbox": _validate_create_sandbox_args,
    "run_command": _validate_run_command_args,
    "run_python_code": _validate_run_python_code_args,
    "upload_file_from_local_to_sandbox": _validate_upload_file_args,
    "download_file_from_internet_to_sandbox": _validate_download_file_args,
}


def _format_validation_error(tool_name: str, err: ValidationError) -> str:
    lines = str(err).splitlines()
    if lines:
        m = re.match(r"^(\d+) validation error(s)? for ", lines[0])
        if m:
            count = m.group(1)
            label = "error" if count == "1" else "errors"
            lines[0] = f"{count} validation {label} for call[{tool_name}]"
    return "\n".join(lines)


def _validate_e2b_tool_args(tool_name: str, tool_args: Dict[str, Any]) -> Optional[str]:
    validator = _E2B_ARG_VALIDATORS.get(tool_name)
    if validator is None:
        return None
    try:
        validator(**tool_args)
    except ValidationError as e:
        return _format_validation_error(tool_name, e)
    return None


def _execute_tool(
    tool_name: str,
    tool_args: Dict[str, Any],
    *,
    filter_huggingface: bool = False,
) -> str:
    if tool_name == "search":
        return Search({"filter_huggingface": filter_huggingface}).call(tool_args)
    if tool_name in ("visit", "visit_summary"):
        return Visit().call(tool_args)
    validation_error = _validate_e2b_tool_args(tool_name, tool_args)
    if validation_error:
        return validation_error
    if tool_name == "create_sandbox":
        timeout = tool_args.get("timeout", 600)
        try:
            timeout_int = int(timeout)
        except (TypeError, ValueError):
            timeout_int = 600
        return asyncio.run(create_sandbox(timeout=timeout_int))
    if tool_name == "run_command":
        cmd = str(tool_args.get("command") or "")
        sid = str(tool_args.get("sandbox_id") or "")
        return asyncio.run(run_command(cmd, sid))
    if tool_name == "run_python_code":
        code = str(tool_args.get("code_block") or "")
        sid = str(tool_args.get("sandbox_id") or "")
        return asyncio.run(run_python_code(code, sid))
    if tool_name == "upload_file_from_local_to_sandbox":
        sid = str(tool_args.get("sandbox_id") or "")
        local_path = str(tool_args.get("local_file_path") or "")
        remote = tool_args.get("sandbox_file_path", "/home/user")
        remote_s = "/home/user" if remote is None else str(remote)
        return asyncio.run(upload_file_from_local_to_sandbox(sid, local_path, remote_s))
    if tool_name == "download_file_from_internet_to_sandbox":
        sid = str(tool_args.get("sandbox_id") or "")
        url = str(tool_args.get("url") or "")
        remote = tool_args.get("sandbox_file_path", "/home/user")
        remote_s = "/home/user" if remote is None else str(remote)
        return asyncio.run(download_file_from_internet_to_sandbox(sid, url, remote_s))
    return "Unknown tool or call tool with incorrect format."


def _render_prompt(messages: List[Dict[str, Any]], tools: List[Dict[str, Any]], *, add_generation_prompt: bool) -> str:
    return template.render(messages=messages, tools=tools, add_generation_prompt=add_generation_prompt)




def call_llm_with_tool(item: Dict[str, Any], args, *, return_metrics: bool = False, return_trace: bool = False):
    query = item["query"]
    
    base_url = os.getenv("OPENSEEKER_BASE_URL", "YOUR_OPENSEEKER_BASE_URL")
    if base_url == "YOUR_OPENSEEKER_BASE_URL":
        raise ValueError("OPENSEEKER_BASE_URL environment variable is required")
    
    completions_url = _normalize_completions_url(base_url)
    model_name = os.getenv("OPENSEEKER_MODEL", "YOUR_MODEL_NAME")
    if model_name == "YOUR_MODEL_NAME":
        raise ValueError("OPENSEEKER_MODEL environment variable is required")

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": developer_prompt},
        {"role": "user", "content": query},
    ]

    trace: List[Dict[str, Any]] = []
    step_num = 0
    tool_count = 0
    tool_count_max = int(getattr(args, "tool_count_max", 200))
    local_max_tokens = int(getattr(args, "max_tokens", 16384))

    _GEN_PROMPT = "<|im_start|>assistant\n<think>\n"
    disable_tools = False  
    pending_assistant_prefix = ""
    _final_answer_retries = 0
    _MAX_FINAL_ANSWER_RETRIES = 3
    finish_reason = "unknown"
    error_info: Optional[Dict[str, Any]] = None
    while True:
        tools_for_render = [] if disable_tools else tools_all
        injected_prefix = pending_assistant_prefix
        if pending_assistant_prefix:
            prompt_text = (
                _render_prompt(messages, tools_for_render, add_generation_prompt=False)
                + _GEN_PROMPT
                + injected_prefix
            )
        else:
            prompt_text = _render_prompt(messages, tools_for_render, add_generation_prompt=True)

        pending_assistant_prefix = ""
        
        payload = {
            "model": model_name,
            "prompt": prompt_text,
            "max_tokens": local_max_tokens,
            "stream": True,
            "skip_special_tokens": False,
        }

        completion_text, too_many_tokens_error = get_stream_response(
            completions_url,
            payload,
            print_stream=bool(getattr(args, "print_stream", False)),
        )

        if not completion_text and not too_many_tokens_error:
            raise RuntimeError(f"Empty response from LLM. completions_url={completions_url}")

        if too_many_tokens_error:
            new_local_max = int(local_max_tokens * 0.9)
            if new_local_max > 2048 and new_local_max < local_max_tokens:
                local_max_tokens = new_local_max
                continue
            elif 128 < new_local_max <= 2048 and new_local_max < local_max_tokens:
                if not pending_assistant_prefix:
                    pending_assistant_prefix = (
                        "I have used too many tokens, so I will conclude my answer.\n"
                    )
                disable_tools = True
                local_max_tokens = new_local_max
                continue
            else:
                completion_text = _GEN_PROMPT + "</think>\n\n\n<answer>\nThe max context length has been reached.</answer><|im_end|>\n"
                too_many_tokens_error = False

        if injected_prefix:
            completion_text = _GEN_PROMPT + injected_prefix + completion_text
        step_num += 1

        reasoning_content, content_raw = _split_think_and_content(completion_text.replace("<|im_end|>", "").replace(_GEN_PROMPT, ""))
        
        has_answer = _has_answer_tag(content_raw)
        
        cleaned_text, tool_calls, parse_error = _parse_tool_calls_from_text(content_raw)
        if disable_tools and tool_calls:
            tool_calls = []
       
        assistant_msg: Dict[str, Any] = {"role": "assistant", "content": completion_text.replace("<|im_end|>", "").replace(_GEN_PROMPT, "")}
        messages.append(assistant_msg)
        if return_trace:
            trace.append(
                {
                    "step": step_num,
                    "type": "model_message",
                    "content": {
                        "reasoning_content": reasoning_content,
                        "content": cleaned_text,
                        "tool_calls": tool_calls,
                    },
                }
            )
        
        if has_answer:
            finish_reason = "answer"
            break
        
        if not tool_calls:
            tool_count += 1
            if tool_count >= tool_count_max:
                finish_reason = "tool_count_limit_no_tool_or_answer"
                error_info = {
                    "type": "NoToolCallOrAnswerLimit",
                    "message": (
                        "The model repeatedly produced neither a final answer nor a valid tool call; "
                        f"stopped after reaching tool_count_max={tool_count_max}."
                    ),
                }
                break
            if disable_tools:
                _final_answer_retries += 1
                if _final_answer_retries >= _MAX_FINAL_ANSWER_RETRIES:
                    finish_reason = "no_final_answer_after_tools_disabled"
                    error_info = {
                        "type": "NoFinalAnswerAfterToolsDisabled",
                        "message": "Tools were disabled, but the model still did not output a final answer.",
                    }
                    break
                pending_assistant_prefix = (
                    "Tools are disabled. Do NOT output any <tool_calls_begin> tags. "
                    "Provide my final answer as plain text immediately."
                )
                messages.pop()
                continue
            error_content = parse_error if parse_error else "ERROR: Tool call parsing failed"
            print(f"\033[91m {error_content}\033[0m")
            messages.append({"role": "tool",
                "name": "unknown",
                "content": error_content,
                "tool_call_id": str(uuid.uuid4())})
            
            if return_trace:
                trace.append(
                    {
                        "step": step_num,
                        "type": "tool_response",
                        "content": {
                            "content": error_content,
                            "counted_as_tool_call": True,
                            "tool_count": tool_count,
                        },
                    }
                )
            continue

        for tc in tool_calls:
            fn = (tc or {}).get("function") or {}
            tool_name = (fn.get("name") or "").strip()
            tool_args = fn.get("arguments") or {}
            if not isinstance(tool_args, dict):
                tool_args = {}

            try:
                tool_output = _execute_tool(
                    tool_name,
                    tool_args,
                    filter_huggingface=bool(getattr(args, "filter_huggingface", False)),
                )
            except Exception as e:
                tool_output = f"Error during tool execution: {type(e).__name__}: {e}"

            try:
                _print_tool_call(tool_name or "unknown", tool_args, tool_output)
            except Exception:
                pass
            tool_count += 1
            tool_msg = {
                "role": "tool",
                "name": tool_name,
                "content": tool_output,
                "tool_call_id": str(uuid.uuid4()),
            }
            messages.append(tool_msg)

            if return_trace:
                trace.append(
                    {
                        "step": step_num,
                        "type": "tool_call",
                        "content": {"tool_name": tool_name, "tool_args": tool_args},
                    }
                )
                trace.append(
                    {
                        "step": step_num,
                        "type": "tool_response",
                        "content": {"tool_name": tool_name, "tool_response": tool_output},
                    }
                )

            
            if tool_count >= tool_count_max:
                if not pending_assistant_prefix:
                    pending_assistant_prefix = (
                        "I have reached the tool usage limit. "
                        "Do NOT call any more tools. Do NOT output <tool_calls_begin> tags. "
                        "Provide my final answer as plain text NOW."
                    )
                disable_tools = True
                break

    full_traj = _render_prompt(messages, tools_for_render, add_generation_prompt=False)
    metrics = {
        "tool_calls": tool_count,
        "context_chars": len(full_traj),
        "finish_reason": finish_reason,
    }
    if error_info is not None:
        metrics["error"] = error_info

    if return_metrics:
        if return_trace:
            return full_traj, metrics, trace
        return full_traj, metrics
    if return_trace:
        return full_traj, trace
    return full_traj


def _get_last_assistant_answer_from_messages(full_traj: str) -> str:
    if not full_traj:
        return ""
    parts = full_traj.split("<|im_start|>assistant")
    if len(parts) < 2:
        return full_traj.strip()
    last = parts[-1]
    if "<|im_end|>" in last:
        last = last.split("<|im_end|>", 1)[0]
    last = _TOOL_CALLS_BLOCK_RE.sub("", last)
    last = _TOOL_CALL_RE.sub("", last)
    last = re.sub(r"<tool_response>.*?</tool_response>", "", last, flags=re.DOTALL)
    if "</think>" in last:
        last = last.split("</think>", 1)[-1]
    return last.strip()


def _estimate_tokens_from_chars(n_chars: int) -> int:
    return max(1, int(n_chars / 4))


def solve_query_with_tools(
    query: str,
    *,
    max_tokens: int = 16384,
    tool_count_max: int = 200,
    print_stream: bool = False,
    filter_huggingface: bool = False,
    return_full_traj: bool = True,
    return_trace: bool = True,
) -> Dict[str, Any]:

    start = time.time()
    args = argparse.Namespace(
        max_tokens=int(max_tokens),
        tool_count_max=int(tool_count_max),
        print_stream=bool(print_stream),
        filter_huggingface=bool(filter_huggingface),
    )

    item = {"query": query}
    call_result = call_llm_with_tool(item, args, return_metrics=True, return_trace=return_trace)
    if return_trace:
        full_traj, metrics, trace = call_result
    else:
        full_traj, metrics = call_result
        trace = []

    elapsed = time.time() - start
    answer = _get_last_assistant_answer_from_messages(full_traj)
    context_chars = int(metrics.get("context_chars", len(full_traj)))
    tool_calls = int(metrics.get("tool_calls", 0))

    result: Dict[str, Any] = {
        "answer": answer,
        "tool_calls": tool_calls,
        "elapsed_seconds": elapsed,
        "context_chars": context_chars,
        "context_est_tokens": _estimate_tokens_from_chars(context_chars),
        "finish_reason": metrics.get("finish_reason", "unknown"),
    }
    if metrics.get("error") is not None:
        result["error"] = metrics.get("error")
        result["answer"] = ""
    if return_full_traj:
        result["full_traj"] = full_traj
    if return_trace:
        result["trace"] = trace
    return result


if __name__ == "__main__":
    q = os.environ.get("OPENSEEKER_QUERY", "what's openai's leader")
    res = solve_query_with_tools(q, print_stream=True,tool_count_max=200)
    result_dir = os.path.join(_THIS_DIR, "../..", "result/test")
    os.makedirs(result_dir, exist_ok=True)
    result_path = os.path.join(result_dir, "res2.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False)
    print("\n\n=== ANSWER ===\n")
    print(res["answer"])
