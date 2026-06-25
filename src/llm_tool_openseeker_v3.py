from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)

from tools.code_exec import CODE_EXEC_TOOL_SCHEMA, code_exec
from tools.search import Search
from tools.visit import Visit


def print_colored(text: str, color: int) -> None:
    print(f"\033[{color}m{text}\033[0m", end="", flush=True)


def _tool_color(tool_name: str) -> int:
    if tool_name == "search":
        return 34
    if tool_name in ("visit", "visit_summary"):
        return 33
    if tool_name == "code_exec":
        return 36
    return 35


def _truncate_text(s: Any, max_chars: int) -> str:
    ss = "" if s is None else str(s)
    if max_chars <= 0 or len(ss) <= max_chars:
        return ss
    head = max(0, int(max_chars * 0.7))
    tail = max_chars - head
    return f"{ss[:head]}...<truncated {len(ss)-max_chars} chars>...{ss[-tail:] if tail > 0 else ''}"


def _print_tool_call(tool_name: str, tool_args: Any, tool_response: str) -> None:
    max_chars = 800
    c = _tool_color(tool_name)
    try:
        args_str = json.dumps(tool_args, ensure_ascii=False, sort_keys=True)
    except Exception:
        args_str = str(tool_args)
    print_colored(f"\n[{tool_name}] args={_truncate_text(args_str, max_chars)}\n", c)
    print_colored(f"[{tool_name}] response={_truncate_text(tool_response, max_chars)}\n", c)


developer_prompt = (
    "You are a tool-augmented QA agent. Cleverly leverage appropriate tools to answer the user's question. "
    "When the answer is ready, provide the final answer directly and do not call more tools."
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

tools_all: List[Dict[str, Any]] = tools_visit + [CODE_EXEC_TOOL_SCHEMA]


def _normalize_base_url(base_or_full: str) -> str:
    s = (base_or_full or "").strip()
    if not s:
        raise ValueError("Empty base_url")
    if s.endswith("/chat/completions"):
        s = s[: -len("/chat/completions")]
    if s.endswith("/completions"):
        s = s[: -len("/completions")]
    if s.endswith("/v1"):
        return s
    if s.endswith("/v1/"):
        return s[:-1]
    return s.rstrip("/") + "/v1"


def _parse_tool_arguments(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if raw is None:
        return {}
    if isinstance(raw, str):
        try:
            obj = json.loads(raw or "{}")
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}
    return {}


def _message_to_dict(message: Any) -> Dict[str, Any]:
    if hasattr(message, "model_dump"):
        dumped = message.model_dump(exclude_none=True)
    elif isinstance(message, dict):
        dumped = {k: v for k, v in message.items() if v is not None}
    else:
        dumped = {
            "role": getattr(message, "role", "assistant"),
            "content": getattr(message, "content", None),
        }
        tool_calls = getattr(message, "tool_calls", None)
        if tool_calls:
            dumped["tool_calls"] = [
                tc.model_dump(exclude_none=True) if hasattr(tc, "model_dump") else tc for tc in tool_calls
            ]
    dumped.setdefault("role", "assistant")
    if "content" not in dumped:
        dumped["content"] = ""
    return dumped


def _extract_tool_calls(message: Dict[str, Any]) -> List[Dict[str, Any]]:
    tool_calls = message.get("tool_calls") or []
    out: List[Dict[str, Any]] = []
    for idx, tc in enumerate(tool_calls):
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function") or {}
        if not isinstance(fn, dict):
            continue
        name = (fn.get("name") or "").strip()
        if not name:
            continue
        out.append(
            {
                "id": tc.get("id") or f"call_{uuid.uuid4().hex}_{idx}",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": fn.get("arguments") or "{}",
                },
            }
        )
    return out


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
    if tool_name == "code_exec":
        return code_exec(str(tool_args.get("cmd") or ""))
    return "Unknown tool or call tool with incorrect format."


def _build_client() -> Tuple[OpenAI, str]:
    base_url = os.getenv("OPENSEEKER_BASE_URL", "YOUR_OPENSEEKER_BASE_URL")
    if base_url == "YOUR_OPENSEEKER_BASE_URL":
        raise ValueError("OPENSEEKER_BASE_URL environment variable is required")
    model_name = os.getenv("OPENSEEKER_MODEL", "YOUR_MODEL_NAME")
    if model_name == "YOUR_MODEL_NAME":
        raise ValueError("OPENSEEKER_MODEL environment variable is required")
    api_key = os.getenv("OPENSEEKER_API_KEY") or os.getenv("OPENAI_API_KEY") or "EMPTY"
    client = OpenAI(api_key=api_key, base_url=_normalize_base_url(base_url))
    return client, model_name


def _extra_body_from_env() -> Dict[str, Any] | None:
    raw = os.getenv("OPENSEEKER_REQUEST_EXTRA_BODY_JSON", "").strip()
    if not raw:
        return None
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid OPENSEEKER_REQUEST_EXTRA_BODY_JSON: {exc}") from exc
    if not isinstance(obj, dict):
        raise ValueError("OPENSEEKER_REQUEST_EXTRA_BODY_JSON must be a JSON object")
    return obj


def _call_chat_completion(
    client: OpenAI,
    *,
    model_name: str,
    messages: List[Dict[str, Any]],
    tools: List[Dict[str, Any]],
    max_tokens: int,
    print_stream: bool,
) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "model": model_name,
        "messages": messages,
        "max_tokens": max_tokens,
    }
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"
    extra_body = _extra_body_from_env()
    if extra_body:
        kwargs["extra_body"] = extra_body
    response = client.chat.completions.create(**kwargs)
    message = _message_to_dict(response.choices[0].message)
    if print_stream and message.get("content"):
        print(message["content"], end="", flush=True)
    return message


def _messages_to_text(messages: List[Dict[str, Any]]) -> str:
    chunks: List[str] = []
    for msg in messages:
        role = msg.get("role", "")
        chunks.append(f"<|im_start|>{role}\n")
        content = msg.get("content")
        if content:
            chunks.append(str(content))
        for tc in msg.get("tool_calls") or []:
            chunks.append("\n<tool_call>\n")
            chunks.append(json.dumps(tc, ensure_ascii=False))
            chunks.append("\n</tool_call>")
        chunks.append("<|im_end|>\n")
    return "".join(chunks)


def call_llm_with_tool(item: Dict[str, Any], args, *, return_metrics: bool = False, return_trace: bool = False):
    query = item["query"]
    client, model_name = _build_client()
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": developer_prompt},
        {"role": "user", "content": query},
    ]

    trace: List[Dict[str, Any]] = []
    step_num = 0
    tool_count = 0
    tool_count_max = int(getattr(args, "tool_count_max", 200))
    max_tokens = int(getattr(args, "max_tokens", 16384))
    print_stream = bool(getattr(args, "print_stream", False))
    filter_huggingface = bool(getattr(args, "filter_huggingface", False))
    finish_reason = "unknown"
    error_info: Optional[Dict[str, Any]] = None

    while True:
        tools_for_call = tools_all if tool_count < tool_count_max else []
        message = _call_chat_completion(
            client,
            model_name=model_name,
            messages=messages,
            tools=tools_for_call,
            max_tokens=max_tokens,
            print_stream=print_stream,
        )
        step_num += 1
        tool_calls = _extract_tool_calls(message)
        if tool_calls:
            message["tool_calls"] = tool_calls
        messages.append(message)

        if return_trace:
            trace.append(
                {
                    "step": step_num,
                    "type": "model_message",
                    "content": {
                        "reasoning_content": message.get("reasoning_content") or message.get("reasoning") or "",
                        "content": message.get("content") or "",
                        "tool_calls": [
                            {"function": tc.get("function", {})} for tc in tool_calls
                        ],
                    },
                }
            )

        if not tool_calls:
            finish_reason = "answer"
            break

        for tc in tool_calls:
            fn = tc.get("function") or {}
            tool_name = (fn.get("name") or "").strip()
            tool_args = _parse_tool_arguments(fn.get("arguments"))
            try:
                tool_output = _execute_tool(
                    tool_name,
                    tool_args,
                    filter_huggingface=filter_huggingface,
                )
            except Exception as exc:
                tool_output = f"Error during tool execution: {type(exc).__name__}: {exc}"

            try:
                _print_tool_call(tool_name or "unknown", tool_args, tool_output)
            except Exception:
                pass

            tool_count += 1
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.get("id") or str(uuid.uuid4()),
                    "name": tool_name,
                    "content": tool_output,
                }
            )
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
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "You have reached the tool usage limit. Do not call any more tools. "
                            "Provide the final answer now."
                        ),
                    }
                )
                break

        if tool_count >= tool_count_max:
            continue

    full_traj = _messages_to_text(messages)
    metrics: Dict[str, Any] = {
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
    res = solve_query_with_tools(q, print_stream=True, tool_count_max=200)
    result_dir = os.path.join(_THIS_DIR, "../..", "result/test")
    os.makedirs(result_dir, exist_ok=True)
    result_path = os.path.join(result_dir, "res3.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False)
    print("\n\n=== ANSWER ===\n")
    print(res["answer"])
