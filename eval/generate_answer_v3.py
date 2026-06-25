#!/usr/bin/env python3
# coding: utf-8


from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple


BOX_FORMAT_INSTRUCTION = (
    "\n\nYou should follow the format instruction in the request strictly "
    "and wrap the final answer in \\\\boxed{}."
)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    data: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def append_box_instruction(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    boxed_data: List[Dict[str, Any]] = []
    for item in data:
        boxed_item = dict(item)
        query = boxed_item.get("query", "")
        if isinstance(query, str) and BOX_FORMAT_INSTRUCTION not in query:
            boxed_item["query"] = query + BOX_FORMAT_INSTRUCTION
        boxed_data.append(boxed_item)
    return boxed_data




def get_queries_without_answer(
    save_path: Path, all_queries: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Check which queries in all_queries don't have valid answers in save_path.
    A query is considered to have an answer if:
    - It exists in save_path AND
    - It has a 'final_response' field that is non-empty (after stripping)

    Returns:
        List of query dicts that don't have valid answers
    """
    if not save_path.exists():
        return all_queries

    query_to_has_answer: Dict[str, bool] = {}
    with save_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                q = obj.get("query")
                if isinstance(q, str) and q:
                    final_response = obj.get("final_response", "")
                    has_answer = bool(final_response and str(final_response).strip())
                    query_to_has_answer[q] = has_answer
            except Exception:
                continue

    missing = []
    for item in all_queries:
        q = item.get("query", "")
        if isinstance(q, str) and q:
            if q not in query_to_has_answer or not query_to_has_answer[q]:
                missing.append(item)

    return missing


def _safe_float(x: Any):
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def compute_metrics(result_jsonl: Path) -> Dict[str, Any]:
    """
    Compute aggregate metrics from a result jsonl file.
    """
    n = 0
    tool_calls: List[float] = []
    context_chars: List[float] = []
    elapsed_seconds: List[float] = []

    if not result_jsonl.exists():
        return {"count": 0}

    with result_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            n += 1
            tc = _safe_float(obj.get("tool_calls"))
            cc = _safe_float(obj.get("context_chars"))
            es = _safe_float(obj.get("elapsed_seconds"))
            if tc is not None:
                tool_calls.append(tc)
            if cc is not None:
                context_chars.append(cc)
            if es is not None:
                elapsed_seconds.append(es)

    def _mean(xs: List[float]):
        return (sum(xs) / len(xs)) if xs else None

    def _min(xs: List[float]):
        return min(xs) if xs else None

    def _max(xs: List[float]):
        return max(xs) if xs else None

    return {
        "count": n,
        "tool_calls": {"mean": _mean(tool_calls), "min": _min(tool_calls), "max": _max(tool_calls)},
        "context_chars": {"mean": _mean(context_chars), "min": _min(context_chars), "max": _max(context_chars)},
        "elapsed_seconds": {"mean": _mean(elapsed_seconds), "min": _min(elapsed_seconds), "max": _max(elapsed_seconds)},
    }


class Tee:
    """
    Duplicate writes to multiple file-like objects (e.g., console + log file).
    """

    def __init__(self, *streams):
        self.streams = streams

    def write(self, s: str) -> int:
        n = 0
        for st in self.streams:
            try:
                n = st.write(s)
            except Exception:
                pass
        return n

    def flush(self) -> None:
        for st in self.streams:
            try:
                st.flush()
            except Exception:
                pass


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_tokens", type=int, default=32768)
    parser.add_argument("--tool_count_max", type=int, default=200)
    parser.add_argument("--max_worker", type=int, default=60)
    parser.add_argument("--pool_no_progress_timeout", type=int, default=18000)
    parser.add_argument("--print_stream", action="store_true", default=True)
    parser.add_argument("--sequential", action="store_true")
    parser.add_argument(
        "--use_box",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Append a \\\\boxed{} final-answer format instruction to each query. Enabled by default; use --no-use_box to disable.",
    )
    parser.add_argument(
        "--pool_restart_rounds",
        type=int,
        default=0,
        help="When no progress for pool_no_progress_timeout, restart the async pool and rerun remaining tasks for this many extra rounds.",
    )
    parser.add_argument(
        "--max_retry_rounds",
        type=int,
        default=1,
        help="Maximum number of retry rounds to process queries without answers. After each round, check which queries still don't have answers and retry them. Set to 0 to disable auto-retry.",
    )
    parser.add_argument(
        "--filter_huggingface",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Filter out search results whose link contains 'huggingface'. Enabled by default; use --no-filter_huggingface to disable.",
    )

    _eval_dir = Path(__file__).parent.absolute()

    parser.add_argument(
        "--dataset_path",
        type=Path,
        default="",
        help="Path to the dataset JSONL file",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default="",
        help="Output directory for results",
    )
    parser.add_argument("--limit", type=int, default=-1, help="Run at most N questions.")
    parser.add_argument(
        "--run-log-path",
        type=Path,
        default=None,
        help="Save full run logs (stdout/stderr) to this file. Default: auto under out_dir.",
    )
    parser.add_argument(
        "--no-run-log",
        action="store_true",
        help="Disable saving full run logs (stdout/stderr).",
    )
    args = parser.parse_args()
    run_start_ts = time.time()

    import sys as _sys

    utils_dir = os.path.join(os.path.dirname(__file__), "..", "src")
    if str(utils_dir) not in _sys.path:
        _sys.path.insert(0, str(utils_dir))
    from llm_tool_openseeker_v3 import solve_query_with_tools

    test_data = read_jsonl(args.dataset_path)
    if args.use_box:
        test_data = append_box_instruction(test_data)
    if args.limit != -1:
        test_data = test_data[: args.limit]
    print(f">> Loaded {len(test_data)} questions from {args.dataset_path}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    save_path = args.out_dir / f"result_tool{args.tool_count_max}.jsonl"
    log_path = args.out_dir / f"result_tool{args.tool_count_max}.log.txt"
    run_log_path = args.run_log_path or (
        args.out_dir / f"result_tool{args.tool_count_max}.run.log"
    )
    metric_path = args.out_dir / f"result_tool{args.tool_count_max}_metrics.json"

    run_log_f = None
    orig_stdout, orig_stderr = sys.stdout, sys.stderr
    if not args.no_run_log:
        try:
            run_log_f = run_log_path.open("a", encoding="utf-8")
            sys.stdout = Tee(orig_stdout, run_log_f)
            sys.stderr = Tee(orig_stderr, run_log_f)
            print(f">> Run log: {run_log_path}")
        except Exception as e:
            sys.stdout, sys.stderr = orig_stdout, orig_stderr
            print(f">> Failed to enable run log at {run_log_path}: {e}")

    print(">> Config:")
    print(
        json.dumps(
            {
                "max_tokens": args.max_tokens,
                "tool_count_max": args.tool_count_max,
                "max_worker": args.max_worker,
                "base_url": os.getenv("OPENSEEKER_BASE_URL", "YOUR_OPENSEEKER_BASE_URL"),
                "model": os.getenv("OPENSEEKER_MODEL", "YOUR_MODEL_NAME"),
                "pool_no_progress_timeout": args.pool_no_progress_timeout,
                "filter_huggingface": args.filter_huggingface,
                "dataset_path": str(args.dataset_path),
                "use_box": args.use_box,
                "save_path": str(save_path),
                "error_log_path": str(log_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


    before = len(test_data)
    test_data = get_queries_without_answer(save_path, test_data)
    skipped = before - len(test_data)
    if skipped:
        print(f">> Dedup (valid answers only): {before} -> {len(test_data)} (skipped={skipped})")

    lock = asyncio.Lock()

    async def process_one(data: Dict[str, Any]) -> bool:
        q = data.get("query", "")
        t0 = time.time()
        print(f">> START query={q[:120]!r}")
        try:
            res = await asyncio.to_thread(
                solve_query_with_tools,
                q,
                max_tokens=args.max_tokens,
                tool_count_max=args.tool_count_max,
                print_stream=args.print_stream,
                filter_huggingface=args.filter_huggingface,
                return_full_traj=True,
            )
            out = dict(data)
            out["final_response"] = res.get("answer", "")
            out["tool_calls"] = res.get("tool_calls", None)
            out["elapsed_seconds"] = res.get("elapsed_seconds", None)
            out["context_chars"] = res.get("context_chars", None)
            out["context_est_tokens"] = res.get("context_est_tokens", None)
            out["finish_reason"] = res.get("finish_reason", None)
            if res.get("error") is not None:
                out["error"] = res.get("error")
            out["full_traj"] = res.get("full_traj", "")
            out["trace"] = res.get("trace", "")
            out["wall_seconds"] = time.time() - t0

            async with lock:
                with save_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(out, ensure_ascii=False) + "\n")
                    f.flush()
            print(
                f">> DONE  query={q[:120]!r} "
                f"wall={out['wall_seconds']:.2f}s tool_calls={out.get('tool_calls')} "
                f"ctx_chars={out.get('context_chars')}"
            )
            return True
        except Exception as e:
            err = traceback.format_exc()
            print(f"\033[91m>> FAILED query={q[:120]!r}\033[0m")
            print(f"\033[91m>> Error type: {type(e).__name__}\033[0m")
            print(f"\033[91m>> Error message: {str(e)}\033[0m")
            print(f"\033[91m>> Full traceback:\n{err}\033[0m")
            async with lock:
                with log_path.open("a", encoding="utf-8") as f:
                    f.write(f">> Error in processing query: {q}\n")
                    f.write(f">> Error type: {type(e).__name__}\n")
                    f.write(f">> Error message: {str(e)}\n")
                    f.write(err + "\n")
            print(f"\033[91m>> Query NOT saved to results file (will be retried later)\033[0m")
            return False

    def finalize() -> None:
        metrics = compute_metrics(save_path)
        metrics["run_total_seconds"] = time.time() - run_start_ts
        metrics["run_started_at_unix"] = run_start_ts
        metrics["run_finished_at_unix"] = time.time()
        try:
            metric_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as e:
            print(f">> Failed to write metrics to {metric_path}: {e}")
        print(f">> Metrics saved to {metric_path}")
        tc_mean = (metrics.get("tool_calls") or {}).get("mean")
        cc_mean = (metrics.get("context_chars") or {}).get("mean")
        print(f">> Avg tool_calls: {tc_mean}")
        print(f">> Avg context_chars: {cc_mean}")
        print(f">> Total run wall time (seconds): {metrics.get('run_total_seconds')}")

    async def run_processing_round(data_to_process: List[Dict[str, Any]]) -> Tuple[Set[str], Set[str]]:
        """
        Run one round of processing on the given data.
        Returns: (completed_ok_set, failed_final_set)
        """
        if args.sequential:
            completed_ok: Set[str] = set()
            failed_final: Set[str] = set()
            for d in data_to_process:
                q = d.get("query", "")
                ok = await process_one(d)
                if isinstance(q, str) and q:
                    if ok:
                        completed_ok.add(q)
                    else:
                        failed_final.add(q)
            return completed_ok, failed_final

        print("\n" + "=" * 100 + "\n>> Start to process the test data...")
        remaining = list(data_to_process)
        rounds_total = max(0, int(args.pool_restart_rounds)) + 1
        completed_ok: Set[str] = set()
        failed_final: Set[str] = set()

        for round_idx in range(rounds_total):
            if not remaining:
                break
            print(f">> Pool round {round_idx + 1}/{rounds_total}: remaining={len(remaining)}")

            semaphore = asyncio.Semaphore(args.max_worker)
            task2item: Dict[asyncio.Task, Dict[str, Any]] = {}

            async def process_with_semaphore(item: Dict[str, Any]) -> Tuple[str, bool]:
                async with semaphore:
                    q = item.get("query", "")
                    ok = await process_one(item)
                    return q, ok

            tasks = []
            for d in remaining:
                task = asyncio.create_task(process_with_semaphore(d))
                tasks.append(task)
                task2item[task] = d

            pending = set(tasks)
            last_progress = time.time()
            done_count = 0
            round_ok: Set[str] = set()
            round_fail: Set[str] = set()

            while pending:
                try:
                    done, pending = await asyncio.wait(pending, timeout=5, return_when=asyncio.FIRST_COMPLETED)
                    if not done:
                        if time.time() - last_progress > args.pool_no_progress_timeout:
                            print(
                                f"\n>> No progress for {args.pool_no_progress_timeout}s. "
                                f"Restarting pool; carry over remaining={len(pending)} tasks…"
                            )
                            break
                        continue

                    last_progress = time.time()
                    for task in done:
                        item = task2item.get(task, {})
                        q = item.get("query", "")
                        try:
                            q_result, ok = task.result()
                            if isinstance(q_result, str) and q_result:
                                q = q_result
                            if isinstance(q, str) and q:
                                if ok:
                                    round_ok.add(q)
                                else:
                                    round_fail.add(q)
                        except Exception as e:
                            if isinstance(q, str) and q:
                                round_fail.add(q)
                    done_count += len(done)
                    if done_count % 10 == 0:
                        print(f">> Progress (round {round_idx + 1}): {done_count}/{len(tasks)}")

                except Exception as e:
                    print(f">> Error in round processing: {e}")
                    break

            to_retry = []
            for task in pending:
                item = task2item.get(task)
                if item is not None:
                    to_retry.append(item)
                task.cancel()

            if pending:
                await asyncio.gather(*pending, return_exceptions=True)

            completed_ok |= round_ok
            failed_final |= round_fail

            remaining = to_retry

        if remaining:
            print(f">> WARNING: still remaining after {rounds_total} rounds: {len(remaining)} (likely stuck).")
            async with lock:
                with log_path.open("a", encoding="utf-8") as f:
                    f.write(f">> WARNING: remaining tasks after rounds_total={rounds_total}: {len(remaining)}\n")
                    for d in remaining:
                        f.write(f"  - {d.get('query','')}\n")

        return completed_ok, failed_final

    total_all = len(test_data)
    all_completed_ok: Set[str] = set()
    all_failed_final: Set[str] = set()
    current_data = list(test_data)
    retry_round = 0

    while current_data and (args.max_retry_rounds == 0 or retry_round < args.max_retry_rounds):
        if retry_round == 0:
            print(f"\n>> Initial processing round: {len(current_data)} queries")
        else:
            print(f"\n>> Retry round {retry_round}: processing {len(current_data)} queries without answers")

        round_ok, round_fail = await run_processing_round(current_data)
        all_completed_ok |= round_ok
        all_failed_final |= round_fail

        # Check which queries still don't have answers
        if args.max_retry_rounds > 0:
            current_data = get_queries_without_answer(save_path, test_data)
            if current_data:
                retry_round += 1
                print(f">> Found {len(current_data)} queries without answers, will retry in next round")
                if retry_round >= args.max_retry_rounds:
                    print(f">> Reached max_retry_rounds={args.max_retry_rounds}, stopping retry")
                    break
            else:
                print(f">> All queries have answers! Stopping retry loop.")
                break
        else:
            # If max_retry_rounds is 0, disable auto-retry
            break

    if current_data and args.max_retry_rounds > 0:
        print(f">> WARNING: After {retry_round} retry rounds, {len(current_data)} queries still don't have answers.")
        async with lock:
            with log_path.open("a", encoding="utf-8") as f:
                f.write(f">> WARNING: After {retry_round} retry rounds, {len(current_data)} queries still don't have answers:\n")
                for d in current_data:
                    f.write(f"  - {d.get('query','')}\n")

    print(f">> Done. Saved to {save_path}")
    print(f">> Summary: total={total_all} ok_written={len(all_completed_ok)} failed_not_written={len(all_failed_final)}")
    if args.max_retry_rounds > 0:
        print(f">> Retry rounds executed: {retry_round}")
    finalize()

    # Restore stdout/stderr and close run log file
    try:
        sys.stdout, sys.stderr = orig_stdout, orig_stderr
    except Exception:
        pass
    try:
        if run_log_f is not None:
            run_log_f.flush()
            run_log_f.close()
    except Exception:
        pass


if __name__ == "__main__":
    asyncio.run(main())
