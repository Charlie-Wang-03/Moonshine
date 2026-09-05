"""Tool discovery and dispatch for Moonshine."""

from __future__ import annotations

import hashlib
import json
import traceback
import uuid
from typing import Dict, List, Optional, Sequence

from moonshine.utils import read_jsonl, shorten, utc_now


TOOL_EXECUTION_STARTED = "tool_execution_started"
TOOL_EXECUTION_FINISHED = "tool_execution_finished"
TOOL_EXECUTION_AMBIGUOUS = "tool_execution_ambiguous"
TOOL_EXECUTION_BLOCKED = "tool_execution_blocked"


def collect_tool_schemas(
    registry,
    mode: Optional[str] = None,
    *,
    include: Optional[Sequence[str]] = None,
    exclude: Optional[Sequence[str]] = None,
) -> List[Dict[str, object]]:
    """Return provider-facing tool schemas."""
    return registry.schemas(mode=mode, include=include, exclude=exclude)


def _execution_store(runtime: Dict[str, object]):
    """Return the session store and id used for durable execution journaling."""
    runtime = dict(runtime or {})
    store = runtime.get("session_store")
    session_id = str(runtime.get("session_id") or "").strip()
    if store is None or not session_id:
        return None, ""
    return store, session_id


def _render_json(value: object) -> str:
    """Render a deterministic JSON-ish representation for hashes and previews."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _fingerprint(value: object, preview_chars: int = 800) -> Dict[str, str]:
    """Return bounded trace metadata without duplicating large tool payloads."""
    rendered = _render_json(value)
    return {
        "sha256": hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
        "preview": shorten(rendered, preview_chars),
    }


def _execution_payload(event: Dict[str, object]) -> Dict[str, object]:
    """Return one execution-event payload defensively."""
    payload = event.get("payload") or {}
    return dict(payload) if isinstance(payload, dict) else {}


def _turn_lifecycle(runtime: Dict[str, object]) -> Dict[str, object]:
    """Return monotonically numbered turn starts and currently open sequences."""
    store, session_id = _execution_store(runtime)
    paths = getattr(store, "paths", None) if store is not None else None
    if paths is None:
        return {"sequence": 0, "open_sequences": []}
    turn_events = [
        item
        for item in read_jsonl(paths.session_turn_events_file(session_id))
        if isinstance(item, dict) and str(item.get("type") or "") in {"turn_started", "turn_completed"}
    ]
    sequence = 0
    open_sequences: List[int] = []
    for item in turn_events:
        if str(item.get("type") or "") == "turn_started":
            sequence += 1
            open_sequences.append(sequence)
        elif open_sequences:
            # A resumed turn can complete while an older interrupted turn remains
            # unresolved, so pair completions with the most recent live start.
            open_sequences.pop()
    return {"sequence": sequence, "open_sequences": open_sequences}


def _unresolved_tool_executions(runtime: Dict[str, object]) -> List[Dict[str, object]]:
    """Return executions that started but cannot be proven terminal.

    ``tool_execution_ambiguous`` remains blocking by design: Moonshine cannot know
    whether a side-effecting handler completed before the interruption, so retrying
    automatically would risk duplicate external effects.
    """
    store, session_id = _execution_store(runtime)
    if store is None or not hasattr(store, "get_conversation_events"):
        return []

    active: Dict[str, Dict[str, object]] = {}
    for event in store.get_conversation_events(session_id):
        kind = str(event.get("event_kind") or "")
        if kind not in {TOOL_EXECUTION_STARTED, TOOL_EXECUTION_FINISHED, TOOL_EXECUTION_AMBIGUOUS}:
            continue
        payload = _execution_payload(event)
        execution_id = str(payload.get("execution_id") or "").strip()
        if not execution_id:
            continue
        if kind == TOOL_EXECUTION_FINISHED:
            active.pop(execution_id, None)
            continue
        record = dict(payload)
        record["state"] = "ambiguous" if kind == TOOL_EXECUTION_AMBIGUOUS else "started"
        record["event_id"] = event.get("id")
        active[execution_id] = record
    return list(active.values())


def _prior_interrupted_tool_turn(runtime: Dict[str, object]) -> Optional[Dict[str, object]]:
    """Return a prior open turn that executed tools before a later turn began.

    A finished handler is not enough to declare the *turn* durable. The process can
    die after the handler returns but before Moonshine records the tool result and
    final assistant message. Each execution intent stores its monotonic turn
    sequence, so recovery does not rely on coarse wall-clock timestamps.
    """
    store, session_id = _execution_store(runtime)
    if store is None or not hasattr(store, "get_conversation_events"):
        return None
    lifecycle = _turn_lifecycle(runtime)
    open_sequences = [int(item) for item in list(lifecycle.get("open_sequences") or [])]

    # During normal dispatch the current turn itself is open. Only older open
    # turns are recovery hazards.
    if len(open_sequences) <= 1:
        return None
    prior_sequences = set(open_sequences[:-1])
    for event in store.get_conversation_events(session_id):
        if str(event.get("event_kind") or "") != TOOL_EXECUTION_STARTED:
            continue
        payload = _execution_payload(event)
        try:
            turn_sequence = int(payload.get("turn_sequence") or 0)
        except (TypeError, ValueError):
            turn_sequence = 0
        if turn_sequence not in prior_sequences:
            continue
        return {
            "state": "interrupted_turn",
            "execution_id": str(payload.get("execution_id") or "turn:%s" % turn_sequence),
            "tool": str(payload.get("tool") or "unknown"),
            "call_id": str(payload.get("call_id") or ""),
            "turn_sequence": turn_sequence,
        }
    return None


def _append_execution_event(
    runtime: Dict[str, object],
    *,
    event_kind: str,
    content: str,
    payload: Dict[str, object],
) -> None:
    """Persist one execution lifecycle event when session storage is available."""
    store, session_id = _execution_store(runtime)
    if store is None or not hasattr(store, "append_conversation_event"):
        return
    store.append_conversation_event(
        session_id,
        event_kind=event_kind,
        role="tool",
        content=content,
        payload=dict(payload),
    )


def _mark_session_interrupted(runtime: Dict[str, object], interruption: Dict[str, object]) -> None:
    """Expose interrupted execution state in both session metadata stores."""
    store, session_id = _execution_store(runtime)
    if store is None:
        return
    now = utc_now()
    if hasattr(store, "update_session_meta"):
        store.update_session_meta(
            session_id,
            status="interrupted",
            updated_at=now,
            interrupted_tool_execution=dict(interruption),
        )
    db = getattr(store, "db", None)
    if db is not None and hasattr(db, "update_session"):
        db.update_session(session_id, updated_at=now, status="interrupted")


def _begin_tool_execution(call: object, runtime: Dict[str, object]) -> str:
    """Write a durable intent record immediately before dispatch."""
    execution_id = "tool-exec-%s" % uuid.uuid4().hex[:12]
    tool_name = str(getattr(call, "name", "") or "")
    call_id = str(getattr(call, "call_id", "") or "")
    arguments = dict(getattr(call, "arguments", {}) or {})
    arguments_fingerprint = _fingerprint(arguments)
    payload = {
        "execution_id": execution_id,
        "tool": tool_name,
        "call_id": call_id,
        "arguments_sha256": arguments_fingerprint["sha256"],
        "arguments_preview": arguments_fingerprint["preview"],
        "turn_sequence": int(_turn_lifecycle(runtime).get("sequence") or 0),
        "tool_round": runtime.get("_current_tool_round", ""),
        "started_at": utc_now(),
    }
    _append_execution_event(
        runtime,
        event_kind=TOOL_EXECUTION_STARTED,
        content="Tool execution started: %s (%s)" % (tool_name, call_id or execution_id),
        payload=payload,
    )
    return execution_id


def _finish_tool_execution(
    call: object,
    runtime: Dict[str, object],
    execution_id: str,
    *,
    output: object,
    error: Optional[str],
) -> None:
    """Write the terminal lifecycle record before the next tool is dispatched."""
    output_fingerprint = _fingerprint(output, preview_chars=1200)
    payload = {
        "execution_id": execution_id,
        "tool": str(getattr(call, "name", "") or ""),
        "call_id": str(getattr(call, "call_id", "") or ""),
        "turn_sequence": int(_turn_lifecycle(runtime).get("sequence") or 0),
        "tool_round": runtime.get("_current_tool_round", ""),
        "outcome": "error" if error else "ok",
        "error": shorten(str(error or ""), 500),
        "output_preview": output_fingerprint["preview"],
        "output_sha256": output_fingerprint["sha256"],
        "finished_at": utc_now(),
    }
    _append_execution_event(
        runtime,
        event_kind=TOOL_EXECUTION_FINISHED,
        content="Tool execution finished: %s (%s)" % (payload["tool"], payload["call_id"] or execution_id),
        payload=payload,
    )


def _mark_tool_execution_ambiguous(
    call: object,
    runtime: Dict[str, object],
    execution_id: str,
    exc: BaseException,
) -> None:
    """Record an interrupted dispatch whose external completion is unknowable."""
    now = utc_now()
    arguments_fingerprint = _fingerprint(dict(getattr(call, "arguments", {}) or {}))
    payload = {
        "execution_id": execution_id,
        "tool": str(getattr(call, "name", "") or ""),
        "call_id": str(getattr(call, "call_id", "") or ""),
        "arguments_sha256": arguments_fingerprint["sha256"],
        "arguments_preview": arguments_fingerprint["preview"],
        "turn_sequence": int(_turn_lifecycle(runtime).get("sequence") or 0),
        "tool_round": runtime.get("_current_tool_round", ""),
        "state": "ambiguous",
        "interruption_type": type(exc).__name__,
        "interruption": shorten(str(exc), 500),
        "interrupted_at": now,
    }
    _append_execution_event(
        runtime,
        event_kind=TOOL_EXECUTION_AMBIGUOUS,
        content="Tool execution became ambiguous after interruption: %s (%s)"
        % (payload["tool"], payload["call_id"] or execution_id),
        payload=payload,
    )
    _mark_session_interrupted(
        runtime,
        {
            "execution_id": execution_id,
            "tool": payload["tool"],
            "call_id": payload["call_id"],
            "state": "ambiguous",
        },
    )


def _blocked_results(calls: List[object], runtime: Dict[str, object], blockers: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    """Fail closed instead of dispatching new tools after an interrupted execution."""
    first = dict(blockers[0]) if blockers else {}
    blocker_tool = str(first.get("tool") or "unknown")
    blocker_call_id = str(first.get("call_id") or "unknown")
    blocker_execution_id = str(first.get("execution_id") or "unknown")
    blocker_state = str(first.get("state") or "ambiguous")
    _mark_session_interrupted(
        runtime,
        {
            "execution_id": blocker_execution_id,
            "tool": blocker_tool,
            "call_id": blocker_call_id,
            "state": blocker_state,
        },
    )
    if blocker_state == "interrupted_turn":
        reason = "a prior tool-bearing turn was interrupted before Moonshine durably completed the turn"
    else:
        reason = "a prior tool execution has ambiguous completion"
    message = (
        "Tool dispatch is blocked because %s: tool=%s, call_id=%s, execution_id=%s. "
        "Moonshine will not replay or dispatch additional tools automatically because prior "
        "handlers may already have produced external side effects. Inspect the session records "
        "and continue in a fresh session once the ambiguity is resolved."
        % (reason, blocker_tool, blocker_call_id, blocker_execution_id)
    )
    public_blockers = [
        {
            key: item.get(key)
            for key in ("state", "execution_id", "tool", "call_id", "turn_sequence", "started_at", "interrupted_at")
            if item.get(key) not in {None, ""}
        }
        for item in blockers
    ]
    results: List[Dict[str, object]] = []
    for call in calls:
        result = {
            "name": getattr(call, "name", ""),
            "call_id": getattr(call, "call_id", ""),
            "arguments": getattr(call, "arguments", {}),
            "output": {
                "status": "blocked_interrupted_execution",
                "message": message,
                "ambiguous_executions": public_blockers,
            },
            "error": message,
        }
        results.append(result)
        runtime.setdefault("_tool_results_in_round", []).append(result)
        _append_execution_event(
            runtime,
            event_kind=TOOL_EXECUTION_BLOCKED,
            content="Blocked tool dispatch: %s" % str(getattr(call, "name", "") or ""),
            payload={
                "tool": str(getattr(call, "name", "") or ""),
                "call_id": str(getattr(call, "call_id", "") or ""),
                "blocked_at": utc_now(),
                "blocker_states": [str(item.get("state") or "") for item in blockers],
                "ambiguous_execution_ids": [str(item.get("execution_id") or "") for item in blockers],
            },
        )
    return results


def handle_function_calls(registry, calls: List[object], runtime: Dict[str, object]) -> List[Dict[str, object]]:
    """Dispatch provider tool calls through the registry with crash-safe journaling.

    Each call is journaled immediately before dispatch and receives a durable
    terminal record before the next call begins. If execution is interrupted by a
    process-level exception such as ``KeyboardInterrupt``, the call is marked
    ambiguous and the exception is re-raised. A later tool-bearing turn also stays
    blocked if an earlier tool-bearing turn never reached ``turn_completed``.
    """
    blockers = _unresolved_tool_executions(runtime)
    interrupted_turn = _prior_interrupted_tool_turn(runtime)
    if interrupted_turn is not None:
        blockers.append(interrupted_turn)
    if blockers:
        return _blocked_results(calls, runtime, blockers)

    results = []
    for call in calls:
        execution_id = _begin_tool_execution(call, runtime)
        try:
            try:
                result = registry.dispatch(call.name, call.arguments, runtime)
                error = None
            except Exception as exc:
                result = {
                    "error": str(exc),
                    "traceback": traceback.format_exc(limit=3),
                }
                error = str(exc)
            _finish_tool_execution(
                call,
                runtime,
                execution_id,
                output=result,
                error=error,
            )
        except BaseException as exc:
            try:
                _mark_tool_execution_ambiguous(call, runtime, execution_id, exc)
            except Exception:
                # Never replace the process-level interruption with a best-effort
                # journaling failure. A durable start record, when it was written,
                # is itself enough for the next process to fail closed.
                pass
            raise

        result_record = {
            "name": call.name,
            "call_id": getattr(call, "call_id", ""),
            "arguments": call.arguments,
            "output": result,
            "error": error,
        }
        results.append(result_record)
        runtime.setdefault("_tool_results_in_round", []).append(result_record)
    return results
