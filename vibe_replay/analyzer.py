"""Session analyzer for Vibe Replay.

Processes raw events into structured insights:
- Timeline phases (exploration, implementation, debugging, etc.)
- Key decision points and turning points
- Learnings, mistakes, and patterns
- Cross-session aggregation
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from typing import Any

from .models import (
    Event,
    EventType,
    Insight,
    InsightType,
    SessionMetadata,
    SessionPhase,
    SessionReplay,
    TimelinePhase,
)
from .store import SessionStore


# Tool-to-phase mapping heuristics
PHASE_INDICATORS: dict[str, list[SessionPhase]] = {
    "Read": [SessionPhase.EXPLORATION],
    "Glob": [SessionPhase.EXPLORATION],
    "Grep": [SessionPhase.EXPLORATION, SessionPhase.DEBUGGING],
    "WebSearch": [SessionPhase.EXPLORATION],
    "WebFetch": [SessionPhase.EXPLORATION],
    "Edit": [SessionPhase.IMPLEMENTATION, SessionPhase.REFACTORING],
    "Write": [SessionPhase.IMPLEMENTATION, SessionPhase.CONFIGURATION],
    "Bash": [SessionPhase.TESTING, SessionPhase.DEBUGGING, SessionPhase.CONFIGURATION],
    "NotebookEdit": [SessionPhase.IMPLEMENTATION],
    "Task": [SessionPhase.IMPLEMENTATION],
}

# Keywords that suggest debugging
DEBUG_KEYWORDS = {
    "error", "fail", "bug", "fix", "debug", "traceback", "exception",
    "broken", "crash", "issue", "wrong", "problem", "unexpected",
}

# Keywords that suggest testing
TEST_KEYWORDS = {
    "test", "pytest", "assert", "spec", "coverage", "mock", "fixture",
}

# Keywords that suggest configuration
CONFIG_KEYWORDS = {
    "config", "setup", "install", "dependency", "pyproject", "package",
    "requirements", "env", "docker", "deploy",
}


def _detect_phase_from_event(event: Event) -> SessionPhase:
    """Detect the likely phase of a single event based on heuristics."""
    summary_lower = event.summary.lower()
    tool = event.tool_name or ""

    # Check for debugging signals
    if any(kw in summary_lower for kw in DEBUG_KEYWORDS):
        return SessionPhase.DEBUGGING

    # Check for testing signals
    if any(kw in summary_lower for kw in TEST_KEYWORDS):
        return SessionPhase.TESTING

    # Check for configuration signals
    if any(kw in summary_lower for kw in CONFIG_KEYWORDS):
        return SessionPhase.CONFIGURATION

    # Check for documentation
    if any(kw in summary_lower for kw in ("readme", "doc", "comment", "docstring")):
        return SessionPhase.DOCUMENTATION

    # Use tool-based heuristic
    phases = PHASE_INDICATORS.get(tool, [SessionPhase.UNKNOWN])
    return phases[0]


def _identify_phase_runs(events: list[Event]) -> list[TimelinePhase]:
    """Group consecutive events into phase runs.

    Uses a sliding window approach: assigns a phase to each event,
    then merges consecutive events of the same phase.
    """
    if not events:
        return []

    # Assign phase to each event
    phases = [_detect_phase_from_event(e) for e in events]

    # Merge consecutive same-phase events (with some tolerance)
    runs: list[TimelinePhase] = []
    current_phase = phases[0]
    start_idx = 0

    for i in range(1, len(phases)):
        if phases[i] != current_phase:
            # Check if this is just a brief interruption (1-2 events)
            # that should be merged with the surrounding phase
            look_ahead = min(i + 3, len(phases))
            if (
                look_ahead - i >= 2
                and all(p == current_phase for p in phases[i + 1 : look_ahead])
            ):
                # Skip this blip
                phases[i] = current_phase
                continue

            runs.append(
                TimelinePhase(
                    phase=current_phase,
                    start_index=start_idx,
                    end_index=i - 1,
                    start_time=events[start_idx].timestamp,
                    end_time=events[i - 1].timestamp,
                    event_count=i - start_idx,
                )
            )
            current_phase = phases[i]
            start_idx = i

    # Final run
    runs.append(
        TimelinePhase(
            phase=current_phase,
            start_index=start_idx,
            end_index=len(events) - 1,
            start_time=events[start_idx].timestamp,
            end_time=events[-1].timestamp,
            event_count=len(events) - start_idx,
        )
    )

    # Merge very short runs (< 3 events) into neighbors
    merged: list[TimelinePhase] = []
    for run in runs:
        if merged and run.event_count < 3 and merged[-1].phase == run.phase:
            # Extend previous run
            prev = merged[-1]
            merged[-1] = TimelinePhase(
                phase=prev.phase,
                start_index=prev.start_index,
                end_index=run.end_index,
                start_time=prev.start_time,
                end_time=run.end_time,
                event_count=prev.event_count + run.event_count,
                key_events=prev.key_events + run.key_events,
            )
        elif merged and run.event_count < 2:
            # Absorb tiny runs into previous
            prev = merged[-1]
            merged[-1] = TimelinePhase(
                phase=prev.phase,
                start_index=prev.start_index,
                end_index=run.end_index,
                start_time=prev.start_time,
                end_time=run.end_time,
                event_count=prev.event_count + run.event_count,
                key_events=prev.key_events,
            )
        else:
            merged.append(run)

    return merged


def _generate_phase_summaries(
    phases: list[TimelinePhase], events: list[Event]
) -> list[TimelinePhase]:
    """Generate human-readable summaries for each phase."""
    result = []
    for phase in phases:
        phase_events = events[phase.start_index : phase.end_index + 1]

        # Count tools used
        tools = Counter(e.tool_name for e in phase_events if e.tool_name)
        files = set()
        for e in phase_events:
            files.update(e.files_affected)

        # Build summary
        parts = []
        if phase.phase == SessionPhase.EXPLORATION:
            parts.append(f"Explored {len(files)} file(s)")
            if "Grep" in tools:
                parts.append(f"searched {tools['Grep']} time(s)")
        elif phase.phase == SessionPhase.IMPLEMENTATION:
            parts.append(f"Modified {len(files)} file(s)")
        elif phase.phase == SessionPhase.DEBUGGING:
            parts.append("Investigated and fixed issues")
        elif phase.phase == SessionPhase.TESTING:
            parts.append("Ran tests")
        elif phase.phase == SessionPhase.CONFIGURATION:
            parts.append("Set up configuration")
        elif phase.phase == SessionPhase.DOCUMENTATION:
            parts.append("Updated documentation")

        parts.append(f"{phase.event_count} event(s)")

        # Identify key events (code changes, errors)
        key_indices = []
        for i, e in enumerate(phase_events):
            global_idx = phase.start_index + i
            if e.event_type in (EventType.CODE_CHANGE, EventType.ERROR):
                key_indices.append(global_idx)

        updated = phase.model_copy(
            update={
                "summary": " | ".join(parts),
                "key_events": key_indices[:10],  # Cap at 10 key events
            }
        )
        result.append(updated)

    return result


def _extract_insights(events: list[Event]) -> list[Insight]:
    """Extract structured insights from the event stream."""
    insights: list[Insight] = []

    # Track patterns
    error_indices: list[int] = []
    fix_after_error: list[tuple[int, int]] = []  # (error_idx, fix_idx)
    tools_sequence: list[str] = []
    files_touched: dict[str, list[int]] = defaultdict(list)

    for i, event in enumerate(events):
        tools_sequence.append(event.tool_name or "")

        for f in event.files_affected:
            files_touched[f].append(i)

        # Track errors and subsequent fixes
        if event.event_type == EventType.ERROR:
            error_indices.append(i)
        elif error_indices and event.event_type == EventType.CODE_CHANGE:
            # A code change after an error might be a fix
            fix_after_error.append((error_indices[-1], i))

    # Insight: Detour detection (error → many exploration events → fix)
    for err_idx, fix_idx in fix_after_error:
        gap = fix_idx - err_idx
        if gap > 5:
            # Significant detour
            insights.append(
                Insight(
                    insight_type=InsightType.MISTAKE,
                    title="Debugging detour detected",
                    description=(
                        f"An error at event #{err_idx} led to {gap} events of "
                        f"investigation before a fix at event #{fix_idx}. "
                        f"Error: {events[err_idx].summary}"
                    ),
                    supporting_events=[err_idx, fix_idx],
                    confidence=0.6,
                )
            )

    # Insight: Frequently modified files (possible hotspot)
    for filepath, indices in files_touched.items():
        if len(indices) >= 4:
            insights.append(
                Insight(
                    insight_type=InsightType.PATTERN,
                    title=f"Hotspot file: {filepath.split('/')[-1]}",
                    description=(
                        f"{filepath} was modified {len(indices)} times during "
                        f"the session, suggesting it's a central piece."
                    ),
                    supporting_events=indices[:5],
                    confidence=0.7,
                )
            )

    # Insight: Exploration patterns
    exploration_runs = 0
    current_run = 0
    for tool in tools_sequence:
        if tool in ("Read", "Glob", "Grep", "WebSearch"):
            current_run += 1
        else:
            if current_run >= 5:
                exploration_runs += 1
            current_run = 0
    if current_run >= 5:
        exploration_runs += 1

    if exploration_runs >= 2:
        insights.append(
            Insight(
                insight_type=InsightType.PATTERN,
                title="Multiple exploration phases",
                description=(
                    f"The session had {exploration_runs} significant exploration "
                    f"phases (5+ consecutive read/search operations), suggesting "
                    f"an iterative discovery process."
                ),
                confidence=0.7,
            )
        )

    # Insight: Tool usage patterns
    tool_counts = Counter(t for t in tools_sequence if t)
    total = sum(tool_counts.values())
    if total > 0:
        # Heavy reading ratio
        read_tools = tool_counts.get("Read", 0) + tool_counts.get("Grep", 0) + tool_counts.get("Glob", 0)
        if read_tools / total > 0.6:
            insights.append(
                Insight(
                    insight_type=InsightType.PATTERN,
                    title="Read-heavy session",
                    description=(
                        f"{read_tools}/{total} events ({read_tools*100//total}%) "
                        f"were reading/searching operations. This session was "
                        f"primarily about understanding existing code."
                    ),
                    confidence=0.8,
                )
            )

        # Heavy writing ratio
        write_tools = tool_counts.get("Edit", 0) + tool_counts.get("Write", 0)
        if total > 5 and write_tools / total > 0.5:
            insights.append(
                Insight(
                    insight_type=InsightType.PATTERN,
                    title="Implementation-heavy session",
                    description=(
                        f"{write_tools}/{total} events ({write_tools*100//total}%) "
                        f"were code modifications. This was a focused "
                        f"implementation session."
                    ),
                    confidence=0.8,
                )
            )

    return insights


def _identify_decision_points(events: list[Event]) -> list[int]:
    """Identify events that represent key decision points.

    A decision point is when the direction of work changes significantly.
    """
    decision_indices: list[int] = []

    for i, event in enumerate(events):
        # Direction changes: switching from exploration to implementation
        if i > 0:
            prev_phase = _detect_phase_from_event(events[i - 1])
            curr_phase = _detect_phase_from_event(event)
            if prev_phase != curr_phase and curr_phase == SessionPhase.IMPLEMENTATION:
                decision_indices.append(i)

        # First code change in a new file
        if event.event_type == EventType.CODE_CHANGE and event.files_affected:
            # Check if this file was touched before
            prev_files = set()
            for prev in events[:i]:
                prev_files.update(prev.files_affected)
            new_files = [f for f in event.files_affected if f not in prev_files]
            if new_files:
                decision_indices.append(i)

    return sorted(set(decision_indices))


def _identify_turning_points(events: list[Event]) -> list[int]:
    """Identify turning points in the session.

    A turning point is when something significant changes the course:
    errors, breakthroughs, test results.
    """
    turning_points: list[int] = []

    for i, event in enumerate(events):
        # Errors are turning points
        if event.event_type == EventType.ERROR:
            turning_points.append(i)

        # Bash commands with test results
        if event.tool_name == "Bash":
            summary_lower = event.summary.lower()
            if any(kw in summary_lower for kw in ("test", "pytest", "npm test")):
                turning_points.append(i)

    return sorted(set(turning_points))


def _compute_statistics(events: list[Event], metadata: SessionMetadata) -> dict[str, Any]:
    """Compute aggregate statistics for the session."""
    tool_counts = Counter(e.tool_name for e in events if e.tool_name)
    event_type_counts = Counter(e.event_type.value for e in events)
    all_files = set()
    for e in events:
        all_files.update(e.files_affected)

    # Compute duration
    if events:
        duration = (events[-1].timestamp - events[0].timestamp).total_seconds()
    else:
        duration = 0

    return {
        "total_events": len(events),
        "duration_seconds": duration,
        "duration_human": _format_duration(duration),
        "tools_used": dict(tool_counts.most_common()),
        "event_types": dict(event_type_counts),
        "files_affected": sorted(all_files),
        "file_count": len(all_files),
        "most_used_tool": tool_counts.most_common(1)[0][0] if tool_counts else None,
        "code_changes": event_type_counts.get("code_change", 0),
        "errors_encountered": event_type_counts.get("error", 0),
    }


def _format_duration(seconds: float) -> str:
    """Format duration in seconds to a human-readable string."""
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        return f"{int(seconds // 60)}m {int(seconds % 60)}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def analyze_session(
    store: SessionStore, session_id: str
) -> SessionReplay:
    """Run full analysis on a captured session.

    Args:
        store: The session store.
        session_id: The session to analyze.

    Returns:
        Complete SessionReplay with timeline, insights, and statistics.
    """
    events = store.get_events(session_id)
    metadata = store.get_metadata(session_id)

    if not metadata:
        # Generate metadata from events
        metadata = SessionMetadata(
            session_id=session_id,
            start_time=events[0].timestamp if events else datetime.now(),
            end_time=events[-1].timestamp if events else None,
            event_count=len(events),
        )

    # Build timeline phases
    phases = _identify_phase_runs(events)
    phases = _generate_phase_summaries(phases, events)

    # Extract insights
    insights = _extract_insights(events)

    # Identify decision and turning points
    decision_points = _identify_decision_points(events)
    turning_points = _identify_turning_points(events)

    # Compute statistics
    stats = _compute_statistics(events, metadata)

    # Update metadata with computed info
    all_files = set()
    tool_counts = Counter()
    for e in events:
        all_files.update(e.files_affected)
        if e.tool_name:
            tool_counts[e.tool_name] += 1

    metadata = metadata.model_copy(
        update={
            "event_count": len(events),
            "end_time": events[-1].timestamp if events else metadata.end_time,
            "duration_seconds": stats["duration_seconds"],
            "files_modified": sorted(all_files),
            "tools_used": dict(tool_counts),
        }
    )

    # Generate session summary
    if not metadata.summary and events:
        metadata = metadata.model_copy(
            update={"summary": _generate_session_summary(events, phases, insights)}
        )

    # Save updated metadata
    store.save_metadata(metadata)

    replay = SessionReplay(
        metadata=metadata,
        timeline=phases,
        insights=insights,
        key_decision_indices=decision_points,
        turning_point_indices=turning_points,
        statistics=stats,
    )

    # Save the replay
    store.save_replay(replay)

    return replay


def _generate_session_summary(
    events: list[Event],
    phases: list[TimelinePhase],
    insights: list[Insight],
) -> str:
    """Generate a concise session summary."""
    parts = []

    # Count main activities
    tool_counts = Counter(e.tool_name for e in events if e.tool_name)
    all_files = set()
    for e in events:
        all_files.update(e.files_affected)

    if all_files:
        parts.append(f"Worked on {len(all_files)} file(s)")

    phase_counts = Counter(p.phase.value for p in phases)
    main_phases = [p for p, _ in phase_counts.most_common(2)]
    if main_phases:
        parts.append(f"primarily {', '.join(main_phases)}")

    error_count = sum(1 for e in events if e.event_type == EventType.ERROR)
    if error_count:
        parts.append(f"encountered {error_count} error(s)")

    return " | ".join(parts) if parts else "Session recorded"


def aggregate_learnings(
    store: SessionStore, limit: int = 50
) -> list[Insight]:
    """Aggregate learnings across multiple sessions.

    Args:
        store: The session store.
        limit: Maximum number of sessions to consider.

    Returns:
        Aggregated insights across sessions.
    """
    sessions = store.list_sessions(limit=limit)
    all_insights: list[Insight] = []

    for session_meta in sessions:
        replay = store.get_replay(session_meta.session_id)
        if replay:
            all_insights.extend(replay.insights)

    # Deduplicate similar insights
    unique: list[Insight] = []
    seen_titles: set[str] = set()
    for insight in all_insights:
        title_key = insight.title.lower().strip()
        if title_key not in seen_titles:
            seen_titles.add(title_key)
            unique.append(insight)

    # Sort by confidence
    unique.sort(key=lambda x: x.confidence, reverse=True)
    return unique
