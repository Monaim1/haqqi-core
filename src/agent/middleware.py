from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class HumanInTheLoopMiddleware:
    """
    Placeholder middleware for human-in-the-loop control.

    Current behavior:
    - counts tool calls per thread
    - allows all calls by default

    Future behavior (not implemented yet):
    - block/queue specific tool calls for human approval
    - route approval requests to UI or queue backend
    """

    enabled: bool = True
    tool_call_counts: dict[str, int] = field(default_factory=dict)
    pending_approvals: dict[str, list[str]] = field(default_factory=dict)

    def before_tool_call(self, *, thread_id: str, tool_name: str) -> bool:
        if not self.enabled:
            return True

        self.tool_call_counts[thread_id] = self.tool_call_counts.get(thread_id, 0) + 1
        # Placeholder: always allow, but keep a hook for future HITL policy.
        del tool_name
        return True

    def register_pending(self, *, thread_id: str, tool_name: str) -> None:
        if thread_id not in self.pending_approvals:
            self.pending_approvals[thread_id] = []
        self.pending_approvals[thread_id].append(tool_name)

    def clear_thread(self, thread_id: str) -> None:
        self.tool_call_counts.pop(thread_id, None)
        self.pending_approvals.pop(thread_id, None)

