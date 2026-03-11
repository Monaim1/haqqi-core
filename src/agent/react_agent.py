from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import asdict
from typing import Any

from src.config import Settings
from src.services.semantic_search import SemanticSearchService
from src.services.semantic_search.ingestion import IngestionProgressEvent


def _extract_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts).strip()
    return str(content)


class GroundedRetrievalAgent:
    """LangGraph ReAct agent over the retrieval pipeline with thread-level state."""

    def __init__(self, settings: Settings | None = None):
        self._settings = settings or Settings()
        self._service = SemanticSearchService()
        self._session_state: dict[str, dict[str, Any]] = {}
        self._active_thread_id = "default"
        self._app = self._build_graph()

    def _build_graph(self):
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            from langchain_core.tools import tool
            from langgraph.checkpoint.memory import MemorySaver
            from langgraph.prebuilt import create_react_agent
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "Agent dependencies are missing. Install them with `uv sync` after updating dependencies."
            ) from exc

        if not os.getenv("GOOGLE_API_KEY", "").strip():
            raise RuntimeError(
                "Missing GOOGLE_API_KEY. Set it in your environment or .env before running `haqqi agent`."
            )

        model = ChatGoogleGenerativeAI(
            model=self._settings.agent_model_name,
            temperature=0,
        )

        def _state_for_current_thread() -> dict[str, Any]:
            thread = self._active_thread_id or "default"
            if thread not in self._session_state:
                self._session_state[thread] = {
                    "tool_calls": 0,
                    "last_query": "",
                    "last_hits": [],
                    "last_ingestion_report": None,
                }
            return self._session_state[thread]

        @tool
        def retrieve_context(query: str, top_k: int | None = None) -> str:
            """Search the vector database and return grounded snippets with source metadata."""
            state = _state_for_current_thread()
            response = self._service.search(query=query, top_k=top_k)
            state["tool_calls"] += 1
            state["last_query"] = query
            state["last_hits"] = [hit.model_dump() for hit in response.hits]

            if not response.hits:
                return "No results found."

            lines = []
            for idx, hit in enumerate(response.hits, start=1):
                source = hit.metadata.get("source_filename", "unknown")
                page = hit.metadata.get("page_number", "n/a")
                score = hit.score if isinstance(hit.score, (float, int)) else 0.0
                lines.append(
                    f"[{idx}] score={float(score):.4f} "
                    f"source={source} page={page}\n{hit.text}"
                )
            return "\n\n".join(lines)

        @tool
        def run_ingestion(
            limit: int | None = None,
            include_colbert: bool = True,
        ) -> str:
            """Index documents into Qdrant. Use this if retrieval returns no results because collection is empty."""
            state = _state_for_current_thread()
            progress_events: list[dict[str, Any]] = []

            def _progress(event: IngestionProgressEvent) -> None:
                progress_events.append(asdict(event))

            dense = self._service.ingest_dense(limit=limit, progress_callback=_progress)
            colbert = self._service.ingest_colbert(limit=limit, progress_callback=_progress) if include_colbert else None
            payload = {
                "dense": asdict(dense),
                "colbert": asdict(colbert) if colbert is not None else None,
                "events": progress_events,
            }
            state["tool_calls"] += 1
            state["last_ingestion_report"] = payload
            return json.dumps(payload, ensure_ascii=False)

        @tool
        def get_session_state() -> str:
            """Return current thread state (last query, last hits, ingestion status)."""
            state = _state_for_current_thread()
            return json.dumps(state, ensure_ascii=False)

        @tool
        def clear_session_state() -> str:
            """Clear this thread's state."""
            thread = self._active_thread_id or "default"
            self._session_state[thread] = {
                "tool_calls": 0,
                "last_query": "",
                "last_hits": [],
                "last_ingestion_report": None,
            }
            return "Session state cleared."

        system_prompt = (
            "You are a grounded legal retrieval agent. "
            "Use tools before answering factual/legal questions. "
            "When you answer, cite retrieved snippets as [1], [2], etc. "
            "If evidence is missing or weak, say so explicitly and ask for a better query."
        )
        checkpointer = MemorySaver()
        return create_react_agent(
            model=model,
            tools=[retrieve_context, run_ingestion, get_session_state, clear_session_state],
            prompt=system_prompt,
            checkpointer=checkpointer,
        )

    def ask(self, message: str, thread_id: str = "default", recursion_limit: int | None = None) -> str:
        final_limit = (
            recursion_limit
            if isinstance(recursion_limit, int) and recursion_limit > 0
            else self._settings.agent_recursion_limit
        )
        self._active_thread_id = thread_id
        result = self._app.invoke(
            {"messages": [("user", message)]},
            config={"configurable": {"thread_id": thread_id}, "recursion_limit": final_limit},
        )
        messages = result.get("messages", [])
        if not messages:
            return ""
        return _extract_text(messages[-1].content)

    def chat_loop(
        self,
        *,
        thread_id: str = "default",
        recursion_limit: int | None = None,
        print_fn: Callable[[str], None] = print,
        input_fn: Callable[[str], str] = input,
    ) -> int:
        print_fn("Interactive agent session. Type 'exit' to stop.")
        while True:
            user_text = input_fn("> ").strip()
            if not user_text:
                continue
            if user_text.lower() in {"exit", "quit"}:
                return 0
            answer = self.ask(user_text, thread_id=thread_id, recursion_limit=recursion_limit)
            print_fn(answer)
