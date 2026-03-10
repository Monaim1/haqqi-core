from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class EvalSample:
    query: str
    expected_ids: list[str]


@dataclass
class EvalDataset:
    name: str
    samples: list[EvalSample]

    @classmethod
    def from_json(cls, path: str | Path) -> EvalDataset:
        """
        Load an evaluation dataset from a JSON file.
        Format expected:
        {
            "name": "My Dataset",
            "samples": [
                {
                    "query": "Le code pénal...",
                    "expected_ids": ["doc_123", "doc_456"]
                }
            ]
        }
        """
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        name = data.get("name", file_path.stem)
        samples_data = data.get("samples", [])
        
        samples = [
            EvalSample(
                query=sample["query"],
                expected_ids=sample.get("expected_ids", [])
            )
            for sample in samples_data
            if "query" in sample
        ]

        return cls(name=name, samples=samples)

    def save_json(self, path: str | Path) -> None:
        """Save the dataset to a JSON file."""
        file_path = Path(path)
        data = {
            "name": self.name,
            "samples": [
                {"query": s.query, "expected_ids": s.expected_ids}
                for s in self.samples
            ]
        }
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


import asyncio
import random
from typing import Any
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from qdrant_client import QdrantClient

class GoldenDatasetGenerator:
    """Generates golden evaluation datasets using an LLM to synthesize queries."""
    
    def __init__(self, client: QdrantClient, collection_name: str, console: Console | None = None):
        self._client = client
        self._collection_name = collection_name
        self.console = console or Console()
        
        # Initialize Gemini Model
        from app.services.deepAgents.agents import _get_model
        self._llm = _get_model()

    async def generate(self, dataset_name: str, limit: int = 20) -> EvalDataset:
        self.console.print(f"\n[bold]🌟 Generating Golden Dataset '{dataset_name}'[/bold]")
        self.console.print(f"Target size: {limit} samples\n")

        # 1. Fetch chunks from Qdrant collection
        # We fetch a larger pool and sample randomly
        if not self._client.collection_exists(self._collection_name):
            self.console.print("[red]Collection does not exist.[/red]")
            return EvalDataset(name=dataset_name, samples=[])
            
        collection_info = self._client.get_collection(self._collection_name)
        total_count = collection_info.points_count
        if total_count == 0:
            self.console.print("[red]Collection is empty. Cannot generate dataset.[/red]")
            return EvalDataset(name=dataset_name, samples=[])

        fetch_limit = min(limit * 5, total_count)
        records, _ = self._client.scroll(
            collection_name=self._collection_name, 
            limit=fetch_limit,
            with_payload=True
        )
        
        if not records:
            self.console.print("[red]Could not retrieve documents from collection.[/red]")
            return EvalDataset(name=dataset_name, samples=[])

        ids = [str(r.id) for r in records]
        documents = [getattr(r, "document", getattr(r, "payload", {}).get("document", "")) for r in records]

        # Filter out empty documents and zip
        valid_pairs = [(i, d) for i, d in zip(ids, documents) if d and isinstance(d, str) and len(d.strip()) > 50]
        
        if len(valid_pairs) < limit:
            self.console.print(f"[yellow]Warning: Only found {len(valid_pairs)} valid chunks (requested {limit}).[/yellow]")
            selected_pairs = valid_pairs
        else:
            selected_pairs = random.sample(valid_pairs, limit)

        samples: list[EvalSample] = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=self.console,
        ) as progress:
            task = progress.add_task("Synthesizing queries...", total=len(selected_pairs))

            # Process in small batches or concurrently
            coroutines = [
                self._generate_query_for_chunk(chunk_id, text)
                for chunk_id, text in selected_pairs
            ]
            
            for coro in asyncio.as_completed(coroutines):
                sample = await coro
                if sample:
                    samples.append(sample)
                progress.advance(task)

        self.console.print(f"\n[bold green]✓ Successfully generated {len(samples)} valid queries.[/bold green]")
        return EvalDataset(name=dataset_name, samples=samples)

    async def _generate_query_for_chunk(self, chunk_id: str, text: str) -> EvalSample | None:
        prompt = (
            "You are a citizen looking for legal information in Moroccan law.\n"
            "Read the following excerpt from a legal document and generate ONE realistic, natural "
            "question that a citizen would ask where this text is the direct answer.\n\n"
            "Rules:\n"
            "- Only output the question text, nothing else.\n"
            "- Do not use quotes around the question.\n"
            "- Write the question in French or Arabic depending on the text.\n"
            "- Be concise but specific enough that this text is the best answer.\n"
            "- Do not simply repeat the title or Dahir number.\n\n"
            f"EXCERPT:\n{text}\n\nQUESTION:"
        )
        
        try:
            from langchain_core.messages import HumanMessage
            from app.services.deepAgents.agents import _extract_text
            
            response = await self._llm.ainvoke([HumanMessage(content=prompt)])
            raw_content = _extract_text(response.content)
            query = raw_content.strip().strip('"').strip("'")
            
            if not query:
                return None
                
            return EvalSample(query=query, expected_ids=[chunk_id])
        except Exception as e:
            # Handle potential rate limits or errors gracefully
            self.console.print(f"[red]Error generating query: {e}[/red]")
            return None
