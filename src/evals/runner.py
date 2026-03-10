import time
from dataclasses import dataclass

from qdrant_client import QdrantClient
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
from rich.table import Table

from app.services.semantic_search.retrieval import RetrieverService
from evals.dataset import EvalDataset
from evals.metrics import ndcg_at_k, mrr_at_k, recall_at_k


@dataclass
class EvalResult:
    query: str
    ndcg: dict[int, float]
    mrr: dict[int, float]
    recall: dict[int, float]
    latency_ms: float


class EvalRunner:
    def __init__(self, retriever_service: RetrieverService, client: QdrantClient, collection_name: str):
        self._retriever_service = retriever_service
        self._client = client
        self._collection_name = collection_name
        self.console = Console()
        self.k_values = [10, 30, 50]

    def run(self, dataset: EvalDataset, limit: int | None = None) -> list[EvalResult]:
        results: list[EvalResult] = []
        samples = dataset.samples if limit is None else dataset.samples[:limit]
        
        self.console.print(f"\n[bold]🧪 Running Evaluation: {dataset.name}[/bold]")
        self.console.print(f"Total samples: {len(samples)}\n")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=self.console,
        ) as progress:
            task = progress.add_task("Evaluating queries...", total=len(samples))

            for sample in samples:
                start_time = time.perf_counter()
                
                # Retrieve documents up to the maximum K
                max_k = max(self.k_values)
                hits = self._retriever_service.search(
                    client=self._client,
                    collection_name=self._collection_name,
                    query=sample.query,
                    top_k=max_k
                )
                
                elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                
                retrieved_ids = [hit.id for hit in hits]
                
                # Compute metrics for each K
                res = EvalResult(
                    query=sample.query,
                    ndcg={k: ndcg_at_k(retrieved_ids, sample.expected_ids, k) for k in self.k_values},
                    mrr={k: mrr_at_k(retrieved_ids, sample.expected_ids, k) for k in self.k_values},
                    recall={k: recall_at_k(retrieved_ids, sample.expected_ids, k) for k in self.k_values},
                    latency_ms=elapsed_ms
                )
                
                results.append(res)
                progress.advance(task)

        self._print_summary(results)
        return results

    def _print_summary(self, results: list[EvalResult]) -> None:
        if not results:
            self.console.print("[yellow]No results to summarize.[/yellow]")
            return

        avg_latency = sum(r.latency_ms for r in results) / len(results)

        table = Table(title="Evaluation Summary", show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan")
        table.add_column("Description", style="dim")
        for k in self.k_values:
            table.add_column(f"@{k}", justify="right", style="green")

        # Compute averages for each K
        avg_ndcg = {k: sum(r.ndcg[k] for r in results) / len(results) for k in self.k_values}
        avg_mrr = {k: sum(r.mrr[k] for r in results) / len(results) for k in self.k_values}
        avg_recall = {k: sum(r.recall[k] for r in results) / len(results) for k in self.k_values}

        table.add_row("Mean Recall", "Signal / noise ratio.", *[f"{avg_recall[k]:.4f}" for k in self.k_values])
        table.add_row("Mean MRR", "How high up in the search results was the first correct answer", *[f"{avg_mrr[k]:.4f}" for k in self.k_values])
        table.add_row("Mean NDCG", "How perfectly sorted is the entire list of results?", *[f"{avg_ndcg[k]:.4f}" for k in self.k_values])
        
        # Add latency row
        table.add_row(
            "Avg Latency", 
            "Time taken for retrieval & reranking",
            f"{avg_latency:.1f}ms", 
            *["" for _ in range(len(self.k_values) - 1)]
        )

        self.console.print("\n")
        self.console.print(table)
        self.console.print("\n")
