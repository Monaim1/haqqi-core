from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class ExportRecord:
    file_name: str
    split: str
    language: str
    issue_number: str
    publication_date: str
    source_pdf_url: str
    checksum_sha256: str
    file_size_bytes: int
    relative_path: str
    repo_path: str
    sidecar_path: str


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Hugging Face upload metadata for PDF datasets")
    parser.add_argument(
        "--source-dir",
        default="data/raw-bulletin-officiel",
        help="Directory containing PDF files and JSON sidecars",
    )
    parser.add_argument("--glob", default="*.pdf", help="Glob pattern inside source-dir")
    parser.add_argument("--output-dir", default="hf_export", help="Output directory for metadata artifacts")
    parser.add_argument(
        "--repo-data-dir",
        default="data/raw-bulletin-officiel",
        help="Destination folder inside the Hugging Face dataset repo",
    )
    parser.add_argument("--repo-id", default="", help="Optional dataset repo id for README hints")
    parser.add_argument("--split", default="train", help="Split value to write in metadata")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of PDFs for smoke tests")
    parser.add_argument(
        "--checksum-mode",
        choices=["sidecar", "compute", "none"],
        default="sidecar",
        help="Use sidecar checksum, compute file checksum, or leave empty",
    )
    return parser.parse_args()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_sidecar(sidecar_path: Path) -> dict[str, Any]:
    if not sidecar_path.exists():
        return {}
    try:
        loaded = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _resolve_checksum(mode: str, sidecar: dict[str, Any], pdf_path: Path) -> str:
    if mode == "none":
        return ""
    if mode == "compute":
        return _file_sha256(pdf_path)

    checksum = sidecar.get("checksum")
    if isinstance(checksum, str) and checksum.strip():
        return checksum.strip()
    return ""


def _build_records(args: argparse.Namespace) -> list[ExportRecord]:
    source_dir = Path(args.source_dir).expanduser().resolve()
    if not source_dir.exists() or not source_dir.is_dir():
        raise FileNotFoundError(f"source-dir does not exist or is not a directory: {source_dir}")

    pdf_paths = sorted(source_dir.glob(args.glob))
    if args.limit is not None:
        pdf_paths = pdf_paths[: max(args.limit, 0)]

    records: list[ExportRecord] = []
    repo_data_dir = args.repo_data_dir.strip("/")
    for pdf_path in pdf_paths:
        sidecar_path = pdf_path.with_suffix(".json")
        sidecar = _read_sidecar(sidecar_path)
        rel_path = pdf_path.relative_to(source_dir).as_posix()
        checksum_sha256 = _resolve_checksum(args.checksum_mode, sidecar, pdf_path)

        language = sidecar.get("language") if isinstance(sidecar.get("language"), str) else ""
        issue_number = sidecar.get("issue_number") if isinstance(sidecar.get("issue_number"), str) else ""
        publication_date = (
            sidecar.get("publication_date") if isinstance(sidecar.get("publication_date"), str) else ""
        )
        source_pdf_url = sidecar.get("pdf_url") if isinstance(sidecar.get("pdf_url"), str) else ""

        records.append(
            ExportRecord(
                file_name=pdf_path.name,
                split=args.split,
                language=language,
                issue_number=issue_number,
                publication_date=publication_date,
                source_pdf_url=source_pdf_url,
                checksum_sha256=checksum_sha256,
                file_size_bytes=pdf_path.stat().st_size,
                relative_path=rel_path,
                repo_path=f"{repo_data_dir}/{rel_path}" if repo_data_dir else rel_path,
                sidecar_path=sidecar_path.relative_to(source_dir).as_posix() if sidecar_path.exists() else "",
            )
        )

    return records


def _write_metadata_csv(path: Path, records: list[ExportRecord]) -> None:
    fieldnames = list(ExportRecord.__dataclass_fields__.keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in records:
            writer.writerow(asdict(row))


def _write_metadata_jsonl(path: Path, records: list[ExportRecord]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")


def _write_upload_manifest(path: Path, source_dir: Path, records: list[ExportRecord]) -> None:
    files: list[dict[str, str]] = []
    for row in records:
        files.append(
            {
                "local_path": str((source_dir / row.relative_path).resolve()),
                "repo_path": row.repo_path,
            }
        )
        if row.sidecar_path:
            files.append(
                {
                    "local_path": str((source_dir / row.sidecar_path).resolve()),
                    "repo_path": row.repo_path.rsplit(".", 1)[0] + ".json",
                }
            )

    payload = {
        "description": "Upload plan for huggingface_hub.upload_folder or upload_file",
        "file_count": len(files),
        "files": files,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_readme(path: Path, repo_id: str, source_dir: Path, records: list[ExportRecord]) -> None:
    langs = sorted({row.language for row in records if row.language})
    total_bytes = sum(row.file_size_bytes for row in records)
    lang_block = "\n".join(f"- {lang}" for lang in langs) if langs else "- unknown"

    repo_line = f"Target HF repo: `{repo_id}`\n\n" if repo_id else ""
    text = (
        "# HF Export Bundle\n\n"
        f"{repo_line}"
        f"Source directory: `{source_dir}`\n\n"
        f"Documents: `{len(records)}`\n"
        f"Total size (bytes): `{total_bytes}`\n\n"
        "Languages:\n"
        f"{lang_block}\n\n"
        "Generated files:\n"
        "- `metadata.csv`\n"
        "- `metadata.jsonl`\n"
        "- `upload_manifest.json`\n"
    )
    path.write_text(text, encoding="utf-8")


def main() -> int:
    args = _parse_args()
    source_dir = Path(args.source_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    records = _build_records(args)

    metadata_csv = output_dir / "metadata.csv"
    metadata_jsonl = output_dir / "metadata.jsonl"
    manifest_json = output_dir / "upload_manifest.json"
    readme_md = output_dir / "README.md"

    _write_metadata_csv(metadata_csv, records)
    _write_metadata_jsonl(metadata_jsonl, records)
    _write_upload_manifest(manifest_json, source_dir, records)
    _write_readme(readme_md, args.repo_id, source_dir, records)

    print(f"Wrote {len(records)} records")
    print(f"- {metadata_csv}")
    print(f"- {metadata_jsonl}")
    print(f"- {manifest_json}")
    print(f"- {readme_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
