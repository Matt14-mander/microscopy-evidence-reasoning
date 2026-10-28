"""Download pinned public data/model assets to the project cache (no remote code)."""

import argparse
import json
from pathlib import Path


def main():
    from huggingface_hub import HfApi, snapshot_download

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=["model", "dataset"], required=True)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    repo = "jmhb/microvqa" if args.kind == "dataset" else "Qwen/Qwen2.5-VL-3B-Instruct"
    info = HfApi().repo_info(repo, repo_type=args.kind, revision=args.revision)
    path = Path(args.manifest)
    path.parent.mkdir(parents=True, exist_ok=True)
    report = {"repo_id": repo, "repo_type": args.kind, "revision": info.sha, "status": "downloading"}
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report), flush=True)
    try:
        snapshot = snapshot_download(repo, repo_type=args.kind, revision=info.sha, max_workers=2,
                                     allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja", "README.md"]
                                     if args.kind == "model" else ["README.md", "data/test-*.parquet"])
    except BaseException as error:
        report.update(status="failed", error_type=type(error).__name__)
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        raise
    report.update(status="complete", snapshot_path=snapshot)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
