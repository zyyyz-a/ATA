from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PACKAGE_ROOT / "configs" / "prospective_select_certify_v12.draft.json"
DEFAULT_OUTPUT = PACKAGE_ROOT / "locks" / "prospective_select_certify_v12.revisions.lock.json"


def sibling_inventory(info) -> dict[str, dict[str, Any]]:
    output = {}
    for sibling in info.siblings or []:
        output[str(sibling.rfilename)] = {
            "bytes": int(sibling.size) if sibling.size is not None else None,
            "blob_id": getattr(sibling, "blob_id", None),
            "lfs_sha256": (getattr(sibling, "lfs", None) or {}).get("sha256") if isinstance(getattr(sibling, "lfs", None), dict) else None,
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Resolve v12 dataset/model revisions and source inventories without downloading data.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    from huggingface_hub import HfApi

    api = HfApi()
    datasets = {}
    for spec in config["datasets"]:
        info = api.dataset_info(spec["hf_id"], revision=spec["requested_revision"], files_metadata=True)
        inventory = sibling_inventory(info)
        expected = []
        if "source_file_pattern" in spec:
            expected = [spec["source_file_pattern"].format(year=year) for year in spec["source_years"]]
            missing = sorted(set(expected) - set(inventory))
            if missing:
                missing_years = {
                    str(year)
                    for year in spec["source_years"]
                    if spec["source_file_pattern"].format(year=year) in missing
                }
                nearby = sorted(
                    path
                    for path in inventory
                    if any(year in path for year in missing_years)
                )
                raise FileNotFoundError(
                    f"Missing registered source files for {spec['dataset_id']}: {missing[:5]}; "
                    f"repository paths containing the missing years: {nearby[:20]}"
                )
        datasets[spec["dataset_id"]] = {
            "hf_id": spec["hf_id"],
            "requested_revision": spec["requested_revision"],
            "resolved_revision": str(info.sha),
            "expected_source_files": expected,
            "expected_source_bytes": sum((inventory[path]["bytes"] or 0) for path in expected),
            "repository_file_count": len(inventory),
            "repository_known_bytes": sum((row["bytes"] or 0) for row in inventory.values()),
            "source_inventory": {path: inventory[path] for path in expected} if expected else inventory,
        }
    backbones = sorted({str(row["backbone"]) for row in config["datasets"]})
    models = {}
    for backbone in backbones:
        info = api.model_info(backbone, revision="main", files_metadata=True)
        models[backbone] = str(info.sha)
    payload = {
        "schema_version": "ata-nlp-select-certify-revisions-12.0",
        "status": "resolved_before_v12_model_or_deployment_outcomes",
        "datasets": datasets,
        "models": models,
        "test_outcomes_used": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    if args.output.exists() and args.output.read_text(encoding="utf-8") != serialized:
        raise FileExistsError(f"Refusing to replace a different revision lock: {args.output}")
    args.output.write_text(serialized, encoding="utf-8")
    print(json.dumps({
        "datasets": {key: {"revision": row["resolved_revision"], "expected_source_bytes": row["expected_source_bytes"]} for key, row in datasets.items()},
        "models": models,
    }, indent=2))


if __name__ == "__main__":
    main()
