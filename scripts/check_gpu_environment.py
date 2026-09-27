from __future__ import annotations

import importlib.metadata
import json
import platform


def version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def main() -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Formal ATA NLP experiments require CUDA.")
    payload = {
        "status": "gpu_ready",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu_count": torch.cuda.device_count(),
        "gpus": [
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "total_memory_bytes": torch.cuda.get_device_properties(index).total_memory,
            }
            for index in range(torch.cuda.device_count())
        ],
        "packages": {
            name: version(name)
            for name in (
                "transformers", "datasets", "peft", "accelerate", "evaluate", "seqeval",
                "numpy", "scipy", "scikit-learn", "sentencepiece", "safetensors",
            )
        },
    }
    missing = [name for name, value in payload["packages"].items() if value is None]
    if missing:
        raise RuntimeError(f"Missing NLP packages: {missing}. Install requirements-nlp.txt without reinstalling torch.")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
