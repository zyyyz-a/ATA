#!/usr/bin/env python3
"""Fail-closed CUDA compute preflight for formal experiment hosts."""

from __future__ import annotations

import json
import platform
import sys
import time
from datetime import datetime, timezone


def main() -> int:
    report: dict[str, object] = {
        "schema_version": "1.0",
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "status": "fail",
    }
    started = time.perf_counter()
    try:
        import torch

        report.update(
            {
                "torch": torch.__version__,
                "cuda_runtime": torch.version.cuda,
                "cuda_available": torch.cuda.is_available(),
                "device_count": torch.cuda.device_count(),
            }
        )
        if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
            raise RuntimeError("CUDA is not available to PyTorch")

        device = torch.device("cuda:0")
        props = torch.cuda.get_device_properties(device)
        x = torch.ones((1024, 1024), dtype=torch.float32, device=device)
        y = x @ x
        torch.cuda.synchronize(device)
        report.update(
            {
                "device": str(device),
                "device_name": props.name,
                "device_total_memory_bytes": props.total_memory,
                "tensor_sum": float(x.sum().item()),
                "matmul_probe": float(y[0, 0].item()),
                "allocated_memory_bytes": torch.cuda.memory_allocated(device),
                "elapsed_seconds": round(time.perf_counter() - started, 6),
                "status": "pass",
            }
        )
    except Exception as exc:  # Fail closed and preserve the diagnostic.
        report["error_type"] = type(exc).__name__
        report["error"] = str(exc)
        report["elapsed_seconds"] = round(time.perf_counter() - started, 6)

    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
