from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE_ROOT.parent
SCRIPTS = PACKAGE_ROOT / "scripts"


def run(script: str, *arguments: str) -> None:
    subprocess.run([sys.executable, str(SCRIPTS / script), *arguments], cwd=PROJECT_ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare v12 without launching any GPU job.")
    parser.add_argument("--resolve-revisions", action="store_true")
    parser.add_argument("--materialize-data", action="store_true")
    parser.add_argument("--seal", action="store_true")
    args = parser.parse_args()
    run("build_prospective_select_certify_v12_manifest.py")
    subprocess.run(
        [sys.executable, "-m", "unittest", "nlp_ata_experiments.tests.test_select_certify_v12"],
        cwd=PROJECT_ROOT, check=True,
    )
    run("validate_select_certify_v12.py")
    if args.resolve_revisions or args.materialize_data or args.seal:
        run("resolve_select_certify_v12_revisions.py")
    if args.materialize_data or args.seal:
        run("materialize_select_certify_v12.py")
    if args.seal:
        run("seal_select_certify_v12.py")
        run("orchestrate_select_certify_v12.py", "--phase", "v12_history_train", "--dry-run")
        run("orchestrate_select_certify_v12.py", "--phase", "v12_candidate_train_calibrate_select_certify", "--dry-run")
        run("orchestrate_select_certify_v12.py", "--phase", "v12_counterfactual_deploy_audit_after_global_barrier", "--dry-run")
    print("v12 preparation completed. No GPU job was launched.")


if __name__ == "__main__":
    main()
