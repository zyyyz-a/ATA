# Adaptation Transport Auditing

This repository contains the code and frozen audit records for studying a specific question in temporal NLP: when an update is approved on one chronological window, does the resulting deployment action remain useful on the next?

Adaptation Transport Auditing (ATA) separates candidate fitting, decision fitting, action selection, authorization, and deployment audit. The repository is organized around that separation. It is an audit package, not a new language-model backbone and not a universal safety certificate.

## Main results represented here

The prospective suite contains two previously untouched temporal streams, eight origins per stream, and three seeds per origin.

| Stream | Authorized cells | Coverage | System gain | Worst served cell |
| --- | ---: | ---: | ---: | ---: |
| arXiv | 19/24 | 79.2% | +2.139 pp | +1.548 pp |
| PatClass | 7/24 | 29.2% | +0.523 pp | +0.069 pp |

Neither stream has a negative served cell under the frozen policy. The selected portfolio does not improve over a fixed full-tuning gate, so the repository does not claim a portfolio advantage.

The residual-sensor analysis is also negative. At 32 temporal blocks, residual direction and scale fail the frozen split-half reliability criterion. Of 96 trajectory-sensor combinations, 53 have no positive noise-corrected signal estimate. Among the 43 identifiable cases, the median estimated requirement for reliability 0.70 is 1,987 records per block, 25.7 times the available budget. This is a measurement-budget diagnostic, not a prospective power guarantee.

## Repository layout

- `src/ata_nlp/`: model, action, and locking utilities used by the prospective pipeline.
- `scripts/`: preparation, validation, execution, action locking, and analysis entry points.
- `configs/`, `manifests/`, `locks/`: the prospective design and its frozen records.
- `artifact/`: the submission-facing audit artifact, including aggregate results, legacy protocol records, analysis code, and figure sources.
- `tests/`: release checks that run without downloading datasets or model weights.
- `docs/`: data-access, provenance, and reproduction notes.

The files under `artifact/protocol/` and the corresponding frozen code snapshots are preserved byte-for-byte. Internal version strings remain because changing them would break provenance.

## Quick verification

The release check uses only the Python standard library:

```bash
python scripts/verify_release.py
python -m unittest discover -s tests -v
```

It checks the frozen action barrier, the two prospective policy rows, the measurement-floor totals, and common credential or local-path patterns.

## Environment

Static verification needs Python 3.10 or newer. Re-running the analyses requires the packages in `requirements.txt`. Model fitting requires a CUDA-enabled PyTorch installation; install PyTorch for the target system before installing the remaining dependencies.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

The complete training replay also requires the original licensed datasets and pretrained checkpoints. They are not redistributed here. See `docs/DATA_ACCESS.md` and `docs/REPRODUCIBILITY.md` before running GPU jobs.

## Evidence boundary

Suites S1-S4 are post-outcome diagnostics. S5 is prospectively locked but authorizes every action. S6 is the selective prospective suite. Deployment labels never select or modify an S6 action. The residual-sensor study is a pre-specified stopping analysis performed after S6 outcomes opened; it is not a second prospective deployment test.

No raw text, original record identifier, model checkpoint, credential, server address, or author metadata is included in the audit artifact.

## License

Code in this repository is released under the MIT License. Dataset names and metadata remain subject to their original licenses and terms of use.
