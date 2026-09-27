# Reproduction guide

## Static audit

Static verification does not need the source datasets or a GPU:

```bash
python scripts/verify_release.py
python -m unittest discover -s tests -v
```

This confirms the release structure and the numerical claims represented in the packaged aggregate tables.

## Prospective replay

A full replay requires the original datasets, pretrained checkpoints, and a CUDA-enabled PyTorch environment. Follow the chronological order encoded by the pipeline:

1. resolve dataset and backbone revisions;
2. materialize chronological roles without opening deployment labels;
3. fit history models and candidate actions;
4. select the nominee and compute its authorization statistic;
5. serialize all actions and write the pre-deployment barrier;
6. open deployment outcomes once;
7. run the audit and aggregate analyses.

The scripts retain their historical file names because those names occur in the execution locks. Start with `scripts/validate_select_certify_v12.py` and inspect the frozen files in `configs/`, `manifests/`, and `locks/` before launching GPU work.

The release intentionally does not provide a one-command script that silently downloads data and opens deployment outcomes. The separation between preparation, action locking, and outcome audit is part of the experimental design.
