# Anonymous Supplement for Adaptation Transport Auditing

This archive contains the protocol and evidence records used by the anonymous submission. It intentionally excludes raw text, original record identifiers, model checkpoints, credentials, server addresses, and author-identifying metadata.

## Directory map

- `protocol/`: frozen configurations, manifests, execution locks, the post-failure recovery amendment, and the prospective S6 action/barrier locks.
- `data_manifests/`: post-quarantine manifests for Spatiotemporal News, Swiss Judgment, and Jigsaw. They report chronological role counts and content hashes without redistributing text.
- `code/`: the legacy closure implementation plus the S6 policy audit, integrity check, split-half reliability gate, and attenuation-budget analysis.
- `analysis/`: frozen aggregate summaries, exact tail/control analysis, the S6 policy record, and sensor-gate outputs in JSON, CSV, and Markdown.
- `figure_source/`: the quantitative-figure generator, plotted source values, vector output, and collision-audit report.

## Evidence roles

S1--S4 are post-outcome diagnostic, comparison, or sensitivity suites. S5 is prospectively locked but has 100% authorization coverage; it tests temporal transport rather than selective authorization. S6 is a separate prospective suite on two previously untouched streams. Its candidate set, selection and LCB authorization rules, matched-coverage controls, action locks, and all-action deployment barrier were frozen before deployment outcomes were opened. The sensor-gate analysis uses S6 predictions after outcomes opened; its block resolutions, split, 0.70 threshold, and stopping rule were fixed before reliability was calculated, so it is a pre-specified stopping analysis rather than a second prospective test.

The measurement-floor output propagates per-record variance through the pooled residual coordinates by a first-order delta method. It is a measurement-budget diagnostic under parallel half-block and locally stable variance assumptions, not a prospective power guarantee.

The term `harm` in machine-readable field names denotes aggregate task-performance loss under the manuscript's declared margin. It is not a measure of individual or social harm.

## Integrity and recovery

The execution lock binds the configuration and manifest digests before deployment evaluation. A subset of DeBERTa full-tuning jobs failed because re-entrant gradient checkpointing produced an incompatible backward graph. The recovery amendment changed checkpoint execution to `use_reentrant=false`; data roles, seeds, model revisions, checkpoints, action hyperparameters, precision, and evaluation code were unchanged. Original failure records were retained.

## Reproduction scope

The archive supports inspection of the chronology, action inventory, selection/calibration rules, frozen lock records, tail statistics, S6 policy summaries, and sensor-gate calculations. Raw corpora and pretrained model weights must be obtained from their original licensed sources. Hardware-specific path prefixes in the legacy draft job manifest have been replaced by `PROJECT_ROOT/` for anonymity and portability; content digests are unchanged. Internal experiment-version strings remain inside immutable lock payloads and analysis scripts because rewriting them would invalidate provenance; the manuscript refers to the corresponding evidence suite as S6.
