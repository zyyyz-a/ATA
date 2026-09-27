# Artifact scope

The public release has two purposes: it records what was frozen before deployment outcomes were opened, and it provides the aggregate evidence needed to inspect the reported claims.

## Included

- prospective configuration, manifest, action lock, deployment barrier, and execution lock;
- code used to fit and evaluate candidate actions;
- aggregate decision-level and policy-level results;
- split-half reliability and measurement-floor outputs;
- post-outcome diagnostic summaries used in the paper;
- source values and code for the quantitative evidence figure.

## Excluded

- raw text and original record identifiers;
- prediction sidecars containing per-record model outputs;
- model checkpoints and cached pretrained weights;
- access tokens, server configuration, and local paths;
- licensed dataset files that must be obtained from their original providers.

The exclusion of per-record sidecars prevents a complete local recomputation of the sensor analysis from this public repository alone. The aggregate outputs, analysis code, lock records, and release checks remain available for inspection. Qualified reviewers can compare regenerated outputs against these records after obtaining the source datasets and models.
