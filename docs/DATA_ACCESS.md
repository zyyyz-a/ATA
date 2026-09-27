# Data access

This repository does not redistribute source text. Each dataset must be obtained from its original provider and used under that provider's license and terms.

The experiments use public temporal text sources, including Jigsaw toxicity data, MultiEURLEX, Swiss Judgment Prediction, arXiv metadata, and PatClass. Dataset-specific revisions and hashes are recorded in the frozen manifests and locks.

Before running a replay:

1. obtain the source dataset from its documented provider;
2. verify the license and terms for the intended use;
3. resolve the recorded dataset revision;
4. reproduce the chronological role assignment;
5. run duplicate and cross-role leakage checks;
6. keep deployment labels closed until the action lock and barrier have been written.

Local data directories are ignored by Git. Do not add raw text, original identifiers, access tokens, or model checkpoints to the repository.
