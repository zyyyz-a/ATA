# Provenance notes

The repository distinguishes readable source code from frozen provenance records.

Files in `src/` and `scripts/` are the prospective pipeline entry points. Files in `artifact/protocol/` and the frozen code snapshots under `artifact/code/` preserve the state used by the reported experiments. Their internal version labels are historical identifiers, not manuscript terminology.

The frozen records are not rewritten for style. Editing a lock payload or a bound code snapshot would invalidate its digest. Public-facing documentation therefore explains those names instead of renaming them inside immutable files.

The release verifier checks the action-lock binding, the reported prospective policy rows, the residual measurement totals, and common secret or machine-path patterns. It does not certify that a third party has reproduced model training.
