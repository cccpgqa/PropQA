# Artifact Provenance

The release contains fixed benchmark labels and category assignments; running
the localization and evaluation commands does not relabel or reclassify data.
Source/target identities, patch metadata, exact AST identifiers and evaluation
folds are retained to support independent inspection.

Existing-statement labels were derived from AST/patch alignment, with
model-assisted review of ambiguous alignments. The package does not establish
new human inter-rater agreement statistics. File correspondences retain the
path/patch mapping used by the evaluation. Empirical summaries consume the
released category assignments instead of assigning categories at runtime.

The practical cases are retrospective validation artifacts, not evidence that
the tool was run before the historical target commits were authored. They do
not establish vulnerability exploitability or generated-patch correctness.

The release includes only the final benchmark configurations and their two
module controls. Provider credentials, private logs and unrelated development
artifacts are not distributed.
