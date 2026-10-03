# Data and Evaluation Protocol

## Cohorts

The empirical corpus has 265 directed pairs across Ethereum, BSC, Polygon Bor,
and Celo. Its code-localization subset has 224 pairs. Each dataset record links
a source artifact and a target artifact; issues are traced to resolving changes.
Stable record IDs are retained rather than renumbered after exclusions.

The statement evaluation contains 184 pairs: 39 file-level pairs have no
evaluable positive existing AST unit and one further pair has an unavailable
snapshot comparison. There are 168,593 units and 3,570 positive units.

`file_correspondences.json` contains the reference source-file/target-file
mapping. Every reference counterpart is counted independently for Hit@1/3/5
and MRR@5. A miss within five predictions contributes zero reciprocal rank.

`statement_labels.json.gz` contains `pair_id`, `group_keys`, and the complete
unit census. Unit IDs encode path, byte span and AST kind. Labels refer to the
unit's own tokens, not nested child statements. Only edits and deletions of
existing statements are positive; pure insertions are outside this task.

## Inference artifacts

`file_inputs.json.gz` includes the complete target pre-change file tree,
source snapshot identifiers, source-only path masks, fixed retrieval results,
and graph/full or graph-free QA plans. It does not contain target change labels.

`statement_inputs.json.gz` contains the target pre-change AST census and
label-free, bounded QA payloads for the full and graph-free configurations.
Statement localization is conditional on the known target changed files.
`statement_graph_seeds.json.gz` preserves the initial graph-selected evidence
used by the constrained-expansion stage. These intermediate artifacts make the
measured candidate sets reproducible without requiring private metadata caches.

The API runner reads inference inputs, not evaluation labels. A scored but
unselected candidate receives zero; an unretrieved unit receives null. Both
remain negative predictions unless a non-null score exceeds the threshold.
If multiple contexts score a unit, its maximum score is retained.

## Evaluation

The evaluator maximizes calibration micro-F1 with strict `score > p` decisions.
Five folds, seed 2027, keep all pairs connected through shared source OR target
changes together. Each method fits its threshold on four folds and predicts
the fifth. Precision, recall, F1 and MCC pool held-out confusion counts.
There is no Top-K statement evaluation or line-distance tolerance.
Full-data deployment thresholds are reported separately and are not test scores.

## Practical cases

The 14 `CASE-001` through `CASE-014` records are historical confirmation
artifacts outside the benchmark. Their archived localization outputs use the
historical practical-search protocol and must not be pooled into the current
184-pair AST binary evaluation. Shared or imported Git commits may retain
upstream timestamps; these timestamps alone do not prove when downstream
maintainers adopted a change.
