# PropQA Replication Package

This package accompanies **Change Propagation in EVM-Compatible Blockchains:
Empirical Evidence and LLM-Guided Localization**.

PropQA uses repository structure, AST evidence and LLM question answering to
localize the impact of an upstream change. File localization returns ranked
target paths. Statement localization identifies existing target statements
requiring edits or deletion using calibrated confidence thresholds.

## Released Artifacts

| Artifact | Description |
|---|---|
| `data/empirical_pairs.json` | 265 directed propagation pairs |
| `data/empirical_records.csv` | Fixed categories and timing records for empirical analysis |
| `data/detection_pairs.json` | 224 file-level propagation pairs |
| `data/file_correspondences.json` | Reference source-file/target-file mapping |
| `data/file_inputs.json.gz` | Source context, target trees, candidates and QA plans |
| `data/statement_inputs.json.gz` | Label-free AST evidence and QA plans for 184 pairs |
| `data/statement_labels.json.gz` | 168,593 exact AST units; 3,570 edited/deleted positives |
| `data/practical_propagation_cases_14.json` | 14 retrospective practical confirmation cases |
| `results/` | Published per-unit scores, file ranks and metrics |
| `models/go_code2vec.pt` | Go code2vec checkpoint used by the baselines |

See [data definitions](docs/DATA_SCHEMA.md) and [baseline setup](docs/BASELINES.md).

## Package Map

| Stage | Entry point | Output |
|---|---|---|
| Collect public repository records | `scripts/collect_github.py` | PRs, issues, reachable commits and optional patches/discussions |
| Initial pair mining | `scripts/filter_candidates.py` | Deduplicated URL-linked candidates and exclusions for review |
| Final labeled data | `data/` | Fixed benchmark inputs and separate reference labels |
| RQ4: localization; RQ5: ablations | `scripts/run_localization.py`, `scripts/run_baselines.py`, `scripts/evaluate.py` | Predictions and evaluation tables |
| RQ6: practical search | `scripts/search_practical_propagation.py`, `scripts/summarize_practical.py` | New review candidates or summaries of archived confirmations |

## Installation and Credentials

Use Python 3.11 and Git. Install dependencies with:

```bash
pip install -r requirements.txt
```

Only users' own credentials are used. Populate a local `.env` using
`.env.example`: `API_KEY`, `BASE_URL`, and `MODEL` configure one
OpenAI-compatible HTTPS endpoint. `GITHUB_TOKEN` is optional for authenticated
public metadata retrieval. No private token or provider key is distributed.
Run different models in separate output directories using the same interface.

## Collect and Initially Filter Repository Data

Preview collection without network access:

```bash
python scripts/collect_github.py
```

Collect the four study repositories and construct initial candidates:

```bash
python scripts/collect_github.py --include-diffs --include-discussions --execute
python scripts/filter_candidates.py --raw raw --output reproduced/mining
```

This can require many GitHub requests. For a small collection, use
`--repos ethereum/go-ethereum --kinds pr --max-pages 1`. Responses are cached;
`--refresh` updates them. PR and issue lists include open and closed records.
Commit collection covers the default branch unless `--ref` specifies another
reachable history. It does not enumerate unreachable commits. The collection
manifest records scope and pagination truncation. GitHub may omit patches or
cap very large file lists; inspect local Git diffs for such changes.

Filtering parses cross-repository URLs in titles, bodies and commit messages,
resolves locally available referenced identities, and removes duplicate pairs,
same-repository links, known unmerged PRs and target reverts. Missing metadata,
unverified merge status and unusual timestamps are flagged for inspection.
Issue timeline links are resolution candidates, not proof of resolution.
Inspect semantic relevance and both patches, and confirm issue resolutions
before accepting a candidate. These scripts do not automatically regenerate
the curated benchmark from arbitrary current repository data.

## RQ4/RQ5: Offline Verification and Result Reproduction

From the package root:

```bash
python scripts/validate_package.py
python scripts/test_release.py
python scripts/test_collection.py
python scripts/analyze_empirical.py
python scripts/evaluate.py --output reproduced/tables
```

The last command recomputes all benchmark and module-ablation metrics from
released scores/ranks, including grouped threshold selection. It checks the
results against the published metrics. This is offline metric reproduction,
not a new LLM inference run, and needs no credentials or network access.

## RQ4: Repeat LLM Inference

The package freezes the exact graph-retrieval evidence used in the benchmark,
including complete target file trees and statement candidate IDs. Re-executing
QA on these inputs avoids dependence on private metadata caches. Evaluation
labels are kept in separate files and are not sent to the API.

```bash
python scripts/run_localization.py --level file --output reproduced/file
python scripts/run_localization.py --level statement --output reproduced/statement
```

Without `--execute`, these commands print the pair and planned request counts.
To run the configured model:

```bash
python scripts/run_localization.py --level file --output reproduced/file --execute --max-requests 1500
python scripts/run_localization.py --level statement --output reproduced/statement --execute --max-requests 1000
python scripts/evaluate.py --predictions reproduced/file --output reproduced/file_metrics
python scripts/evaluate.py --predictions reproduced/statement --output reproduced/statement_metrics
```

Use `--max-pairs 1` and a separate output directory for a small API smoke test.
Validated responses and completed pairs are cached for resumption. Errors stop
the run rather than being interpreted as negative predictions. Repeated LLM
inference may differ from archived outputs even at temperature zero.

File QA has an action-answering round followed by an evidence consolidation
round. Statement QA uses similarity-constrained candidates and compact block
context, with at most 160 units per request. It evaluates exact AST units, not
Top-100 lists or nearby-line matches. Unretrieved positives count as misses.

The frozen graph seeds are also released. `rebuild_statement_candidates.py`
can recompute constrained expansion from these seeds and public Git snapshots:

```bash
python scripts/rebuild_statement_candidates.py --max-pairs 1 --execute
```

This verifies expansion, not a fresh reconstruction of the initial graph
retrieval. The benchmark API commands deliberately start from the released
intermediate evidence; they do not claim a raw-repository end-to-end rebuild.

## RQ4 Baselines and RQ5 Module Ablations

[Baseline instructions](docs/BASELINES.md) cover native NiCad-Go and code2vec,
plus file-level +Path variants. The actual code2vec checkpoint is included.
NiCad's external comparison engine must be built separately.
The Go adaptation source is included in `scripts/nicad_go.py`,
`scripts/go_code2vec.py` and `scripts/statement_vectors.py`.
Run `python scripts/smoke_go_baselines.py` for an offline implementation check.
The baseline guide lists all implementation files and dependencies.

For PropQA's two module controls, add `--variant without_graph` or
`--variant without_qa` to `run_localization.py`, using a different output
directory. Without Graph uses an independently selected, matched-size lexical
candidate set and code-only QA. Without QA uses source-to-target cosine over
eligible units without the full-method mask or LLM calls. Parsing and unit
definitions remain shared. Without QA requires Git snapshots at both levels.

## RQ6: Practical Repository-History Search

The practical utility starts with historical merged upstream PRs, localizes
candidate target code at the source merge time, and searches subsequent target
commits for matching changes. The 14 confirmation cases are not search inputs.
Results require human review and are not proof of exploitability.

```bash
python scripts/search_practical_propagation.py --smoke-test
python scripts/summarize_practical.py
python scripts/search_practical_propagation.py --source-repo ethereum/go-ethereum --targets bnb-chain/bsc,0xPolygon/bor,celo-org/celo-blockchain --since 2024-01-01T00:00:00Z --until 2024-12-31T23:59:59Z --max-sources 20 --history-days 365 --resume --output reproduced/practical_candidates.json
```

This preserves the practical-history protocol associated with the archived
cases. Its statement coverage is separate from the exact-AST binary benchmark
and must not be substituted for the 184-pair evaluation.
`results/practical_pilot_observations.json` retains the five-source, ten-target
inspection pilot, and `results/maintainer_feedback.json` records four reported
maintainer outcomes. These are archived observations, not newly rerun results
or evidence that a planned change has already been merged.

## Results

| Method | File Hit@1 | File MRR@5 | Statement Precision | Statement Recall | Statement F1 |
|---|---:|---:|---:|---:|---:|
| PropQA (DeepSeek) | 0.856 | 0.866 | 0.740 | 0.711 | 0.725 |
| PropQA (Qwen) | 0.859 | 0.871 | 0.839 | 0.666 | 0.742 |

Statement thresholds maximize calibration micro-F1 within five source/target-
grouped folds (seed 2027). Metrics pool held-out predictions. Statement results
assume known target changed files and are not end-to-end file-plus-statement
scores. Full baseline and ablation tables are generated by `evaluate.py`.

## Practical Confirmation Cases

The following source/target links document 14 historical cases outside the
benchmark. They are confirmation artifacts, not a prospective success rate.
Imported commits can preserve upstream dates; the links alone do not establish
a downstream adoption timestamp.

| ID | Source | Target |
|---|---|---|
| CASE-001 | [ethereum #14718](https://github.com/ethereum/go-ethereum/pull/14718) | [bnb-chain dfd07624](https://github.com/bnb-chain/bsc/commit/dfd076244dd0c2d809f9dd0080feab167ba9560c) |
| CASE-002 | [ethereum #31394](https://github.com/ethereum/go-ethereum/pull/31394) | [bnb-chain a5d39a4e](https://github.com/bnb-chain/bsc/commit/a5d39a4ec8cdc7260be6ea300076762c18c78c73) |
| CASE-003 | [ethereum #25289](https://github.com/ethereum/go-ethereum/pull/25289) | [bnb-chain e9a04cca](https://github.com/bnb-chain/bsc/commit/e9a04cca302a9e122ca867d73b1ead30388d4c22) |
| CASE-004 | [ethereum #23312](https://github.com/ethereum/go-ethereum/pull/23312) | [celo-org 62ad17fb](https://github.com/celo-org/celo-blockchain/commit/62ad17fb0046243255048fbf8cb0882f48d8d850) |
| CASE-005 | [ethereum #15131](https://github.com/ethereum/go-ethereum/pull/15131) | [celo-org 216e5848](https://github.com/celo-org/celo-blockchain/commit/216e584899ed522088419438c9c605a20b5dc9ae) |
| CASE-006 | [ethereum #21232](https://github.com/ethereum/go-ethereum/pull/21232) | [celo-org bcb30874](https://github.com/celo-org/celo-blockchain/commit/bcb308745010675671991522ad2a9e811938d7fb) |
| CASE-007 | [ethereum #17118](https://github.com/ethereum/go-ethereum/pull/17118) | [celo-org 83e2761c](https://github.com/celo-org/celo-blockchain/commit/83e2761c3a13524bd5d6597ac08994488cf872ef) |
| CASE-008 | [ethereum #27887](https://github.com/ethereum/go-ethereum/pull/27887) | [celo-org #2280](https://github.com/celo-org/celo-blockchain/pull/2280) |
| CASE-009 | [ethereum #27702](https://github.com/ethereum/go-ethereum/pull/27702) | [celo-org #2284](https://github.com/celo-org/celo-blockchain/pull/2284) |
| CASE-010 | [ethereum #22919](https://github.com/ethereum/go-ethereum/pull/22919) | [celo-org 59f259b0](https://github.com/celo-org/celo-blockchain/commit/59f259b058b85eea38cd2686051a9076abb1e712) |
| CASE-011 | [ethereum #23225](https://github.com/ethereum/go-ethereum/pull/23225) | [celo-org 2faf796d](https://github.com/celo-org/celo-blockchain/commit/2faf796d2a502ef6d3c02681a649bd3f41999ccc) |
| CASE-012 | [ethereum #22957](https://github.com/ethereum/go-ethereum/pull/22957) | [celo-org ee35ddc8](https://github.com/celo-org/celo-blockchain/commit/ee35ddc8fdf5fe12f42cac3bd7a40d8fe7a384f2) |
| CASE-013 | [ethereum #21427](https://github.com/ethereum/go-ethereum/pull/21427) | [0xPolygon 8f240978](https://github.com/0xPolygon/bor/commit/8f24097836b7e9265b73cfcdb586cd967e63d656) |
| CASE-014 | [ethereum #20860](https://github.com/ethereum/go-ethereum/pull/20860) | [0xPolygon 228a2970](https://github.com/0xPolygon/bor/commit/228a2970566261df7f86764ca94cb6a670500064) |

## Scope and External Dependencies

The release contains the final localization configurations and module controls.
No private API response logs or credentials are required. Raw Git repositories
and the NiCad executable are fetched/built by users when rerunning baselines;
offline metric reproduction does not require them. Frozen evidence includes
public code excerpts, which retain their upstream licensing obligations.
