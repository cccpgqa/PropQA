# PropQA Replication Package

This package accompanies the paper **Change Propagation in EVM-Compatible
Blockchains: Empirical Evidence and LLM-Guided Localization**.

PropQA is an LLM-guided repository question-answering approach for localizing
the impact of an upstream software change in a downstream blockchain client.
It represents repository file trees and source-code ASTs as graphs and returns
ranked candidate files and existing target statements that may require editing
or deletion.

For each Graph action, the executor first retrieves a bounded target subgraph
containing file-tree, AST, symbol, and pre-fix code evidence. The LLM answers
the action-specific question using identifiers from that evidence. A second
LLM round, `FinalizeResults`, reconciles all action answers and returns the
final ranking. The released result files retain these action answers and final
QA traces for auditing.

## Package contents

```text
PropQA/
|-- data/
|   |-- empirical_pairs_266.json
|   |-- detection_pairs_225.json
|   `-- practical_propagation_cases_14.json
|-- docs/
|   `-- DATA_SCHEMA.md
|-- results/
|   |-- baselines/
|   |-- file_localization.json
|   |-- statement_localization.json
|   `-- practical_localization_results_14.json
|-- scripts/
|   |-- analyze_empirical_propagation.py
|   |-- rerun_file_llm_v3.py
|   |-- rerun_statement_existing_code_v3.py
|   |-- search_practical_propagation.py
|   |-- rerun_prefiltered_baselines.py
|   |-- rerun_pure_nicad_only.py
|   `-- validate_package.py
|-- outputs/                 # Compatibility inputs used by original scripts
|-- .env.example
`-- requirements.txt
```

The `scripts/` directory also contains the local dependency modules imported
by the entry-point scripts. The compatibility files under `outputs/` preserve
the paths expected by the original experiment implementation; the curated
public datasets are the clearly named files under `data/`.

## Datasets

### Empirical dataset

`data/empirical_pairs_266.json` contains 266 manually validated, URL-backed
propagation pairs among Ethereum, BSC, Polygon Bor, and Celo. Each directed
pair connects a source artifact to a target artifact addressing the same
software issue. Artifacts may be pull requests, commits, or issues resolved by
merged pull requests or commits.

### Detection dataset

`data/detection_pairs_225.json` is the code-localization subset derived from
the empirical dataset. It contains 225 pairs with attributable target changed
files. These pairs yield 767 curated source-file--target-file records. Of the
225 pairs, 199 contain existing target statements that can be labeled as
edited or deleted, yielding 5,486 statement-level ground-truth records.
Newly inserted statements are not part of the statement-localization task
because they do not exist in the target pre-fix revision.

### Practical validation cases

`data/practical_propagation_cases_14.json` contains 14 manually validated
real-world propagation cases outside the detection benchmark. Given each
upstream change and the downstream repository state at the upstream merge
time, PropQA localized the affected target files and, where evaluable,
existing target statements. Downstream developers subsequently committed
identical or semantically similar changes at these locations, providing
retrospective evidence that the localized code was genuinely affected.
These records are released as confirmation artifacts and are not supplied as
search queries to the practical-history pipeline.

| ID | Source | Target | File coverage | Statement coverage |
|---|---|---|---:|---:|
| CASE-001 | [Geth #14718](https://github.com/ethereum/go-ethereum/pull/14718) | [BSC dfd07624](https://github.com/bnb-chain/bsc/commit/dfd076244dd0c2d809f9dd0080feab167ba9560c) | 2/2 | 8/9 |
| CASE-002 | [Geth #31394](https://github.com/ethereum/go-ethereum/pull/31394) | [BSC a5d39a4e](https://github.com/bnb-chain/bsc/commit/a5d39a4ec8cdc7260be6ea300076762c18c78c73) | 1/1 | 7/7 |
| CASE-003 | [Geth #25289](https://github.com/ethereum/go-ethereum/pull/25289) | [BSC e9a04cca](https://github.com/bnb-chain/bsc/commit/e9a04cca302a9e122ca867d73b1ead30388d4c22) | 1/1 | 1/1 |
| CASE-004 | [Geth #23312](https://github.com/ethereum/go-ethereum/pull/23312) | [Celo 62ad17fb](https://github.com/celo-org/celo-blockchain/commit/62ad17fb0046243255048fbf8cb0882f48d8d850) | 2/2 | 8/9 |
| CASE-005 | [Geth #15131](https://github.com/ethereum/go-ethereum/pull/15131) | [Celo 216e5848](https://github.com/celo-org/celo-blockchain/commit/216e584899ed522088419438c9c605a20b5dc9ae) | 3/3 | 52/57 |
| CASE-006 | [Geth #21232](https://github.com/ethereum/go-ethereum/pull/21232) | [Celo bcb30874](https://github.com/celo-org/celo-blockchain/commit/bcb308745010675671991522ad2a9e811938d7fb) | 6/6 | 66/98 |
| CASE-007 | [Geth #17118](https://github.com/ethereum/go-ethereum/pull/17118) | [Celo 83e2761c](https://github.com/celo-org/celo-blockchain/commit/83e2761c3a13524bd5d6597ac08994488cf872ef) | 5/5 | 22/22 |
| CASE-008 | [Geth #27887](https://github.com/ethereum/go-ethereum/pull/27887) | [Celo #2280](https://github.com/celo-org/celo-blockchain/pull/2280) | 1/1 | 2/2 |
| CASE-009 | [Geth #27702](https://github.com/ethereum/go-ethereum/pull/27702) | [Celo #2284](https://github.com/celo-org/celo-blockchain/pull/2284) | 3/3 | 4/14 |
| CASE-010 | [Geth #22919](https://github.com/ethereum/go-ethereum/pull/22919) | [Celo 59f259b0](https://github.com/celo-org/celo-blockchain/commit/59f259b058b85eea38cd2686051a9076abb1e712) | 2/2 | 0/3 |
| CASE-011 | [Geth #23225](https://github.com/ethereum/go-ethereum/pull/23225) | [Celo 2faf796d](https://github.com/celo-org/celo-blockchain/commit/2faf796d2a502ef6d3c02681a649bd3f41999ccc) | 3/3 | 3/3 |
| CASE-012 | [Geth #22957](https://github.com/ethereum/go-ethereum/pull/22957) | [Celo ee35ddc8](https://github.com/celo-org/celo-blockchain/commit/ee35ddc8fdf5fe12f42cac3bd7a40d8fe7a384f2) | 1/1 | N/A |
| CASE-013 | [Geth #21427](https://github.com/ethereum/go-ethereum/pull/21427) | [Bor 8f240978](https://github.com/0xPolygon/bor/commit/8f24097836b7e9265b73cfcdb586cd967e63d656) | 1/1 | 9/9 |
| CASE-014 | [Geth #20860](https://github.com/ethereum/go-ethereum/pull/20860) | [Bor 228a2970](https://github.com/0xPolygon/bor/commit/228a2970566261df7f86764ca94cb6a670500064) | 1/1 | 15/15 |

File and statement coverage in this table report the number of ground-truth
elements returned by PropQA divided by the number of labeled elements. The
statement task is unavailable for `CASE-012` because its target patch
contains no evaluable existing statement.

## Environment

The experiments were run with Python 3.11. A CUDA-enabled PyTorch build is
optional but substantially accelerates code2vec training and inference.

```bash
conda create -n propqa python=3.11
conda activate propqa
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and set the required credentials. Do not commit
API keys. A GitHub token is required when repository objects are not already
present in `.cache/github`. `API_KEY`, `BASE_URL`, and `MODEL` configure one
OpenAI-compatible LLM endpoint. To compare multiple LLMs, run the experiment
separately with a different configuration and output file for each model.

## Reproduction

Run all commands from the package root.

### 1. Validate packaged artifacts

```bash
python scripts/validate_package.py
```

### 2. Reproduce the empirical analysis

```bash
python scripts/analyze_empirical_propagation.py \
  --input data/empirical_pairs_266.json \
  --output-dir results/empirical_reproduced
```

### 3. Reproduce file-level localization

```bash
python scripts/rerun_file_llm_v3.py \
  --dataset data/detection_pairs_225.json \
  --output results/file_localization_reproduced.json
```

The script evaluates the configured LLM over 767 source-file queries and
reports Hit@1, Hit@3, Hit@5, and MRR@5. Use `--max-pairs` for a smoke test
before launching the full API-backed run.

### 4. Reproduce statement-level localization

```bash
python scripts/rerun_statement_existing_code_v3.py \
  --dataset data/detection_pairs_225.json \
  --output results/statement_localization_reproduced.json
```

This experiment evaluates only edited and deleted statements in the 199
evaluable pairs. It reports Precision@100, Coverage@100, F1@100, and MRR@100.

### 5. Reproduce path-prefiltered baselines

```bash
python scripts/rerun_prefiltered_baselines.py \
  --dataset data/detection_pairs_225.json \
  --mode all \
  --output results/baselines/path_prefiltered_reproduced.json
```

The released code2vec checkpoint is expected at
`.cache/go_code2vec/geth_go_code2vec.pt`. If it is absent, train or place the
Go-specific checkpoint there before running code2vec experiments. Open-NiCad
must be installed separately when reproducing the external clone detector;
the package retains the normalized local comparison used to assemble the
paper tables.

### 6. Search repository history for practical propagation cases

The practical pipeline starts from upstream merged PRs rather than known
propagation pairs. For each source PR, it checks out each target repository at
the source merge time, uses PropQA to localize affected files and statements,
and searches later commits touching those locations. Candidate commits are
ranked using localized-file overlap, patch similarity, change-description
similarity, and statement evidence. The resulting queue requires manual
confirmation that both changes address the same software issue.

Run the offline smoke test first:

```bash
python scripts/search_practical_propagation.py --smoke-test
```

Then run a bounded history search, for example:

```bash
python scripts/search_practical_propagation.py \
  --source-repo ethereum/go-ethereum \
  --targets bnb-chain/bsc,0xPolygon/bor,celo-org/celo-blockchain \
  --since 2024-01-01T00:00:00Z \
  --until 2024-12-31T23:59:59Z \
  --history-days 365 \
  --max-sources 20 \
  --resume \
  --output results/practical_history_candidates.json
```

Alternatively, `--source-changes` accepts a JSON list of objects containing
`repo` and `number`, allowing an exact set of upstream PRs to be reproduced.
The full run requires `GITHUB_TOKEN` and the configured PropQA LLM endpoint.

## Headline results

Using DeepSeek-v4-pro, PropQA reaches file-level Hit@1 of 0.857 and MRR@5 of
0.867 over 767 file-pair records. At statement level, it reaches Precision@100
of 0.424, Coverage@100 of 0.901, F1@100 of 0.531, and MRR@100 of 0.959 over 199
evaluable pairs. The complete machine-readable rows and QA trajectories are
under `results/`.

## Notes on external state

Full reproduction requires repository revisions referenced by the datasets,
GitHub metadata, the configured LLM services, and the Go-specific code2vec
checkpoint. The scripts cache GitHub responses in `.cache/github`, repository
checkouts in `.cache/git_repos`, and model vectors in `.cache/go_code2vec`.
These caches, credentials, and third-party repositories are intentionally not
included in the package.
