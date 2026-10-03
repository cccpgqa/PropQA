# Baseline Reproduction

## NiCad-Go

This is a Go adaptation, not an official Go language plugin. The frontend
extracts Go functions with tree-sitter and applies blind identifier
normalization. Comparison uses the unmodified Open-NiCad `crossclones` engine
at commit `7a90d11795a7fe585282e25b7fa8d9964f202965`.
The maximum difference is 0.3 and fragment sizes are 5--2500 normalized rows.
No source changed-position alignment is applied.

Build the pinned upstream engine using `scripts/setup_nicad_core.ps1` on
Windows with Visual Studio C++ build tools. The script downloads public source
and its license into `.cache/tools/nicad-core`; no executable is distributed.
On another platform build the same upstream engine with its supported C tools
and pass its absolute path with `--engine`.

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_nicad_core.ps1
python scripts/run_baselines.py --level file --method nicad --engine .cache/tools/nicad-core/crossclones.exe --output reproduced/nicad_file --execute
python scripts/run_baselines.py --level statement --method nicad --engine .cache/tools/nicad-core/crossclones.exe --output reproduced/nicad_statement --execute
```

For the file-level +Path variant add `--path-prefilter` and use a new output
directory. The mask keeps 80 source-derived paths without changing similarity.
Statement confidence is the maximum clone similarity covering the unit's owned
lines. Unmatched units remain in the evaluation census.

## code2vec

`models/go_code2vec.pt` is the actual Go path-context checkpoint used for the
reported scores. Its training revision and hyperparameters are recorded in
`models/go_code2vec_training.json`; its SHA-256 is in the artifact manifest.
It is a Go-specific implementation of the code2vec architecture, not the
original Java-pretrained model. No API key is needed.

```bash
python scripts/run_baselines.py --level file --method code2vec --output reproduced/code2vec_file --execute
python scripts/run_baselines.py --level statement --method code2vec --output reproduced/code2vec_statement --execute
```

File vectors are normalized means of function vectors, ranked by cosine over
all target Go files. Add `--path-prefilter` for the file-level +Path variant.
Statement scores use the maximum source-to-target AST-unit cosine, clamped to
[0,1] and rounded to six decimals before threshold calibration. The +Path
variant is not used for the statement benchmark.

These commands fetch the frozen public Git revisions if missing. Full reruns
can require substantial disk space and CPU time. First omit `--execute` for a
plan, or run `--max-pairs 1` in a separate smoke-test output directory.

After a complete run, use `scripts/evaluate.py --predictions OUTPUT_DIR
--name METHOD --output EVALUATION_DIR`. Partial runs are not reported as full
benchmark results. Existing released scores can be evaluated entirely offline.
