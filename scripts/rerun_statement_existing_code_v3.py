#!/usr/bin/env python3
"""Re-evaluate statement localization for edited and deleted target code only."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from no_leak_ground_truth import statement_ground_truth_from_files
from propagation_detector import GitHubClient
from rerun_prefiltered_baselines import code2vec_prefilter, open_nicad_prefilter
from localization_baselines import ExhaustiveCode2Vec, local_nicad
from run_agent_strategy_matrix import map_candidates
from run_block_first_statement_experiment import metrics as statement_metrics
from run_graphqa_action_enhancement_experiment import direct_intent_ast_blocks, expand_direct_blocks
from run_llm_in_loop_hard_cases import prepare_state
from run_no_leak_five_module_pipeline import normalize_target_oracle_to_dataset_files, split_detector_state
from run_qa_mcts_small_sample import PropagationQAMCTS, load_dotenv


BASELINES = ("Open-NiCad", "code2vec", "Open-NiCad+Path", "code2vec+Path")
METHODS = BASELINES + ("PropQA",)


class LLMClient:
    """Client for one user-configured OpenAI-compatible LLM endpoint."""

    def __init__(
        self,
        name: str,
        api_key: str,
        base_url: str,
        model: str,
        timeout: int = 90,
        fallback_base_urls: list[str] | None = None,
    ) -> None:
        self.name = name
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.base_urls = [self.base_url] + [
            url.rstrip("/")
            for url in (fallback_base_urls or [])
            if url and url.rstrip("/") != self.base_url
        ]
        self.used_base_url = self.base_url
        self.model = model
        self.timeout = timeout
        self.last_error = ""

    def complete_json(self, system: str, payload: dict[str, Any]) -> dict[str, Any] | None:
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            "temperature": 0,
            "max_tokens": 4000,
        }
        if "deepseek" in f"{self.model} {self.base_url}".lower():
            body["response_format"] = {"type": "json_object"}
            body["thinking"] = {"type": "disabled"}
        errors = []
        for base_url in self.base_urls:
            url = base_url
            if not url.endswith("/chat/completions"):
                url += "/chat/completions"
            request = urllib.request.Request(
                url,
                data=json.dumps(body).encode("utf-8"),
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                method="POST",
            )
            for attempt in range(3):
                try:
                    with urllib.request.urlopen(request, timeout=self.timeout) as response:
                        data = json.loads(response.read().decode("utf-8"))
                    choice = (data.get("choices") or [{}])[0]
                    message = choice.get("message") or {}
                    content = message.get("content") or ""
                    parsed = extract_json(content)
                    if parsed is not None:
                        self.last_error = ""
                        self.used_base_url = base_url
                        self.base_urls = [base_url] + [item for item in self.base_urls if item != base_url]
                        return parsed
                    errors.append(
                        f"{base_url}: unparseable content={content[:80]!r} "
                        f"finish={choice.get('finish_reason')} reasoning_len={len(message.get('reasoning_content') or '')} "
                        f"usage={data.get('usage')}"
                    )
                except urllib.error.HTTPError as exc:
                    errors.append(f"{base_url}: HTTP {exc.code}")
                    if exc.code < 500 and exc.code != 429:
                        break
                except Exception as exc:
                    errors.append(f"{base_url}: {type(exc).__name__}: {exc}")
                time.sleep(1.5 * (attempt + 1))
        self.last_error = " | ".join(errors[-3:])
        return None


def extract_json(text: str) -> dict[str, Any] | None:
    text = text.strip()
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    for candidate in (text, text[text.find("{") : text.rfind("}") + 1]):
        if not candidate:
            continue
        try:
            value = json.loads(candidate)
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            pass
    return None


def deletion_side_blocks(
    state: dict[str, Any], runner: PropagationQAMCTS, target_paths: list[str]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    blocks, intents = direct_intent_ast_blocks(state, runner, target_paths, 2)
    intents = [intent for intent in intents if intent.get("kind") == "delete"]
    retained = []
    for block in blocks:
        evidence = [intent for intent in block.get("intent_evidence", []) if intent.get("kind") == "delete"]
        if not evidence:
            continue
        item = dict(block)
        item["intent_evidence"] = evidence
        item["actions"] = ["DeleteRegion"]
        retained.append(item)
    return retained, intents


def average(rows: list[dict[str, Any]]) -> dict[str, Any]:
    fields = ("precision", "coverage", "f1", "mrr")
    return {
        "pairs": len(rows),
        **{field: sum(float(row[field]) for row in rows) / len(rows) if rows else 0.0 for field in fields},
    }


def markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Statement Localization on Existing Affected Code",
        "",
        "Ground truth contains only edited and deleted statements; added-new statements and InsertAnchor are excluded.",
        "",
    ]
    lines += [
        "| Method | Pairs | Precision@100 | Coverage@100 | F1@100 | MRR@100 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in BASELINES + ("PropQA",):
        if method not in payload["summary"]:
            continue
        row = payload["summary"][method]
        lines.append(
            f"| {method} | {row['pairs']} | {row['precision']:.3f} | {row['coverage']:.3f} | "
            f"{row['f1']:.3f} | {row['mrr']:.3f} |"
        )
    lines += [
        "",
        f"- Existing-code ground-truth statements: {payload['ground_truth_counts']['all']}",
        f"- Edited statements: {payload['ground_truth_counts']['edited']}",
        f"- Deleted statements: {payload['ground_truth_counts']['deleted']}",
    ]
    return "\n".join(lines) + "\n"


def source_context(state: dict[str, Any], intents: list[dict[str, Any]]) -> dict[str, Any]:
    source = state["propagator"]
    return {
        "repository": source.repo,
        "title": (source.title or "")[:500],
        "body": (source.body or "")[:1000],
        "commit_message": (source.message or "")[:500],
        "changed_files": state.get("source_changed_paths", [])[:20],
        "deletion_side_intents": [
            {
                "intent_id": index,
                "source_path": intent.get("path"),
                "source_symbol": intent.get("symbol"),
                "operation": "edit-or-delete-existing-code",
                "old_statements": intent.get("texts", [])[:5],
            }
            for index, intent in enumerate(intents[:16])
        ],
    }


def block_code_preview(
    client: GitHubClient,
    state: dict[str, Any],
    block: dict[str, Any],
) -> str:
    path = str(block.get("path") or "")
    if not path:
        return ""
    content = client.file_at(state["target"].repo, state["target"].base_sha, path) or ""
    lines = content.splitlines()
    start = max(0, int(block.get("start") or 1) - 1)
    end = min(len(lines), max(start + 1, int(block.get("end") or start + 1)))
    return "\n".join(lines[start:end])[:1800]


def valid_ids(values: Any, upper_bound: int) -> list[int]:
    result = []
    for value in values or []:
        try:
            candidate_id = int(value)
        except (TypeError, ValueError):
            continue
        if 0 <= candidate_id < upper_bound and candidate_id not in result:
            result.append(candidate_id)
    return result


def llm_rerank_statements(
    client: GitHubClient,
    llm: LLMClient,
    state: dict[str, Any],
    blocks: list[dict[str, Any]],
    intents: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    top: int,
    prompt_limit: int = 120,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pool = candidates[:prompt_limit]
    candidate_payload = [
        {
            "id": index,
            "path": item.get("path"),
            "line": item.get("line"),
            "symbol": item.get("target_symbol"),
            "statement": (item.get("line_text") or "")[:140],
            "matched_source_statement": (item.get("matched_source_text") or "")[:140],
            "action": item.get("action"),
        }
        for index, item in enumerate(pool)
    ]
    block_payload = [
        {
            "block_id": index,
            "path": block.get("path"),
            "symbol": block.get("symbol"),
            "start_line": block.get("start"),
            "end_line": block.get("end"),
            "actions": block.get("actions", []),
            "target_pre_fix_code": block_code_preview(client, state, block),
        }
        for index, block in enumerate(blocks[:24])
    ]
    shared_context = {
        "source_change": source_context(state, intents),
        "target_repository": {
            "repository": state["target"].repo,
            "candidate_ast_blocks": block_payload,
        },
        "candidate_statements": candidate_payload,
    }
    action_payload = {
        "task": "Answer each named statement-level repository-graph action question.",
        **shared_context,
        "action_questions": {
            "ConstructSourceIntent": "For each source intent, what existing behavior is edited or deleted and which symbols or operations identify it?",
            "LocateAffectedBlock": "For each source intent, which target AST block IDs contain corresponding affected behavior?",
            "IdentifyDeletionRegions": "Which target AST block IDs contain functions, blocks, APIs, or references that should be removed or substantially changed?",
            "ShowCode": "After reading the target pre-fix block and statement code, which candidate statement IDs contain the affected behavior?",
        },
        "output_schema": {
            "source_intents": [
                {"intent_id": 0, "interpretation": "concise intent", "operation": "edit-or-delete"}
            ],
            "LocateAffectedBlock": [{"intent_id": 0, "block_ids": [0]}],
            "IdentifyDeletionRegions": [{"intent_id": 0, "block_ids": [0]}],
            "ShowCode": {"candidate_statement_ids": [0, 1], "evidence": "brief explanation"},
        },
        "constraints": [
            "Use only supplied intent, block, and candidate-statement IDs.",
            "Use only target pre-fix code and do not invent target change metadata.",
            "Only existing statements requiring editing or deletion are in scope.",
            "Return strict JSON only.",
        ],
    }
    action_system = (
        "You are the statement-level repository-graph QA agent in PropQA. Answer every named action "
        "using source intents and target pre-fix AST/code evidence. Do not output chain-of-thought. Return JSON only."
    )
    action_result = llm.complete_json(action_system, action_payload)

    clean_action_result: dict[str, Any] = {}
    if action_result:
        clean_action_result["source_intents"] = action_result.get("source_intents", [])[:16]
        for action in ("LocateAffectedBlock", "IdentifyDeletionRegions"):
            mappings = []
            for mapping in action_result.get(action, []):
                try:
                    intent_id = int(mapping.get("intent_id"))
                except (TypeError, ValueError):
                    continue
                if 0 <= intent_id < len(intents):
                    mappings.append(
                        {
                            "intent_id": intent_id,
                            "block_ids": valid_ids(mapping.get("block_ids"), len(block_payload)),
                        }
                    )
            clean_action_result[action] = mappings
        show_code = action_result.get("ShowCode") or {}
        clean_action_result["ShowCode"] = {
            "candidate_statement_ids": valid_ids(
                show_code.get("candidate_statement_ids"), len(candidate_payload)
            ),
            "evidence": str(show_code.get("evidence") or "")[:400],
        }

    finalize_payload = {
        "task": (
            "Finalize the ranked target pre-fix statements affected by the source change. "
            "Only edited or deleted existing code is in scope."
        ),
        **shared_context,
        "action_answers": clean_action_result,
        "output_schema": {
            "ranked_candidate_ids": [0, 1],
            "relevant_candidate_ids": [0, 1],
            "brief_rationale": "one short evidence-based sentence",
        },
        "constraints": [
            "Use only integer IDs present in candidate_statements.",
            "Reconcile the answers from all statement-level graph actions.",
            "Prioritize old target statements that should be edited or deleted.",
            "Return strict JSON only and at most 100 ranked IDs.",
        ],
    }
    finalize_system = (
        "You are the FinalizeResults agent in PropQA. Rank affected existing target statements from "
        "the completed graph-action answers. Do not output chain-of-thought. Return JSON only."
    )
    result = llm.complete_json(finalize_system, finalize_payload)
    ordered_ids = []
    if result:
        for key in ("ranked_candidate_ids", "relevant_candidate_ids"):
            for value in result.get(key, []):
                try:
                    candidate_id = int(value)
                except (TypeError, ValueError):
                    continue
                if 0 <= candidate_id < len(pool) and candidate_id not in ordered_ids:
                    ordered_ids.append(candidate_id)
    selected = [pool[index] for index in ordered_ids]
    selected_keys = {(item.get("path"), item.get("line")) for item in selected}
    for item in candidates:
        key = (item.get("path"), item.get("line"))
        if key not in selected_keys:
            selected.append(item)
            selected_keys.add(key)
        if len(selected) >= top:
            break
    return selected[:top], {
        "provider": llm.name,
        "model": llm.model,
        "base_url": llm.used_base_url,
        "llm_used": action_result is not None and result is not None,
        "llm_error": llm.last_error,
        "selected_ids": ordered_ids[:100],
        "brief_rationale": (result or {}).get("brief_rationale", ""),
        "action_answers": clean_action_result,
        "action_qa_used": action_result is not None,
        "finalize_qa_used": result is not None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="outputs/URL_Results_detection_url_celo_v2.json")
    parser.add_argument("--output", default="outputs/statement_existing_code_v3.json")
    parser.add_argument("--cache-dir", default=".cache/github")
    parser.add_argument("--git-root", default=".cache/git_repos")
    parser.add_argument("--model", default=".cache/go_code2vec/geth_go_code2vec.pt")
    parser.add_argument("--vector-cache", default=".cache/go_code2vec/blob_vectors_ctx12_term8")
    parser.add_argument("--checkpoint", default=".cache/statement_existing_code_v3.pkl")
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument(
        "--propqa-only",
        action="store_true",
        help="Skip unchanged clone/code2vec baselines and run only action-based PropQA.",
    )
    args = parser.parse_args()

    load_dotenv()
    pairs = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    if args.max_pairs:
        pairs = pairs[: args.max_pairs]
    client = GitHubClient(Path(args.cache_dir), token=os.environ.get("GITHUB_TOKEN"))
    runner = PropagationQAMCTS(client, max_nodes=8, top_files=20, top_statements=100)
    code2vec = None if args.propqa_only else ExhaustiveCode2Vec(
        Path(args.model), Path(args.git_root), Path(args.vector_cache)
    )
    llm = LLMClient(
        "PropQA",
        os.environ.get("API_KEY", ""),
        os.environ.get("BASE_URL", ""),
        os.environ.get("MODEL", ""),
    )
    if not llm.api_key or not llm.base_url or not llm.model:
        raise RuntimeError("API_KEY, BASE_URL, and MODEL must be configured")
    checkpoint = Path(args.checkpoint)
    if checkpoint.exists() and not args.max_pairs:
        saved = pickle.loads(checkpoint.read_bytes())
        completed = int(saved["completed"])
        rows = saved["rows"]
        details = saved["details"]
        counts = saved["counts"]
        errors = saved["errors"]
        print(f"resuming after pair {completed}", flush=True)
    else:
        completed = 0
        methods = ("PropQA",) if args.propqa_only else METHODS
        rows = {method: [] for method in methods}
        details = []
        counts = {"all": 0, "edited": 0, "deleted": 0}
        errors = []
    snapshot_cache: dict[tuple[str, str], dict[str, str]] = {}

    for index, pair in enumerate(pairs, start=1):
        if index <= completed:
            continue
        try:
            setup = normalize_target_oracle_to_dataset_files(prepare_state(runner, pair))
            state, oracle = split_detector_state(setup)
            target_paths = list(oracle["target_changed_paths"])
            gt = [
                item
                for item in statement_ground_truth_from_files(oracle["target_files"])
                if item.kind in {"edited", "deleted"}
            ]
            counts["all"] += len(gt)
            counts["edited"] += sum(item.kind == "edited" for item in gt)
            counts["deleted"] += sum(item.kind == "deleted" for item in gt)
            if not gt:
                details.append(
                    {
                        "index": index,
                        "status": "excluded",
                        "reason": "no edited/deleted statement ground truth",
                    }
                )
                print(f"[{index}/{len(pairs)}] excluded no existing-code gt", flush=True)
                continue

            blocks, intents = deletion_side_blocks(state, runner, target_paths)
            budget = min(1200, max(100, 8 * len(intents), 6 * len(target_paths)))
            graphqa = expand_direct_blocks(
                client,
                state,
                blocks,
                intents,
                budget,
                6,
                use_insert_anchor=False,
            )
            predictions = {}
            if not args.propqa_only:
                assert code2vec is not None
                _nicad_files, nicad_statements = local_nicad(
                    state, runner, Path(args.git_root), 10, snapshot_cache
                )
                c2v_statements = code2vec.rank_statements(
                    state["propagator"].repo,
                    state["propagator"].merged_sha,
                    state["source_changed_paths"],
                    state["target"].repo,
                    state["target"].base_sha,
                    target_paths,
                    100,
                )
                nicad_path_files = open_nicad_prefilter(state, Path(args.git_root), 10, 80)
                c2v_path_files = code2vec_prefilter(state, code2vec, 10, 80)
                predictions.update(
                    {
                        "Open-NiCad": nicad_statements,
                        "code2vec": c2v_statements,
                        "Open-NiCad+Path": map_candidates(
                            runner, state, nicad_path_files, 200
                        )[:100],
                        "code2vec+Path": map_candidates(
                            runner, state, c2v_path_files, 200
                        )[:100],
                    }
                )
            ranked, llm_trace = llm_rerank_statements(
                client, llm, state, blocks, intents, graphqa, 100
            )
            predictions["PropQA"] = ranked

            detail = {
                "index": index,
                "ground_truth": len(gt),
                "metrics": {},
                "llm": llm_trace,
            }
            for method, predicted in predictions.items():
                result = statement_metrics(predicted, gt, 100, tolerance=2)
                rows[method].append(result)
                detail["metrics"][method] = result
            details.append(detail)
            print(f"[{index}/{len(pairs)}] ok gt={len(gt)}", flush=True)
        except Exception as exc:
            errors.append({"index": index, "reason": f"{type(exc).__name__}: {exc}"})
            print(f"[{index}/{len(pairs)}] error {type(exc).__name__}: {exc}", flush=True)
        if not args.max_pairs and (index % 5 == 0 or index == len(pairs)):
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_bytes(
                pickle.dumps(
                    {
                        "completed": index,
                        "rows": rows,
                        "details": details,
                        "counts": counts,
                        "errors": errors,
                    }
                )
            )

    payload = {
        "dataset": args.dataset,
        "task": "localize existing affected statements (edited and deleted only)",
        "llm": {
            "model": llm.model,
            "configured_base_url": llm.base_url,
            "used_base_url": llm.used_base_url,
        },
        "evaluated_pairs": len(rows["PropQA"]),
        "ground_truth_counts": counts,
        "summary": {method: average(method_rows) for method, method_rows in rows.items()},
        "details": details,
        "errors": errors,
    }
    output = Path(args.output)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    output.with_suffix(".md").write_text(markdown(payload), encoding="utf-8")
    checkpoint.unlink(missing_ok=True)
    print(f"wrote {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
