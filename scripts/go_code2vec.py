#!/usr/bin/env python3
"""A compact Go-specific code2vec implementation for retrieval experiments.

The original tech-srl/code2vec release targets Java. This module keeps the
code2vec path-context encoder and attention aggregation, but extracts contexts
from Go with tree-sitter-go and trains a function-name prediction objective.
Retrieval uses only cosine similarity between learned code vectors.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import subprocess
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn
from tree_sitter import Language, Node, Parser
import tree_sitter_go


TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+|[^\s]")
NAME_RE = re.compile(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)|\d+")
FUNCTION_TYPES = {"function_declaration", "method_declaration"}
LEAF_TYPES = {
    "identifier",
    "field_identifier",
    "type_identifier",
    "package_identifier",
    "int_literal",
    "float_literal",
    "imaginary_literal",
    "rune_literal",
    "raw_string_literal",
    "interpreted_string_literal",
    "true",
    "false",
    "nil",
}


def parser() -> Parser:
    return Parser(Language(tree_sitter_go.language()))


def split_name(name: str) -> list[str]:
    parts = []
    for segment in re.split(r"[_\W]+", name):
        parts.extend(match.group(0).lower() for match in NAME_RE.finditer(segment))
    return [part for part in parts if part]


def git_output(git_dir: Path, *args: str, timeout: int = 60) -> bytes:
    result = subprocess.run(
        ["git", f"--git-dir={git_dir}", *args],
        check=True,
        capture_output=True,
        timeout=timeout,
    )
    return result.stdout


def git_head(git_dir: Path) -> str:
    return git_output(git_dir, "rev-parse", "HEAD").decode().strip()


def git_go_paths(git_dir: Path, sha: str) -> list[str]:
    data = git_output(git_dir, "ls-tree", "-r", "--name-only", sha, timeout=180)
    return [
        path
        for path in data.decode("utf-8", errors="replace").splitlines()
        if path.endswith(".go") and not path.endswith(("_test.go", ".pb.go", "_generated.go"))
    ]


def git_file(git_dir: Path, sha: str, path: str) -> str:
    try:
        return git_output(git_dir, "show", f"{sha}:{path}", timeout=30).decode(
            "utf-8", errors="replace"
        )
    except Exception:
        return ""


def node_text(source: bytes, node: Node) -> str:
    return source[node.start_byte : node.end_byte].decode("utf-8", errors="replace")


def function_name(source: bytes, node: Node) -> str:
    child = node.child_by_field_name("name")
    return node_text(source, child) if child else ""


def walk(node: Node) -> Iterable[Node]:
    yield node
    for child in node.children:
        yield from walk(child)


def terminal_token(source: bytes, node: Node) -> str:
    text = node_text(source, node).strip()
    if node.type.endswith("literal") and node.type not in {"true", "false", "nil"}:
        return f"<{node.type}>"
    if len(text) > 80:
        text = text[:80]
    return text.lower() or f"<{node.type}>"


def ancestors(node: Node, root: Node) -> list[Node]:
    out = [node]
    current = node
    while current.id != root.id and current.parent is not None:
        current = current.parent
        out.append(current)
    return out


def ast_path(left: Node, right: Node, root: Node, max_length: int) -> str | None:
    left_anc = ancestors(left, root)
    right_anc = ancestors(right, root)
    right_pos = {node.id: index for index, node in enumerate(right_anc)}
    lca_left = lca_right = None
    for index, node in enumerate(left_anc):
        if node.id in right_pos:
            lca_left = index
            lca_right = right_pos[node.id]
            break
    if lca_left is None or lca_right is None:
        return None
    length = lca_left + lca_right
    if length > max_length:
        return None
    up = [node.type + "^" for node in left_anc[:lca_left]]
    middle = [left_anc[lca_left].type]
    down = [node.type + "v" for node in reversed(right_anc[:lca_right])]
    return "|".join(up + middle + down)


def extract_function_contexts(
    source_text: str,
    max_contexts: int = 100,
    max_path_length: int = 8,
    max_path_width: int = 30,
    max_terminals: int = 60,
) -> list[dict[str, Any]]:
    source = source_text.encode("utf-8", errors="replace")
    tree = parser().parse(source)
    functions = []
    for node in walk(tree.root_node):
        if node.type not in FUNCTION_TYPES:
            continue
        name = function_name(source, node)
        leaves = [
            leaf
            for leaf in walk(node)
            if leaf.type in LEAF_TYPES and leaf.end_byte > leaf.start_byte
        ]
        if len(leaves) > max_terminals:
            seed = int(hashlib.sha1(f"leaves:{name}:{node.start_byte}".encode()).hexdigest()[:8], 16)
            indices = sorted(random.Random(seed).sample(range(len(leaves)), max_terminals))
            leaves = [leaves[index] for index in indices]
        contexts = []
        for i, left in enumerate(leaves):
            upper = min(len(leaves), i + max_path_width + 1)
            for right in leaves[i + 1 : upper]:
                path = ast_path(left, right, node, max_path_length)
                if not path:
                    continue
                contexts.append((terminal_token(source, left), path, terminal_token(source, right)))
        if len(contexts) > max_contexts:
            seed = int(hashlib.sha1(f"{name}:{node.start_byte}".encode()).hexdigest()[:8], 16)
            contexts = random.Random(seed).sample(contexts, max_contexts)
        if contexts:
            functions.append(
                {
                    "name": name,
                    "name_parts": split_name(name),
                    "start_line": node.start_point[0] + 1,
                    "end_line": node.end_point[0] + 1,
                    "contexts": contexts,
                }
            )
    return functions


@dataclass
class Vocabulary:
    tokens: dict[str, int]
    paths: dict[str, int]
    labels: dict[str, int]

    @classmethod
    def build(
        cls,
        examples: list[dict[str, Any]],
        max_tokens: int = 50000,
        max_paths: int = 50000,
        max_labels: int = 20000,
    ) -> "Vocabulary":
        token_counts: Counter[str] = Counter()
        path_counts: Counter[str] = Counter()
        label_counts: Counter[str] = Counter()
        for example in examples:
            token_counts.update(context[0] for context in example["contexts"])
            token_counts.update(context[2] for context in example["contexts"])
            path_counts.update(context[1] for context in example["contexts"])
            label_counts.update(example["name_parts"])

        def make_vocab(counts: Counter[str], limit: int) -> dict[str, int]:
            return {"<PAD>": 0, "<UNK>": 1, **{
                item: index + 2 for index, (item, _) in enumerate(counts.most_common(limit - 2))
            }}

        return cls(
            make_vocab(token_counts, max_tokens),
            make_vocab(path_counts, max_paths),
            make_vocab(label_counts, max_labels),
        )


class Code2Vec(nn.Module):
    def __init__(self, vocab: Vocabulary, dim: int = 128) -> None:
        super().__init__()
        self.dim = dim
        self.token_embedding = nn.Embedding(len(vocab.tokens), dim, padding_idx=0)
        self.path_embedding = nn.Embedding(len(vocab.paths), dim, padding_idx=0)
        self.project = nn.Linear(dim * 3, dim, bias=False)
        self.attention = nn.Parameter(torch.empty(dim))
        self.classifier = nn.Linear(dim, len(vocab.labels), bias=False)
        nn.init.normal_(self.attention, std=0.1)

    def encode_contexts(
        self,
        starts: torch.Tensor,
        paths: torch.Tensor,
        ends: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        contexts = torch.cat(
            [
                self.token_embedding(starts),
                self.path_embedding(paths),
                self.token_embedding(ends),
            ],
            dim=-1,
        )
        contexts = torch.tanh(self.project(contexts))
        scores = contexts @ self.attention
        scores = scores.masked_fill(~mask, -1e9)
        weights = torch.softmax(scores, dim=1)
        return torch.sum(contexts * weights.unsqueeze(-1), dim=1)

    def forward(
        self,
        starts: torch.Tensor,
        paths: torch.Tensor,
        ends: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.classifier(self.encode_contexts(starts, paths, ends, mask))


def vectorize_contexts(
    contexts: list[tuple[str, str, str]],
    vocab: Vocabulary,
    max_contexts: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    chosen = contexts[:max_contexts]
    starts = np.zeros(max_contexts, dtype=np.int64)
    paths = np.zeros(max_contexts, dtype=np.int64)
    ends = np.zeros(max_contexts, dtype=np.int64)
    mask = np.zeros(max_contexts, dtype=bool)
    for index, (start, path, end) in enumerate(chosen):
        starts[index] = vocab.tokens.get(start, 1)
        paths[index] = vocab.paths.get(path, 1)
        ends[index] = vocab.tokens.get(end, 1)
        mask[index] = True
    return starts, paths, ends, mask


def save_checkpoint(path: Path, model: Code2Vec, vocab: Vocabulary, config: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "vocab": {
                "tokens": vocab.tokens,
                "paths": vocab.paths,
                "labels": vocab.labels,
            },
            "config": config,
        },
        path,
    )


def load_checkpoint(path: Path) -> tuple[Code2Vec, Vocabulary, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    vocab = Vocabulary(**payload["vocab"])
    model = Code2Vec(vocab, dim=int(payload["config"]["dim"]))
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, vocab, payload["config"]


class GoCode2VecEncoder:
    def __init__(
        self,
        checkpoint: str | Path,
        inference_max_contexts: int = 12,
        inference_max_terminals: int = 8,
    ) -> None:
        self.model, self.vocab, self.config = load_checkpoint(Path(checkpoint))
        self.inference_max_contexts = min(
            inference_max_contexts, int(self.config["max_contexts"])
        )
        self.inference_max_terminals = inference_max_terminals

    def function_vectors(self, source_text: str) -> list[dict[str, Any]]:
        functions = extract_function_contexts(
            source_text,
            max_contexts=self.inference_max_contexts,
            max_path_length=int(self.config["max_path_length"]),
            max_path_width=int(self.config["max_path_width"]),
            max_terminals=self.inference_max_terminals,
        )
        output = []
        if not functions:
            return output
        arrays = [
            vectorize_contexts(
                function["contexts"], self.vocab, int(self.config["max_contexts"])
            )
            for function in functions
        ]
        with torch.no_grad():
            tensors = [
                torch.from_numpy(np.stack([item[position] for item in arrays]))
                for position in range(4)
            ]
            vectors = self.model.encode_contexts(*tensors).numpy()
            for function, vector in zip(functions, vectors):
                norm = np.linalg.norm(vector)
                if norm:
                    vector = vector / norm
                output.append({**function, "vector": vector})
        return output

    def file_vector(self, source_text: str) -> np.ndarray | None:
        functions = self.function_vectors(source_text)
        if not functions:
            return None
        vector = np.mean([item["vector"] for item in functions], axis=0)
        norm = np.linalg.norm(vector)
        return vector / norm if norm else None


def collect_training_examples(
    git_dir: Path,
    max_files: int,
    max_functions: int,
    max_contexts: int,
    max_path_length: int,
    max_path_width: int,
    max_terminals: int = 60,
) -> list[dict[str, Any]]:
    sha = git_head(git_dir)
    paths = git_go_paths(git_dir, sha)
    random.Random(2027).shuffle(paths)
    examples = []
    for position, path in enumerate(paths[:max_files], start=1):
        text = git_file(git_dir, sha, path)
        if not text:
            continue
        examples.extend(
            function
            for function in extract_function_contexts(
                text, max_contexts, max_path_length, max_path_width
                , max_terminals
            )
            if function["name_parts"]
        )
        if len(examples) >= max_functions:
            break
        if position % 250 == 0:
            print(f"[extract] files={position} functions={len(examples)}", flush=True)
    return examples[:max_functions]


def train(
    git_dir: Path,
    output: Path,
    max_files: int = 4000,
    max_functions: int = 30000,
    max_contexts: int = 100,
    max_path_length: int = 8,
    max_path_width: int = 30,
    max_terminals: int = 60,
    dim: int = 128,
    epochs: int = 3,
    batch_size: int = 128,
) -> dict[str, Any]:
    random.seed(2027)
    np.random.seed(2027)
    torch.manual_seed(2027)
    examples = collect_training_examples(
        git_dir,
        max_files,
        max_functions,
        max_contexts,
        max_path_length,
        max_path_width,
        max_terminals,
    )
    vocab = Vocabulary.build(examples)
    usable = [
        example
        for example in examples
        if any(label in vocab.labels and vocab.labels[label] > 1 for label in example["name_parts"])
    ]
    model = Code2Vec(vocab, dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    epoch_losses = []
    for epoch in range(epochs):
        random.Random(2027 + epoch).shuffle(usable)
        losses = []
        for offset in range(0, len(usable), batch_size):
            batch = usable[offset : offset + batch_size]
            arrays = [vectorize_contexts(item["contexts"], vocab, max_contexts) for item in batch]
            starts = torch.from_numpy(np.stack([item[0] for item in arrays]))
            paths = torch.from_numpy(np.stack([item[1] for item in arrays]))
            ends = torch.from_numpy(np.stack([item[2] for item in arrays]))
            mask = torch.from_numpy(np.stack([item[3] for item in arrays]))
            labels = torch.tensor(
                [vocab.labels.get(item["name_parts"][0], 1) for item in batch],
                dtype=torch.long,
            )
            optimizer.zero_grad()
            loss = loss_fn(model(starts, paths, ends, mask), labels)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        epoch_loss = float(np.mean(losses)) if losses else 0.0
        epoch_losses.append(epoch_loss)
        print(f"[train] epoch={epoch + 1}/{epochs} loss={epoch_loss:.4f}", flush=True)
    config = {
        "architecture": "code2vec path-context attention",
        "language": "Go",
        "training_repo": str(git_dir),
        "training_revision": git_head(git_dir),
        "functions": len(usable),
        "dim": dim,
        "max_contexts": max_contexts,
        "max_path_length": max_path_length,
        "max_path_width": max_path_width,
        "max_terminals": max_terminals,
        "epochs": epochs,
        "epoch_losses": epoch_losses,
    }
    save_checkpoint(output, model, vocab, config)
    output.with_suffix(".json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    return config


if __name__ == "__main__":
    import argparse

    cli = argparse.ArgumentParser()
    cli.add_argument("--git-dir", default=".cache/git_repos/ethereum_go-ethereum.git")
    cli.add_argument("--output", default=".cache/go_code2vec/geth_go_code2vec.pt")
    cli.add_argument("--max-files", type=int, default=4000)
    cli.add_argument("--max-functions", type=int, default=30000)
    cli.add_argument("--epochs", type=int, default=3)
    args = cli.parse_args()
    print(
        json.dumps(
            train(
                Path(args.git_dir),
                Path(args.output),
                max_files=args.max_files,
                max_functions=args.max_functions,
                epochs=args.epochs,
            ),
            indent=2,
        )
    )
