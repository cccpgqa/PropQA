"""Go frontend for the unmodified official NiCad crossclones engine.

This is an adaptation, not the official TXL frontend. Formatting units and
original line maps are explicit. No labels or target patch enter inference.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from xml.etree import ElementTree as ET
from xml.sax.saxutils import escape, quoteattr

from tree_sitter import Language, Parser
import tree_sitter_go
from rapidfuzz.distance import LCSseq

UPSTREAM_COMMIT = "7a90d11795a7fe585282e25b7fa8d9964f202965"
METHOD = "Open-NiCad-Go"
IDENTIFIERS = {"identifier", "field_identifier", "type_identifier", "package_identifier", "label_name"}
ATOMIC = {"interpreted_string_literal", "raw_string_literal", "rune_literal"}




@dataclass
class Fragment:
    path: str
    start: int
    end: int
    rows: list[str]
    line_map: list[list[int]]

def extract(content: str, path: str, rename: str = "blind") -> list[Fragment]:
    if rename not in {"blind", "none", "consistent"}:
        raise ValueError("Unknown normalization")
    data = content.encode("utf-8")
    tree = Parser(Language(tree_sitter_go.language())).parse(data)
    if tree.root_node.has_error:
        raise ValueError(f"Go parse error: {path}")
    result = []
    for node in tree.root_node.named_children:
        if node.type not in {"function_declaration", "method_declaration"}:
            continue
        boundaries = set()
        leaves = []

        def walk(n):
            if n.type == "comment":
                return
            if n.type.endswith("statement") or n.type in {"short_var_declaration", "var_declaration", "const_declaration"}:
                boundaries.add(n.end_byte)
            if n.type in ATOMIC or not n.children:
                if n.end_byte > n.start_byte:
                    leaves.append(n)
                return
            for child in n.children:
                walk(child)

        walk(node)
        rows, maps, tokens, positions, names = [], [], [], set(), {}

        def flush():
            if tokens:
                rows.append(" ".join(tokens))
                maps.append(sorted(positions))
                tokens.clear()
                positions.clear()

        for leaf in leaves:
            value = data[leaf.start_byte:leaf.end_byte].decode("utf-8")
            if value in {"{", "}", ";"}:
                flush()
                if value != ";":
                    rows.append(value)
                    maps.append([leaf.start_point.row + 1])
                continue
            if leaf.type in IDENTIFIERS:
                if rename == "blind":
                    value = "ID"
                elif rename == "consistent":
                    value = names.setdefault(value, f"ID{len(names)}")
            if leaf.type in ATOMIC:
                value = json.dumps(value, ensure_ascii=True)
                if len(value) > 512:
                    value = "LONG_LITERAL_" + hashlib.sha256(value.encode("utf-8")).hexdigest()
            tokens.append(value)
            positions.update(range(leaf.start_point.row + 1, leaf.end_point.row + 2))
            if leaf.end_byte in boundaries:
                flush()
        flush()
        # The official core stores lines in fixed 4096-byte strings.
        if any(len(escape(row).encode("utf-8")) >= 4090 for row in rows):
            raise ValueError(f"Normalized row exceeds core string limit: {path}")
        result.append(Fragment(path, node.start_point.row + 1, node.end_point.row + 1, rows, maps))
    return result

def write_fragments(path: Path, fragments: list[Fragment], side: str):
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for f in fragments:
            handle.write(f'<source file={quoteattr(side + "/" + f.path)} startline="{f.start}" endline="{f.end}">\n')
            handle.write("\n".join(escape(row) for row in f.rows) + "\n</source>\n")

def compare(executable, source, target, output, difference=0.3, min_lines=5, max_lines=2500):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    executable = Path(executable).resolve()
    # ASCII working paths avoid the upstream C runtime's Windows fopen limitation.
    with tempfile.TemporaryDirectory(prefix="nicad_go_") as tmp:
        cwd = Path(tmp)
        write_fragments(cwd / "source_functions.xml", source, "source")
        write_fragments(cwd / "target_functions.xml", target, "target")
        command = [str(executable), "source_functions.xml", "target_functions.xml",
                   str(difference), str(min_lines), str(max_lines)]
        run = subprocess.run(command, cwd=cwd, capture_output=True, timeout=300)
        (output / "clones.xml").write_bytes(run.stdout)
        (output / "engine.log").write_bytes(run.stderr)
        manifest = {
            "method": METHOD, "upstream_commit": UPSTREAM_COMMIT,
            "engine_sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
            "command": command, "returncode": run.returncode,
            "source_fragments": len(source), "target_fragments": len(target),
            "frontend": "tree-sitter-go functions; explicit syntax-boundary rows",
            "source_input_sha256": hashlib.sha256((cwd / "source_functions.xml").read_bytes()).hexdigest(),
            "target_input_sha256": hashlib.sha256((cwd / "target_functions.xml").read_bytes()).hexdigest(),
        }
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        (output / "fragments.json").write_text(json.dumps({
            "source": [asdict(f) for f in source], "target": [asdict(f) for f in target],
        }), encoding="utf-8")
        if run.returncode:
            raise RuntimeError(f"Official NiCad failed: see {output / 'engine.log'}")
    return ET.fromstring(run.stdout)
