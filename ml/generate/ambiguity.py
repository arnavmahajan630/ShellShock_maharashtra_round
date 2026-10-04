"""Normalised AST hash and the T1 ambiguity pass (03 §3.5.5)."""
from __future__ import annotations

import hashlib

from pycparser import c_ast, c_generator

from ml.generate.ops_main import parse_ast

_BUILTINS = frozenset({
    "printf", "strlen", "fire", "launch", "open_door", "close_door", "scan",
})
_GEN = c_generator.CGenerator()


def _walk(node):
    yield node
    for _, child in node.children():
        yield from _walk(child)


def _drop_debug_printfs(ast):
    """Remove printf statements whose format contains ``debug=true``."""

    def clean(node):
        if isinstance(node, c_ast.Compound) and node.block_items:
            kept = []
            for item in node.block_items:
                if _is_debug_printf(item):
                    continue
                clean(item)
                kept.append(item)
            node.block_items = kept
        else:
            for _, child in node.children():
                clean(child)

    clean(ast)


def _is_debug_printf(node):
    if not isinstance(node, c_ast.FuncCall) or not isinstance(node.name, c_ast.ID):
        return False
    if node.name.name != "printf" or node.args is None or not node.args.exprs:
        return False
    fmt = node.args.exprs[0]
    return isinstance(fmt, c_ast.Constant) and "debug=true" in (fmt.value or "")


def _canon_incr(ast):
    """Statement-level ``i++`` / ``++i`` / ``i += 1`` become ``i = i + 1``."""

    def make(name):
        return c_ast.Assignment(
            "=", c_ast.ID(name),
            c_ast.BinaryOp("+", c_ast.ID(name), c_ast.Constant("int", "1")),
        )

    def one(node):
        name = _incr_name(node)
        if name is None:
            return node
        return make(name)

    def visit(node):
        if isinstance(node, c_ast.Compound) and node.block_items:
            node.block_items = [one(item) if _incr_name(item) else item for item in node.block_items]
            for item in node.block_items:
                visit(item)
            return
        if isinstance(node, c_ast.For) and node.next is not None:
            node.next = _canon_next(node.next, one)
        for _, child in node.children():
            visit(child)

    visit(ast)


def _incr_name(node):
    if isinstance(node, c_ast.UnaryOp) and node.op in ("p++", "++") and isinstance(node.expr, c_ast.ID):
        return node.expr.name
    if isinstance(node, c_ast.Assignment) and isinstance(node.lvalue, c_ast.ID):
        if node.op == "+=" and _is_one(node.rvalue):
            return node.lvalue.name
        if (node.op == "=" and isinstance(node.rvalue, c_ast.BinaryOp) and node.rvalue.op == "+"
                and isinstance(node.rvalue.left, c_ast.ID) and node.rvalue.left.name == node.lvalue.name
                and _is_one(node.rvalue.right)):
            return node.lvalue.name
    return None


def _is_one(node):
    return isinstance(node, c_ast.Constant) and node.type == "int" and node.value == "1"


def _canon_next(node, one):
    if isinstance(node, c_ast.ExprList):
        node.exprs = [one(item) if _incr_name(item) else item for item in node.exprs]
        return node
    if _incr_name(node):
        return one(node)
    return node


def _canon_names(ast):
    """Rename locals and parameters to ``v0, v1, …`` in first-seen order.

    The entry function name and builtins stay, so two problems do not collapse
    into one hash just because their bodies were normalised.
    """
    fn_names = set()
    for node in getattr(ast, "ext", None) or []:
        if isinstance(node, c_ast.FuncDef) and node.decl is not None:
            fn_names.add(node.decl.name)
    declared = []
    seen = set()

    def add(name):
        if not name or name in seen or name in _BUILTINS or name in fn_names:
            return
        seen.add(name)
        declared.append(name)

    for node in _walk(ast):
        if isinstance(node, c_ast.Decl) and node.name:
            add(node.name)
        elif isinstance(node, c_ast.TypeDecl) and node.declname:
            add(node.declname)
    mapping = {}
    for index, name in enumerate(declared):
        mapping[name] = f"v{index}"
    if not mapping:
        return

    def apply(node):
        if isinstance(node, c_ast.ID) and node.name in mapping:
            node.name = mapping[node.name]
        if isinstance(node, c_ast.Decl) and node.name in mapping:
            node.name = mapping[node.name]
        if isinstance(node, c_ast.TypeDecl) and node.declname in mapping:
            node.declname = mapping[node.declname]
        for _, child in node.children():
            apply(child)

    apply(ast)


def normalised_text(code):
    """Canonical reprint: no comments, no debug printfs, canonical increments and names."""
    ast = parse_ast(code)
    _drop_debug_printfs(ast)
    _canon_incr(ast)
    _canon_names(ast)
    return _GEN.visit(ast)


def ast_hash(code):
    try:
        text = normalised_text(code)
    except Exception:
        text = code
    return "h_" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def execution_key(code):
    """Reprint of the parsed tree. Comments and spacing share a key; debug printfs do not."""
    try:
        return _GEN.visit(parse_ast(code))
    except Exception:
        return None


def ambiguity_pass(rows, amb_ops):
    """Soft-label exact ``{M01, M08}`` hash groups. Any other clash is printed and reduced to one label.

    AMB operators are stored once at generation (label M01). This pass copies each
    of those rows onto M08 so the group carries both labels, both with weight 0.5.
    """
    clashes = []
    grouped = {}
    for row in rows:
        grouped.setdefault(row["ast_hash"], []).append(row)
    dropped = set()
    for digest, group in grouped.items():
        labels = {row["label"] for row in group}
        if len(labels) <= 1:
            continue
        if labels == {"M01", "M08"}:
            _soften(group, digest)
            continue
        counts = {}
        for row in group:
            counts[row["label"]] = counts.get(row["label"], 0) + 1
        winner = max(counts, key=lambda lab: (counts[lab], lab))
        sample = group[0]
        clashes.append(
            f"CLASH {digest} labels={sorted(labels)} kept={winner} "
            f"problem={sample['problem_id']} ops={[row.get('op_id') for row in group[:6]]}"
        )
        print(clashes[-1])
        for row in group:
            if row["label"] != winner:
                dropped.add(id(row))
    kept = [row for row in rows if id(row) not in dropped]
    grouped = {}
    for row in kept:
        grouped.setdefault(row["ast_hash"], []).append(row)
    extra = []
    for digest, group in grouped.items():
        labels = {row["label"] for row in group}
        if labels == {"M01", "M08"}:
            _soften(group, digest)
        amb_rows = [
            row for row in group
            if row.get("op_id") in amb_ops and not row.get("is_two_bug")
        ]
        if not amb_rows:
            continue
        if not labels <= {"M01", "M08"}:
            continue
        _soften(amb_rows, digest)
        seen_codes = {}
        for row in amb_rows:
            seen_codes.setdefault(row["code"], row)
        present = {(row["code"], row["label"]) for row in group}
        for code, row in seen_codes.items():
            for label in ("M01", "M08"):
                if (code, label) in present:
                    continue
                clone = dict(row)
                clone["label"] = label
                clone["labels_all"] = ["M01", "M08"]
                clone["soft_label"] = {"M01": 0.5, "M08": 0.5}
                clone["source"] = "AMB"
                clone["ambiguous_group"] = digest
                clone["verified"] = dict(row["verified"])
                extra.append(clone)
                present.add((code, label))
    return kept + extra, clashes


def _soften(group, digest):
    for row in group:
        if row["label"] not in ("M01", "M08"):
            continue
        row["source"] = "AMB"
        row["soft_label"] = {"M01": 0.5, "M08": 0.5}
        row["labels_all"] = ["M01", "M08"]
        row["ambiguous_group"] = digest
