"""Semantics-preserving rewrites of a C function for E11 (private to E-a).

Each function takes source text and returns new source text, or None when the rewrite does not apply.
E11 itself re-runs the problem's tests on the rewritten code and keeps a rewrite only if the pass/fail vector is
unchanged, so a rewrite that turns out not to preserve behaviour is counted and left out.
"""
from __future__ import annotations

import re

from pycparser import c_ast, c_generator, c_parser

from ml.c_interp.preprocess import preprocess

_GEN = c_generator.CGenerator()
_POOL = ["ans", "res", "tmp", "cnt", "num", "val", "idx", "pos", "ctr", "tot", "acc", "cur", "item", "k2", "x1", "z"]


def parse(code):
    return c_parser.CParser().parse(preprocess(code))


def generate(ast):
    return _GEN.visit(ast)


def _walk(node):
    yield node
    for _, child in node.children():
        yield from _walk(child)


# ---------------------------------------------------------------- rename

def rename(code):
    """Rename every parameter and local variable (not function names, not called functions)."""
    ast = parse(code)
    declared, used = set(), set()
    for node in _walk(ast):
        if isinstance(node, c_ast.ID):
            used.add(node.name)
    funcs = [e for e in ast.ext if isinstance(e, c_ast.FuncDef)]
    for fn in funcs:
        for node in _walk(fn):
            if isinstance(node, c_ast.Decl) and node is not fn.decl and node.name:
                declared.add(node.name)
    called = {n.name.name for n in _walk(ast) if isinstance(n, c_ast.FuncCall) and isinstance(n.name, c_ast.ID)}
    declared -= called | {f.decl.name for f in funcs}
    if not declared:
        return None
    taken = used | declared | called | {f.decl.name for f in funcs}
    pool = [p for p in _POOL if p not in taken]
    extra = 0
    mapping = {}
    for name in sorted(declared):
        if pool:
            mapping[name] = pool.pop(0)
        else:
            extra += 1
            mapping[name] = f"var{extra}_q"
    if all(k == v for k, v in mapping.items()):
        return None

    def fix_type(t):
        while t is not None and not isinstance(t, c_ast.TypeDecl):
            t = getattr(t, "type", None)
        if t is not None and t.declname in mapping:
            t.declname = mapping[t.declname]

    for fn in funcs:
        for node in _walk(fn):
            if isinstance(node, c_ast.ID) and node.name in mapping and not _is_call_target(fn, node):
                node.name = mapping[node.name]
            elif isinstance(node, c_ast.Decl) and node is not fn.decl and node.name in mapping:
                node.name = mapping[node.name]
                fix_type(node.type)
    return generate(ast)


def _is_call_target(fn, node):
    for n in _walk(fn):
        if isinstance(n, c_ast.FuncCall) and n.name is node:
            return True
    return False


# ---------------------------------------------------------------- reformat / comments / dead variable

def reformat(code):
    text = generate(parse(code))
    return text if text.strip() != code.strip() else None


def add_comments(code):
    lines = []
    for line in code.split("\n"):
        s = line.rstrip()
        if s.endswith(";") or s.endswith("{"):
            s += "  // step"
        lines.append(s)
    return "/* my solution, hope it works */\n" + "\n".join(lines)


def dead_variable(code):
    masked = re.sub(r"//[^\n]*|/\*.*?\*/", lambda m: re.sub(r"[^\n]", " ", m.group(0)), code, flags=re.S)
    i = masked.find("{")
    if i < 0:
        return None
    return code[: i + 1] + "\n    int unused_zz = 0;" + code[i + 1:]


# ---------------------------------------------------------------- for <-> while

def _has_continue(node):
    return any(isinstance(n, c_ast.Continue) for n in _walk(node))


def _eligible_for(node):
    return isinstance(node, c_ast.For) and node.cond is not None and not _has_continue(node.stmt)


def _as_items(stmt):
    if stmt is None:
        return []
    if isinstance(stmt, c_ast.Compound):
        return list(stmt.block_items or [])
    return [stmt]


def _for_to_while(node):
    init_items = []
    if isinstance(node.init, c_ast.DeclList):
        init_items = list(node.init.decls)
    elif node.init is not None:
        init_items = [node.init]
    body = _as_items(node.stmt)
    if node.next is not None:
        body = body + [node.next]
    loop = c_ast.While(node.cond, c_ast.Compound(body))
    return c_ast.Compound(init_items + [loop])


def _transform(node, convert):
    for name in node.__slots__:
        if name in ("coord", "__weakref__"):
            continue
        value = getattr(node, name, None)
        if isinstance(value, c_ast.Node):
            setattr(node, name, _transform(value, convert))
        elif isinstance(value, list):
            setattr(node, name, [_transform(v, convert) if isinstance(v, c_ast.Node) else v for v in value])
    return convert(node)


def for_while(code):
    """for -> while when the code has an eligible for loop, otherwise while -> for."""
    ast = parse(code)
    if any(_eligible_for(n) for n in _walk(ast)):
        out = _transform(ast, lambda n: _for_to_while(n) if _eligible_for(n) else n)
    elif any(isinstance(n, c_ast.While) for n in _walk(ast)):
        out = _transform(ast, lambda n: c_ast.For(None, n.cond, None, n.stmt) if isinstance(n, c_ast.While) else n)
    else:
        return None
    return generate(out)


_FLIP = {"<": ">", ">": "<", "<=": ">=", ">=": "<=", "==": "==", "!=": "!="}


def _pure(node):
    return not any(isinstance(n, (c_ast.Assignment, c_ast.FuncCall)) or (isinstance(n, c_ast.UnaryOp) and n.op in ("p++", "p--", "++", "--"))
                   for n in _walk(node))


def flip_compare(code):
    """`i < n` -> `n > i` (stress rewrite, not in the plan's list)."""
    ast = parse(code)
    changed = [False]

    def convert(n):
        if isinstance(n, c_ast.BinaryOp) and n.op in _FLIP and _pure(n.left) and _pure(n.right):
            changed[0] = True
            return c_ast.BinaryOp(_FLIP[n.op], n.right, n.left)
        return n
    out = _transform(ast, convert)
    return generate(out) if changed[0] else None


def incr_form(code):
    """Standalone `i++` / `i--` -> `i += 1` / `i -= 1` (stress rewrite, not in the plan's list)."""
    ast = parse(code)
    changed = [False]

    def conv(u):
        if isinstance(u, c_ast.UnaryOp) and u.op in ("p++", "++", "p--", "--"):
            changed[0] = True
            return c_ast.Assignment("+=" if "+" in u.op else "-=", u.expr, c_ast.Constant("int", "1"))
        return u

    def convert(n):
        if isinstance(n, c_ast.For):
            n.next = conv(n.next)
        elif isinstance(n, c_ast.Compound) and n.block_items:
            n.block_items = [conv(i) for i in n.block_items]
        return n
    out = _transform(ast, convert)
    return generate(out) if changed[0] else None


KINDS = {"rename": rename, "reformat": reformat, "comments": add_comments, "dead_variable": dead_variable,
         "for_while": for_while}
STRESS = {"flip_compare": flip_compare, "incr_form": incr_form}


def safe(fn, code):
    try:
        out = fn(code)
    except Exception:
        return None
    return out if out and out.strip() != code.strip() else None


def combined(code):
    """All five rewrites in turn (each skipped if it does not apply)."""
    out = code
    for name in ("for_while", "rename", "reformat", "dead_variable", "comments"):
        nxt = safe(KINDS[name], out)
        if nxt is not None:
            out = nxt
    return out if out != code else None
