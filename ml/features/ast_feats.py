"""Group A: AST structure features (plans/03 §4.1), package F1.

    feats, meta = ast_features(code, entry=None, trace=None, display_test=0)

`feats` has exactly the names in `ml.contracts.feature_names.GROUP_A` (floats).
`meta` carries what the evidence templates need:
    meta["lines"][feature]   line of the first place the feature fired
    meta["values"][feature]  placeholder values for that feature's sentence
    meta["loops"]            main / outer / inner loop lines and the main loop's condition
                             variables (used by the trace features)
    meta["parse_ok"]         False when pycparser rejected the code (every feature is NaN)

Rules fixed here (the contract leaves them open):

* Main loop (03 §4.1): the loop with the most iterations on the display test, ties to the
  outermost, then to the first in source order. Iteration counts come from
  `trace["per_test"][display_test]["loop_iters"]` ("L<line>" keys). **Without a trace the
  static rule is used: the first outermost loop in source order.** For nested loops the two
  rules can pick different loops (the inner loop usually iterates more), so rows built with
  and without a trace can differ in the `a_main_*`, `a_bound_form_*`, `a_init_form_*` and
  `a_update_*` columns.
* A learner-written `main` is ignored when the file defines any other function.
* Index shapes: "i-like" is a loop variable, any scalar the function changes after declaring
  it, or a local declared inside a loop (`a[mid]`, `a[min]` count as `a_index_i`). "n-like"
  is every other bare identifier (an unchanged parameter). An unchanged local with an
  initialiser stands for its initialiser (`int last = n - 1; a[last]` is `a_index_n_minus_1`).
* Bound and init shapes: "n" is a bare identifier other than the loop variable, or a
  `strlen(...)` call. A condition is read as `var op bound`; `n > i` is flipped to `i < n`
  when the left side is neither changed in the loop nor a local.
* All flags are 0/1 for "present anywhere in the analysed functions". Counts are
  `a_n_loops`, `a_max_nesting`, `a_n_nested_loops`, `a_n_self_calls`; `a_update_dir` is -1/0/+1.

No text, identifier, length or problem-id features are produced (03 §4.7): identifiers are
only read to decide roles (loop variable, parameter, array) and appear in `meta` for the
evidence sentences, never in `feats`.
"""
from __future__ import annotations

import math
import re

from pycparser import CParser, c_ast, c_generator

from ml.contracts.feature_names import GROUP_A

REL_OPS = {"<": "lt", "<=": "le", ">": "gt", ">=": "ge", "!=": "ne", "==": "eq"}
_FLIP = {"<": ">", "<=": ">=", ">": "<", ">=": "<=", "!=": "!=", "==": "=="}
_NEGATE = {"<": ">=", "<=": ">", ">": "<=", ">=": "<", "!=": "==", "==": "!="}
_COND_OPS = ["lt", "le", "gt", "ge", "ne", "eq", "other"]
_BOUND_FORMS = ["n", "n_minus_1", "n_plus_1", "const", "other"]
_INIT_FORMS = ["0", "1", "n", "n_minus_1", "other"]
_LOOPS = (c_ast.For, c_ast.While, c_ast.DoWhile)
_GEN = c_generator.CGenerator()


# ---------------------------------------------------------------- source clean-up

_SMART = {"“": '"', "”": '"', "‘": "'", "’": "'"}
_LITERAL = re.compile(r'"(?:\\.|[^"\\\n])*"|\'(?:\\.|[^\'\\\n])*\'')
_DEFINE = re.compile(r"^\s*#\s*define\s+([A-Za-z_]\w*)\s+(.+?)\s*$")


def strip_source(code: str) -> str:
    """Make learner code parseable by pycparser without moving any line.

    Comments become spaces (newlines kept), `#include` and other `#` lines become empty
    lines, `#define NAME literal` is applied as a whole-word substitution outside string
    and character literals. Smart quotes are normalised. This is a local copy of what
    `ml/c_interp/preprocess.py` (package A1) does; F1 must not import the interpreter.
    """
    for bad, good in _SMART.items():
        code = code.replace(bad, good)
    out, i, n, state = [], 0, len(code), None
    while i < n:
        ch = code[i]
        nxt = code[i + 1] if i + 1 < n else ""
        if state is None:
            if ch == "/" and nxt == "/":
                state, i = "line", i + 2
                out.append("  ")
                continue
            if ch == "/" and nxt == "*":
                state, i = "block", i + 2
                out.append("  ")
                continue
            if ch in "\"'":
                state = ch
            out.append(ch)
        elif state == "line":
            if ch == "\n":
                state = None
                out.append(ch)
            else:
                out.append(" ")
        elif state == "block":
            if ch == "*" and nxt == "/":
                state, i = None, i + 2
                out.append("  ")
                continue
            out.append(ch if ch == "\n" else " ")
        else:                                   # inside a string or char literal
            out.append(ch)
            if ch == "\\" and nxt:
                out.append(nxt)
                i += 2
                continue
            if ch == state or ch == "\n":
                state = None
        i += 1
    lines, defines = "".join(out).split("\n"), {}
    for k, line in enumerate(lines):
        if line.lstrip().startswith("#"):
            m = _DEFINE.match(line)
            if m:
                defines[m.group(1)] = m.group(2)
            lines[k] = ""
    text = "\n".join(lines)
    if defines:
        word = re.compile(r"\b(" + "|".join(map(re.escape, defines)) + r")\b")
        parts, last = [], 0
        for m in _LITERAL.finditer(text):
            parts.append(word.sub(lambda w: defines[w.group(1)], text[last:m.start()]))
            parts.append(m.group(0))
            last = m.end()
        parts.append(word.sub(lambda w: defines[w.group(1)], text[last:]))
        text = "".join(parts)
    return text


def parse(code: str):
    """pycparser AST of the cleaned code, or None when it does not parse."""
    try:
        return CParser().parse(strip_source(code))
    except Exception:
        return None


# ---------------------------------------------------------------- small AST helpers

def _src(node) -> str:
    return _GEN.visit(node).replace("(float) (", "(float)(").replace("(double) (", "(double)(")


def _line(node):
    coord = getattr(node, "coord", None)
    return getattr(coord, "line", None)


def _walk(node):
    yield node
    for child in node:
        yield from _walk(child)


def _is_id(node, name=None):
    return isinstance(node, c_ast.ID) and (name is None or node.name == name)


def _int_const(node):
    """Integer value of a (possibly negated) integer or char-free constant, else None."""
    sign = 1
    while isinstance(node, c_ast.UnaryOp) and node.op in ("-", "+"):
        sign = -sign if node.op == "-" else sign
        node = node.expr
    if isinstance(node, c_ast.Constant) and node.type in ("int", "long int", "unsigned int"):
        try:
            return sign * int(node.value.rstrip("uUlL"), 0)
        except ValueError:
            return None
    return None


def _is_number(node):
    while isinstance(node, c_ast.UnaryOp) and node.op in ("-", "+"):
        node = node.expr
    return isinstance(node, c_ast.Constant) and node.type != "string"


def _is_atom(node, not_name=None):
    """A bare identifier or a strlen(...) call: the things that play the role of `n`."""
    if isinstance(node, c_ast.ID):
        return node.name != not_name
    return isinstance(node, c_ast.FuncCall) and _is_id(node.name, "strlen")


def _binop(node, op):
    return isinstance(node, c_ast.BinaryOp) and node.op == op


def _base_type(names):
    if "float" in names or "double" in names:
        return "float"
    if "char" in names:
        return "char"
    if "void" in names:
        return "void"
    return "int"


def _decl_type(decl):
    """(base type, is_array) of a Decl / Typename."""
    t, is_array = decl.type, False
    while isinstance(t, (c_ast.ArrayDecl, c_ast.PtrDecl)):
        is_array, t = True, t.type
    if isinstance(t, c_ast.FuncDecl):
        t = t.type
        while isinstance(t, (c_ast.ArrayDecl, c_ast.PtrDecl)):
            t = t.type
    names = getattr(getattr(t, "type", None), "names", None) or ["int"]
    return _base_type(names), is_array


def _direct(stmt):
    """The statements directly inside a branch or body."""
    if stmt is None:
        return []
    if isinstance(stmt, c_ast.Compound):
        return list(stmt.block_items or [])
    return [stmt]


class _Fn:
    def __init__(self, node):
        self.node = node
        self.name = node.decl.name
        self.body = node.body
        self.ret, _ = _decl_type(node.decl)
        self.params = []                        # (name, base, is_array)
        args = node.decl.type.args
        for p in (args.params if args else []):
            if isinstance(p, c_ast.Decl) and p.name:
                base, is_array = _decl_type(p)
                self.params.append((p.name, base, is_array))
        self.types = {name: base for name, base, _ in self.params}
        self.arrays = {name for name, _, is_array in self.params if is_array}
        self.char_arrays = {name for name, base, is_array in self.params if is_array and base == "char"}
        self.locals = set()
        for node_ in _walk(self.body):
            if isinstance(node_, c_ast.Decl) and node_.name:
                base, is_array = _decl_type(node_)
                self.types[node_.name] = base
                self.locals.add(node_.name)
                if is_array:
                    self.arrays.add(node_.name)
                    if base == "char":
                        self.char_arrays.add(node_.name)
        self.parent = {}
        stack = [node]
        while stack:
            cur = stack.pop()
            for child in cur:
                self.parent[id(child)] = cur
                stack.append(child)

    def ancestors(self, node):
        cur = self.parent.get(id(node))
        while cur is not None:
            yield cur
            cur = self.parent.get(id(cur))


class _Mod:
    """One change of a scalar variable: ++/--, compound assignment or plain assignment."""
    __slots__ = ("name", "dir", "step", "self_update", "node")

    def __init__(self, name, direction, step, self_update, node):
        self.name, self.dir, self.step, self.self_update, self.node = name, direction, step, self_update, node


def _mods(node):
    """All scalar modifications below `node`, in source order."""
    found = []
    if node is None:
        return found
    for cur in _walk(node):
        if isinstance(cur, c_ast.UnaryOp) and cur.op in ("p++", "++", "p--", "--") and _is_id(cur.expr):
            found.append(_Mod(cur.expr.name, 1 if "++" in cur.op else -1, True, True, cur))
        elif isinstance(cur, c_ast.Assignment) and _is_id(cur.lvalue):
            name, rv = cur.lvalue.name, cur.rvalue
            const = _int_const(rv)
            if cur.op in ("+=", "-="):
                sign = 1 if cur.op == "+=" else -1
                if const is not None and const < 0:
                    sign = -sign
                found.append(_Mod(name, sign, const is not None and const != 0, True, cur))
            elif cur.op in ("*=", "/=", "%="):
                found.append(_Mod(name, 1 if cur.op == "*=" else -1, False, True, cur))
            elif cur.op == "=":
                direction, step, self_update = 0, False, False
                if isinstance(rv, c_ast.BinaryOp) and rv.op in ("+", "-", "*", "/", "%"):
                    left_self, right_self = _is_id(rv.left, name), _is_id(rv.right, name)
                    if left_self or (right_self and rv.op in ("+", "*")):
                        self_update = True
                        other = _int_const(rv.right if left_self else rv.left)
                        if rv.op in ("+", "-"):
                            direction = 1 if rv.op == "+" else -1
                            if other is not None and other < 0:
                                direction = -direction
                            step = other is not None and other != 0
                        else:
                            direction = 1 if rv.op == "*" else -1
                found.append(_Mod(name, direction, step, self_update, cur))
    return found


def _ids(node, fn):
    """Identifier names read as values below `node` (not array bases, not called names)."""
    names = []
    if node is None:
        return names
    skip = set()
    for cur in _walk(node):
        if isinstance(cur, c_ast.FuncCall):
            skip.add(id(cur.name))
        elif isinstance(cur, c_ast.ArrayRef):
            skip.add(id(cur.name))
        elif isinstance(cur, c_ast.ID) and id(cur) not in skip and cur.name not in names:
            names.append(cur.name)
    return names


# ---------------------------------------------------------------- loops

class _Loop:
    def __init__(self, node, fn, depth, parent, order):
        self.node, self.fn, self.depth, self.parent, self.order = node, fn, depth, parent, order
        self.kind = {c_ast.For: "for", c_ast.While: "while", c_ast.DoWhile: "while"}[type(node)]
        self.line = _line(node)
        self.children = []
        self.cond = node.cond
        self.body = node.stmt
        self.next = node.next if isinstance(node, c_ast.For) else None
        self.cond_vars = _ids(self.cond, fn)
        # Changes that can move the loop: for-update, side effects in the condition, the body.
        self.header_mods = _mods(self.next) + _mods(self.cond)
        self.mods = self.header_mods + _mods(self.body)
        self.cond_mods = [m for m in self.mods if m.name in self.cond_vars]
        self.var, self.init_node = self._for_init()
        self.op, self.bound, self.var_side_is_var = self._relation()
        if self.var is None:                    # e.g. `while (s[i] != '\0')`: the stepped name, else any changed one
            stepped = [m.name for m in self.cond_mods if m.self_update]
            updated = [m.name for m in self.cond_mods]
            self.var = next((v for v in self.cond_vars if v in stepped),
                            next((v for v in self.cond_vars if v in updated), None))
        if self.init_node is None and self.var is not None and not self._has_for_init():
            self.init_node = self._init_before()

    def _has_for_init(self):
        return isinstance(self.node, c_ast.For) and self.node.init is not None

    def _for_init(self):
        if not self._has_for_init():
            return None, None
        init = self.node.init
        if isinstance(init, c_ast.DeclList):
            for decl in init.decls:
                if decl.init is not None:
                    return decl.name, decl.init
            return (init.decls[0].name, None) if init.decls else (None, None)
        items = init.exprs if isinstance(init, c_ast.ExprList) else [init]
        assigns = [a for a in items if isinstance(a, c_ast.Assignment) and a.op == "=" and _is_id(a.lvalue)]
        for a in assigns:                       # prefer the variable the condition tests
            if a.lvalue.name in self.cond_vars:
                return a.lvalue.name, a.rvalue
        return (assigns[0].lvalue.name, assigns[0].rvalue) if assigns else (None, None)

    def _relations(self, cond):
        if isinstance(cond, c_ast.BinaryOp):
            if cond.op in REL_OPS:
                return [cond]
            if cond.op in ("&&", "||"):
                return self._relations(cond.left) + self._relations(cond.right)
        return []

    def _relation(self):
        """(operator name, bound expression, True when the other side is the bare loop variable)."""
        rels = self._relations(self.cond)
        if not rels:
            return "other", None, False
        if self.var:                            # for-loop: the relation that tests the init variable
            for rel in rels:
                if _is_id(rel.left, self.var):
                    return REL_OPS[rel.op], rel.right, True
                if _is_id(rel.right, self.var):
                    return REL_OPS[_FLIP[rel.op]], rel.left, True
            return REL_OPS[rels[0].op], rels[0].right, False
        # No init to name the variable. The left side is the variable when it is a changed
        # name or a local; `n > i` / `0 < n` are read from the right (operator flipped).
        updated = {m.name for m in self.cond_mods}

        def varlike(x):
            return isinstance(x, c_ast.ID) and (x.name in updated or x.name in self.fn.locals)

        for rel in rels:
            left, right = rel.left, rel.right
            if isinstance(left, c_ast.ID) and (varlike(left) or not varlike(right)):
                self.var = left.name
                return REL_OPS[rel.op], right, True
            if varlike(right):
                self.var = right.name
                return REL_OPS[_FLIP[rel.op]], left, True
        return REL_OPS[rels[0].op], rels[0].right, False

    def _init_before(self):
        """Last value given to the loop variable before the loop starts (while loops)."""
        value = None
        for cur in _walk(self.fn.body):
            if cur is self.node:
                break
            if isinstance(cur, c_ast.Decl) and cur.name == self.var:
                value = cur.init
            elif isinstance(cur, c_ast.Assignment) and cur.op == "=" and _is_id(cur.lvalue, self.var):
                value = cur.rvalue
        return value

    def bound_form(self):
        b = self.bound
        if b is None or not self.var_side_is_var:
            return "other"
        if _is_atom(b, self.var):
            return "n"
        if _binop(b, "-") and _is_atom(b.left, self.var) and _int_const(b.right) == 1:
            return "n_minus_1"
        if _binop(b, "+") and ((_is_atom(b.left, self.var) and _int_const(b.right) == 1)
                               or (_is_atom(b.right, self.var) and _int_const(b.left) == 1)):
            return "n_plus_1"
        if _int_const(b) is not None:
            return "const"
        return "other"

    def half_bound(self):
        b = self.bound
        return b is not None and _binop(b, "/") and _is_atom(b.left) and _int_const(b.right) == 2

    def init_form(self):
        v = self.init_node
        if v is None:
            return "other"
        if _int_const(v) == 0:
            return "0"
        if _int_const(v) == 1:
            return "1"
        if _is_atom(v, self.var):
            return "n"
        if _binop(v, "-") and _is_atom(v.left, self.var) and _int_const(v.right) == 1:
            return "n_minus_1"
        return "other"

    def updates(self):
        """Step-like changes of any variable plus every change of a condition variable."""
        return [m for m in self.mods if m.step or m.name in self.cond_vars]

    def primary_update(self):
        ups = self.updates()
        for pick in (lambda m: m.name == self.var, lambda m: m.name in self.cond_vars, lambda m: True):
            for m in ups:
                if pick(m):
                    return m
        return None

    def always_progresses(self):
        """True when every pass through the loop changes a condition variable or leaves the loop."""
        names = set(self.cond_vars)
        if any(m.name in names for m in self.header_mods):
            return True

        def stmt(s):
            if s is None:
                return False
            if isinstance(s, c_ast.Compound):
                return any(stmt(c) for c in (s.block_items or []))
            if isinstance(s, c_ast.If):
                return stmt(s.iftrue) and stmt(s.iffalse)
            if isinstance(s, (c_ast.Return, c_ast.Break)):
                return True
            if isinstance(s, _LOOPS):
                return False
            return any(m.name in names for m in _mods(s))

        return stmt(self.body)

    def indexes_with_var(self):
        if self.var is None:
            return None
        for part in (self.body, self.cond):
            if part is None:
                continue
            for cur in _walk(part):
                if isinstance(cur, c_ast.ArrayRef) and _is_id(cur.subscript, self.var):
                    return cur
        return None


def _find_loops(fn, start_order):
    loops = []

    def visit(node, depth, parent):
        for child in node:
            if isinstance(child, _LOOPS):
                loop = _Loop(child, fn, depth + 1, parent, start_order + len(loops))
                loops.append(loop)
                if parent is not None:
                    parent.children.append(loop)
                visit(child, depth + 1, loop)
            else:
                visit(child, depth, parent)

    visit(fn.body, 0, None)
    return loops


def _iter_counts(trace, display_test):
    if not trace:
        return None
    per_test = trace.get("per_test") or []
    if 0 <= display_test < len(per_test):
        return per_test[display_test].get("loop_iters") or {}
    return trace.get("loop_iters") or {}


def pick_main_loop(loops, trace=None, display_test=0):
    """The main loop by the 03 §4.1 rule; static fallback = first outermost loop."""
    if not loops:
        return None
    counts = _iter_counts(trace, display_test)
    if counts is None:
        return min(loops, key=lambda lp: (lp.depth, lp.order))
    return min(loops, key=lambda lp: (-counts.get(f"L{lp.line}", 0), lp.depth, lp.order))


# ---------------------------------------------------------------- expression typing

def _etype(node, fn, ret_types):
    """'float' or 'int' for an expression, from declarations only."""
    if isinstance(node, c_ast.Constant):
        return "float" if node.type in ("float", "double", "long double") else "int"
    if isinstance(node, c_ast.ID):
        return "float" if fn.types.get(node.name) == "float" else "int"
    if isinstance(node, c_ast.ArrayRef):
        return _etype(node.name, fn, ret_types)
    if isinstance(node, c_ast.Cast):
        return "float" if _decl_type(node.to_type)[0] == "float" else "int"
    if isinstance(node, c_ast.UnaryOp):
        return "int" if node.op in ("!", "sizeof") else _etype(node.expr, fn, ret_types)
    if isinstance(node, c_ast.BinaryOp):
        if node.op in ("+", "-", "*", "/"):
            sides = (_etype(node.left, fn, ret_types), _etype(node.right, fn, ret_types))
            return "float" if "float" in sides else "int"
        return "int"
    if isinstance(node, c_ast.TernaryOp):
        sides = (_etype(node.iftrue, fn, ret_types), _etype(node.iffalse, fn, ret_types))
        return "float" if "float" in sides else "int"
    if isinstance(node, c_ast.Assignment):
        return _etype(node.lvalue, fn, ret_types)
    if isinstance(node, c_ast.FuncCall) and isinstance(node.name, c_ast.ID):
        return "float" if ret_types.get(node.name.name) == "float" else "int"
    return "int"


def _has_float_literal(node):
    if isinstance(node, c_ast.Constant):
        return node.type in ("float", "double", "long double")
    if isinstance(node, c_ast.BinaryOp) and node.op in ("+", "-", "*", "/"):
        return _has_float_literal(node.left) or _has_float_literal(node.right)
    if isinstance(node, c_ast.UnaryOp) and node.op in ("-", "+"):
        return _has_float_literal(node.expr)
    return False


def _flows_into_float(div, fn, ret_types):
    """Does this int/int division end up in a float (variable, return value, cast, expression)?"""
    cur = div
    while True:
        p = fn.parent.get(id(cur))
        if p is None:
            return False
        if isinstance(p, c_ast.Cast):
            return _decl_type(p.to_type)[0] == "float"
        if isinstance(p, c_ast.BinaryOp) and p.op in ("+", "-", "*", "/", "%"):
            if _etype(p, fn, ret_types) == "float":
                return True
            cur = p
            continue
        if isinstance(p, c_ast.UnaryOp) and p.op in ("-", "+"):
            cur = p
            continue
        if isinstance(p, c_ast.TernaryOp) and cur is not p.cond:
            cur = p
            continue
        if isinstance(p, c_ast.Assignment) and cur is p.rvalue:
            return _etype(p.lvalue, fn, ret_types) == "float"
        if isinstance(p, c_ast.Decl) and cur is p.init:
            return _decl_type(p)[0] == "float" and not _decl_type(p)[1]
        if isinstance(p, c_ast.Return):
            return fn.ret == "float"
        return False


# ---------------------------------------------------------------- read / write order

def _rw_events(node, out):
    """Variable uses in evaluation order: ('r' | 'w' | 'd', name, line, decl node or None)."""
    if node is None:
        return
    if isinstance(node, c_ast.ID):
        out.append(("r", node.name, _line(node), None))
    elif isinstance(node, c_ast.Assignment):
        _rw_events(node.rvalue, out)
        if node.op == "=" and isinstance(node.lvalue, c_ast.ID):
            out.append(("w", node.lvalue.name, _line(node), None))
        elif node.op == "=" and isinstance(node.lvalue, c_ast.ArrayRef):
            _rw_events(node.lvalue.subscript, out)
        else:
            _rw_events(node.lvalue, out)
    elif isinstance(node, c_ast.Decl):
        _rw_events(node.init, out)
        out.append(("d", node.name, _line(node), node))
    elif isinstance(node, c_ast.FuncCall):
        _rw_events(node.args, out)
    elif isinstance(node, c_ast.For):
        for part in (node.init, node.cond, node.stmt, node.next):
            _rw_events(part, out)
    elif isinstance(node, c_ast.DoWhile):
        _rw_events(node.stmt, out)
        _rw_events(node.cond, out)
    elif isinstance(node, (c_ast.Typename, c_ast.TypeDecl, c_ast.IdentifierType)):
        return
    else:
        for child in node:
            _rw_events(child, out)


# ---------------------------------------------------------------- recursion

def _self_calls(fn, node=None):
    return [c for c in _walk(node if node is not None else fn.body)
            if isinstance(c, c_ast.FuncCall) and _is_id(c.name, fn.name)]


def _base_cmps(cond, negate, params):
    """Comparisons `param op value` that are true in the base case. value: int, or None for a non-constant."""
    if isinstance(cond, c_ast.BinaryOp):
        if cond.op in ("&&", "||"):
            return _base_cmps(cond.left, negate, params) + _base_cmps(cond.right, negate, params)
        if cond.op in REL_OPS:
            left, right, op = cond.left, cond.right, cond.op
            if not (isinstance(left, c_ast.ID) and left.name in params):
                left, right, op = right, left, _FLIP[op]
            if isinstance(left, c_ast.ID) and left.name in params:
                return [(left.name, _NEGATE[op] if negate else op, _int_const(right))]
        return []
    if isinstance(cond, c_ast.UnaryOp) and cond.op == "!":
        return _base_cmps(cond.expr, not negate, params)
    if isinstance(cond, c_ast.ID) and cond.name in params:
        return [(cond.name, "==" if negate else "!=", 0)]
    return []


def _arg_shape(arg, param):
    """(form, direction) of a recursive argument relative to its parameter."""
    if _is_id(arg, param):
        return "same", "same"
    if isinstance(arg, c_ast.BinaryOp) and _is_id(arg.left, param):
        c = _int_const(arg.right)
        if c is not None and c > 0:
            if arg.op == "-":
                return ("minus1", "dec1") if c == 1 else (("minus2" if c == 2 else "other"), "dec2")
            if arg.op == "+":
                return ("plus1" if c == 1 else "other"), "inc"
            if arg.op == "/" and c >= 2:
                return ("div10" if c == 10 else "other"), "div"
    if _binop(arg, "+") and _is_id(arg.right, param) and (_int_const(arg.left) or 0) > 0:
        return ("plus1" if _int_const(arg.left) == 1 else "other"), "inc"
    return "other", "unknown"


def _reaches(direction, cmps):
    ops = [(op, value) for _, op, value in cmps]
    if direction == "dec1":
        return any(op in ("<=", "<", "==") for op, _ in ops)
    if direction == "dec2":
        consts = {v for op, v in ops if op == "==" and v is not None}
        return any(op in ("<=", "<") for op, _ in ops) or any(v + 1 in consts for v in consts)
    if direction == "div":
        return any(op in ("<=", "<") or (op == "==" and v == 0) for op, v in ops)
    if direction == "inc":
        return any(op in (">=", ">") or (op == "==" and v is None) for op, v in ops)
    return False


def _recursion(fn):
    """Everything the recursion features need for one recursive function."""
    calls = _self_calls(fn)
    scalars = [name for name, _, is_array in fn.params if not is_array]
    index = {name: k for k, (name, _, _) in enumerate(fn.params)}
    base_line, cmps, present = None, [], False
    for cur in _walk(fn.body):
        if isinstance(cur, c_ast.If):
            for branch, negate in ((cur.iftrue, False), (cur.iffalse, True)):
                if branch is None or _self_calls(fn, branch):
                    continue
                if any(isinstance(s, c_ast.Return) for s in _walk(branch)):
                    present, base_line = True, base_line or _line(cur)
                    cmps += _base_cmps(cur.cond, negate, scalars)
        elif isinstance(cur, c_ast.TernaryOp):
            in_true, in_false = bool(_self_calls(fn, cur.iftrue)), bool(_self_calls(fn, cur.iffalse))
            if in_true != in_false:
                present, base_line = True, base_line or _line(cur)
                cmps += _base_cmps(cur.cond, in_true, scalars)
    if not present and calls:
        guards = []
        for call in calls:                      # recursion only inside an `if`: the fall-through is the base
            prev, guard = call, None
            for anc in fn.ancestors(call):
                if isinstance(anc, c_ast.If) and prev is not anc.cond:
                    guard = (anc, prev is anc.iftrue)
                    break
                prev = anc
            guards.append(guard)
        if all(guards):
            present, base_line = True, _line(guards[0][0])
            for anc, in_true in guards:
                cmps += _base_cmps(anc.cond, in_true, scalars)
    param = next((name for name, _, _ in cmps), None)
    if param is None:
        for name in scalars:
            args = [c.args.exprs[index[name]] for c in calls if c.args and len(c.args.exprs) > index[name]]
            if any(not _is_id(a, name) for a in args):
                param = name
                break
        else:
            param = scalars[0] if scalars else None
    shapes = []
    for call in calls:
        exprs = call.args.exprs if call.args else []
        if param is not None and index[param] < len(exprs):
            shapes.append(_arg_shape(exprs[index[param]], param) + (call,))
        else:
            shapes.append(("other", "unknown", call))
    own = [c for c in cmps if c[0] == param]
    if not present:
        reachable = False
    elif own:
        reachable = all(_reaches(direction, own) for _, direction, _ in shapes)
    else:                                       # base condition not understood: only `f(n)` is surely stuck
        reachable = all(direction != "same" for _, direction, _ in shapes)
    return {"calls": calls, "present": present, "base_line": base_line, "reachable": reachable,
            "shapes": shapes, "param": param}


# ---------------------------------------------------------------- the feature function

def _nan_result():
    return ({name: math.nan for name in GROUP_A},
            {"parse_ok": False, "lines": {}, "values": {}, "entry": None,
             "loops": {"main": None, "outer": None, "inner": None, "all": [], "inline": [], "main_var": None,
                       "main_cond_vars": [], "selected_by": "none"}})


def _definitely_returns(stmt):
    if isinstance(stmt, c_ast.Return):
        return True
    if isinstance(stmt, c_ast.Compound):
        return any(_definitely_returns(s) for s in (stmt.block_items or []))
    if isinstance(stmt, c_ast.If):
        return (stmt.iffalse is not None and _definitely_returns(stmt.iftrue)
                and _definitely_returns(stmt.iffalse))
    return False


def ast_features(code, entry=None, trace=None, display_test=0):
    """Group-A features of `code`. See the module docstring for the rules.

    entry         name of the function the problem asks for; default = the last function
                  defined that is not `main`.
    trace         the learner's trace dict (schemas.Trace) or None. Only used to choose the
                  main loop. None → static rule (first outermost loop).
    display_test  index into trace["per_test"] whose iteration counts choose the main loop.
    """
    ast = parse(code)
    if ast is None:
        return _nan_result()
    all_fns = [_Fn(node) for node in ast.ext if isinstance(node, c_ast.FuncDef)]
    fns = [f for f in all_fns if f.name != "main"] or all_fns
    if not fns:
        return _nan_result()
    entry_fn = next((f for f in fns if f.name == entry), fns[-1])
    ret_types = {f.name: f.ret for f in all_fns}

    feats = {name: 0.0 for name in GROUP_A}
    lines, values = {}, {}

    def hit(name, node=None, **vals):
        """Set a flag; keep the place and the sentence values of its first occurrence."""
        first = feats[name] == 0.0
        feats[name] = 1.0
        if first:
            line = node if isinstance(node, int) or node is None else _line(node)
            if line is not None:
                lines[name] = line
                vals.setdefault("line", line)
            values[name] = vals

    # ---- loops
    loops = []
    for fn in fns:
        loops += _find_loops(fn, len(loops))
    feats["a_n_loops"] = float(len(loops))
    feats["a_max_nesting"] = float(max((lp.depth for lp in loops), default=0))
    feats["a_n_nested_loops"] = float(sum(lp.depth >= 2 for lp in loops))
    main = pick_main_loop(loops, trace, display_test)
    outer = next((lp for lp in loops if lp.depth == 1 and lp.children), None)
    inner = outer.children[0] if outer else None

    if main is not None:
        hit(f"a_main_cond_op_{main.op}", main.line, cond=_src(main.cond) if main.cond is not None else "")
        bound = _src(main.bound) if main.bound is not None else ""
        hit(f"a_bound_form_{main.bound_form()}", main.line, bound=bound, var=main.var or "")
        init = _src(main.init_node) if main.init_node is not None else ""
        hit(f"a_init_form_{main.init_form()}", _line(main.init_node) or main.line, init=init, var=main.var or "")
        primary = main.primary_update()
        if primary is not None:
            hit("a_update_present", primary.node, var=primary.name)
            if primary.name in main.cond_vars:  # a step on some other variable does not move the loop
                feats["a_update_dir"] = float(primary.dir)
        else:                                   # value stays 0; the sentence still needs the place
            lines["a_update_present"] = main.line
            values["a_update_present"] = {"line": main.line, "var": main.var or ""}
        lines["a_update_dir"] = _line(primary.node) if primary is not None else main.line
        values["a_update_dir"] = {"line": lines["a_update_dir"], "var": primary.name if primary else (main.var or "")}
        if main.cond_mods:
            hit("a_update_var_is_cond_var", main.cond_mods[0].node, var=main.cond_mods[0].name)
            if not main.always_progresses():
                hit("a_update_in_branch", main.cond_mods[0].node, var=main.cond_mods[0].name)
        else:
            lines["a_update_var_is_cond_var"] = main.line
            values["a_update_var_is_cond_var"] = {
                "line": main.line, "var": primary.name if primary else (main.var or "")}
    if outer is not None:
        hit(f"a_outer_cond_op_{outer.op}", outer.line)
        hit(f"a_inner_cond_op_{inner.op}", inner.line)
        form = outer.bound_form()
        hit(f"a_outer_bound_{form}", outer.line, bound=_src(outer.bound) if outer.bound is not None else "")
        if form == "const" and _int_const(outer.bound) == 1:
            hit("a_outer_bound_const1", outer.line)
    for lp in loops:
        if lp.half_bound():
            hit("a_half_bound", lp.line, bound=_src(lp.bound))
        ref = lp.indexes_with_var()
        if ref is not None:
            if lp.init_form() == "1":
                hit("a_array_loop_start1", lp.line, var=lp.var, arr=_src(ref.name))
            if lp.op == "le" and lp.bound_form() == "n":
                hit("a_array_loop_le_n", lp.line, var=lp.var, arr=_src(ref.name), bound=_src(lp.bound))

    swap_found = False
    for fn in fns:
        fn_loops = [lp for lp in loops if lp.fn is fn]
        # "i-like" names: loop variables and every scalar the function changes after declaring it.
        # Everything else (unchanged parameters, unchanged locals) is "n-like".
        # A local declared inside a loop gets a new value every pass, so it is i-like too.
        lvars = {m.name for m in _mods(fn.body)} | {lp.var for lp in fn_loops if lp.var}
        decls = [d for d in _walk(fn.body) if isinstance(d, c_ast.Decl) and not _decl_type(d)[1]]
        lvars |= {d.name for d in decls if any(isinstance(a, _LOOPS) for a in fn.ancestors(d))}
        # An unchanged local with an initialiser stands for that expression: `int last = n - 1; a[last]`.
        aliases = {d.name: d.init for d in decls if d.init is not None and d.name not in lvars}

        def in_loop(node, fn=fn):
            return any(isinstance(a, _LOOPS) for a in fn.ancestors(node))

        # ---- statement shapes
        for cur in _walk(fn.body):
            if isinstance(cur, c_ast.If):
                if isinstance(cur.iftrue, c_ast.EmptyStatement):
                    hit("a_empty_body_if", cur)
                parent = fn.parent.get(id(cur))
                chained = (isinstance(parent, c_ast.If) and parent.iffalse is cur) or isinstance(cur.iffalse, c_ast.If)
                if cur.iffalse is not None and not chained and in_loop(cur):
                    else_stmts, then_stmts = _direct(cur.iffalse), _direct(cur.iftrue)
                    ret = next((s for s in else_stmts if isinstance(s, c_ast.Return)), None)
                    if ret is not None:
                        hit("a_return_in_loop_else", ret,
                            value=_src(ret.expr) if ret.expr is not None else "")

                    def plain(stmts):
                        return {s.lvalue.name: s for s in stmts
                                if isinstance(s, c_ast.Assignment) and s.op == "=" and _is_id(s.lvalue)}

                    both = [name for name in plain(else_stmts) if name in plain(then_stmts)]
                    if both:
                        hit("a_assign_in_loop_else", plain(else_stmts)[both[0]], var=both[0])
            elif isinstance(cur, c_ast.For) and isinstance(cur.stmt, c_ast.EmptyStatement):
                hit("a_empty_body_for", cur)
            elif isinstance(cur, c_ast.While) and isinstance(cur.stmt, c_ast.EmptyStatement):
                hit("a_empty_body_while", cur)

            if isinstance(cur, (c_ast.If, c_ast.While, c_ast.DoWhile, c_ast.For)) and cur.cond is not None:
                stack = [cur.cond]
                while stack:                    # the condition itself, or an operand of && || !
                    c = stack.pop()
                    if isinstance(c, c_ast.Assignment) and c.op == "=":
                        hit("a_assign_in_cond", c, expr=_src(c),
                            var=_src(c.lvalue), value=_src(c.rvalue))
                    elif isinstance(c, c_ast.BinaryOp) and c.op in ("&&", "||"):
                        stack += [c.right, c.left]
                    elif isinstance(c, c_ast.UnaryOp) and c.op == "!":
                        stack.append(c.expr)

            # ---- division typing
            if _binop(cur, "/"):
                if _has_float_literal(cur.left) or _has_float_literal(cur.right):
                    hit("a_float_literal_in_div", cur, expr=_src(cur))
                if _etype(cur.left, fn, ret_types) == "int" and _etype(cur.right, fn, ret_types) == "int":
                    parent = fn.parent.get(id(cur))
                    if isinstance(parent, c_ast.Cast) and _decl_type(parent.to_type)[0] == "float":
                        hit("a_cast_wraps_division", parent, expr=_src(parent))
                    if _flows_into_float(cur, fn, ret_types):
                        hit("a_int_div_into_float", cur, expr=_src(cur))

            # ---- index shapes
            if isinstance(cur, c_ast.ArrayRef):
                sub, text = cur.subscript, _src(cur)
                if isinstance(sub, c_ast.ID) and sub.name in aliases:
                    sub = aliases[sub.name]
                c = _int_const(sub)
                if isinstance(sub, c_ast.ID):
                    hit("a_index_i" if sub.name in lvars else "a_index_n", cur, expr=text, arr=_src(cur.name))
                elif c == 0:
                    hit("a_index_const0", cur, expr=text, arr=_src(cur.name))
                elif c == 1:
                    hit("a_index_const1", cur, expr=text, arr=_src(cur.name))
                elif _binop(sub, "+"):
                    for a, b in ((sub.left, sub.right), (sub.right, sub.left)):
                        if isinstance(a, c_ast.ID) and a.name in lvars and _int_const(b) == 1:
                            hit("a_index_i_plus_1", cur, expr=text, arr=_src(cur.name))
                            break
                elif _binop(sub, "-"):
                    left, right = sub.left, sub.right
                    if isinstance(left, c_ast.ID) and _int_const(right) == 1:
                        hit("a_index_i_minus_1" if left.name in lvars else "a_index_n_minus_1",
                            cur, expr=text, arr=_src(cur.name))
                    elif _is_atom(left) and isinstance(right, c_ast.ID) and right.name in lvars \
                            and not _is_id(left, right.name):
                        hit("a_index_mirror_n_minus_i", cur, expr=text, arr=_src(cur.name))
                    else:
                        def is_lvar(x):
                            return isinstance(x, c_ast.ID) and x.name in lvars

                        shapes = (
                            _binop(left, "-") and _is_atom(left.left) and _int_const(left.right) == 1 and is_lvar(right),
                            _binop(left, "-") and _is_atom(left.left) and is_lvar(left.right) and _int_const(right) == 1,
                            _is_atom(left) and _binop(right, "+") and (
                                (is_lvar(right.left) and _int_const(right.right) == 1)
                                or (is_lvar(right.right) and _int_const(right.left) == 1)),
                        )
                        if any(shapes):
                            hit("a_index_mirror_n_minus_1_minus_i", cur, expr=text, arr=_src(cur.name))
                parent = fn.parent.get(id(cur))
                written = (isinstance(parent, c_ast.Assignment) and parent.lvalue is cur) or (
                    isinstance(parent, c_ast.UnaryOp) and parent.op in ("p++", "++", "p--", "--"))
                if written:
                    hit("a_has_array_write", cur, expr=text, arr=_src(cur.name))

            # ---- binary search shapes
            target = value = None
            if isinstance(cur, c_ast.Decl) and cur.init is not None:
                target, value = cur.name, cur.init
            elif isinstance(cur, c_ast.Assignment) and cur.op == "=" and isinstance(cur.lvalue, c_ast.ID):
                target, value = cur.lvalue.name, cur.rvalue
            if target is not None and _is_mid_expr(value):
                hit("a_mid_like_var", cur, var=target, expr=_src(value))
                _mid_vars(fn).add(target)

            # ---- strings
            if isinstance(cur, c_ast.BinaryOp) and cur.op in ("==", "!="):
                sides = (cur.left, cur.right)
                if any(isinstance(s, c_ast.Constant) and s.type == "string" for s in sides):
                    hit("a_str_literal_compare", cur, expr=_src(cur))
                elif all(isinstance(s, c_ast.ID) and s.name in fn.arrays for s in sides):
                    hit("a_array_name_compare", cur, expr=_src(cur))
                for a, b in (sides, sides[::-1]):
                    if isinstance(a, c_ast.ArrayRef) and _is_id(a.name) and a.name.name in fn.char_arrays \
                            and _int_const(b) == 0:
                        hit("a_uses_terminator", cur, expr=_src(cur))
            if isinstance(cur, c_ast.Constant) and cur.type == "char" and cur.value in ("'\\0'", "'\\x0'", "'\\x00'"):
                hit("a_uses_terminator", cur, expr=cur.value)
            if isinstance(cur, _LOOPS + (c_ast.If,)) and cur.cond is not None:
                c = cur.cond
                while isinstance(c, c_ast.UnaryOp) and c.op == "!":
                    c = c.expr
                if isinstance(c, c_ast.ArrayRef) and _is_id(c.name) and c.name.name in fn.char_arrays:
                    hit("a_uses_terminator", c, expr=_src(c))

            # ---- swaps (consecutive statements of one block)
            if isinstance(cur, c_ast.Compound):
                items = cur.block_items or []
                for k in range(len(items) - 1):
                    a, b = items[k], items[k + 1]
                    if _plain_assign(a) and _plain_assign(b):
                        ax, ay, bx, by = _src(a.lvalue), _src(a.rvalue), _src(b.lvalue), _src(b.rvalue)
                        if ax == by and ay == bx and ax != ay:
                            hit("a_swap_no_temp", a, x=ax, y=ay)
                            swap_found = True
                    if k + 2 < len(items):
                        c3 = items[k + 2]
                        temp = saved = None
                        if _plain_assign(a) and isinstance(a.lvalue, c_ast.ID):
                            temp, saved = a.lvalue.name, _src(a.rvalue)
                        elif isinstance(a, c_ast.Decl) and a.init is not None and not _decl_type(a)[1]:
                            temp, saved = a.name, _src(a.init)
                        if temp and _plain_assign(b) and _plain_assign(c3) and _src(b.lvalue) == saved \
                                and _src(b.rvalue) == _src(c3.lvalue) and _is_id(c3.rvalue, temp) \
                                and saved != _src(b.rvalue):
                            hit("a_swap_with_temp", a, x=saved, y=_src(b.rvalue), temp=temp)
                            swap_found = True

        # `low = mid` / `high = mid` (needs the mid-like names of the whole function first)
        mids = _mid_vars(fn)
        for cur in _walk(fn.body):
            if isinstance(cur, c_ast.Assignment) and cur.op == "=" and isinstance(cur.lvalue, c_ast.ID) \
                    and isinstance(cur.rvalue, c_ast.ID) and cur.rvalue.name in mids \
                    and cur.lvalue.name not in mids:
                hit("a_mid_assign_no_offset", cur, var=cur.lvalue.name, mid=cur.rvalue.name)

        # ---- pair access a[v] together with a[v + 1]
        plain_idx, plus_idx = {}, {}
        for cur in _walk(fn.body):
            if isinstance(cur, c_ast.ArrayRef) and _is_id(cur.name):
                sub = cur.subscript
                if isinstance(sub, c_ast.ID):
                    plain_idx.setdefault((cur.name.name, sub.name), cur)
                elif _binop(sub, "+"):
                    for a, b in ((sub.left, sub.right), (sub.right, sub.left)):
                        if isinstance(a, c_ast.ID) and _int_const(b) == 1:
                            plus_idx.setdefault((cur.name.name, a.name), cur)
        for key, node in plus_idx.items():
            if key in plain_idx:
                hit("a_index_pair_plus1", node, expr=_src(node), arr=key[0], var=key[1])

        # ---- declared without a value, first use is a read
        events = []
        _rw_events(fn.body, events)
        for k, (kind, name, line, decl) in enumerate(events):
            if kind != "d" or decl.init is not None or _decl_type(decl)[1]:
                continue
            for kind2, name2, line2, _ in events[k + 1:]:
                if name2 != name:
                    continue
                if kind2 == "r":
                    hit("a_decl_noinit_read_first", line2, var=name, decl_line=line)
                break

        # ---- accumulator set to a constant inside the loop that also updates it
        for lp in fn_loops:
            inner_cond_vars = set(lp.cond_vars)
            for other in fn_loops:
                if other is not lp and any(a is lp.node for a in fn.ancestors(other.node)):
                    inner_cond_vars.update(other.cond_vars)
                    if other.var:
                        inner_cond_vars.add(other.var)
            updated = {m.name for m in _mods(lp.body) if m.self_update}
            for stmt in _direct(lp.body):
                name = None
                if isinstance(stmt, c_ast.Decl) and stmt.init is not None and _is_number(stmt.init) \
                        and not _decl_type(stmt)[1]:
                    name = stmt.name
                elif _plain_assign(stmt) and isinstance(stmt.lvalue, c_ast.ID) and _is_number(stmt.rvalue):
                    name = stmt.lvalue.name
                if name and name in updated and name not in inner_cond_vars:
                    hit("a_init_inside_loop", stmt, var=name)

        # ---- output vs return
        if fn.ret != "void":
            for cur in _walk(fn.body):
                if isinstance(cur, c_ast.FuncCall) and _is_id(cur.name, "printf"):
                    hit("a_printf_in_nonvoid", cur, fn=fn.name)
                    break
            if not _definitely_returns(fn.body):
                hit("a_nonvoid_missing_return", _line(fn.node), fn=fn.name)

        if fn.char_arrays & {name for name, _, _ in fn.params}:
            hit("a_char_array_param", _line(fn.node), fn=fn.name)

        # ---- recursion
        calls = _self_calls(fn)
        if calls:
            rec = _recursion(fn)
            feats["a_n_self_calls"] += float(len(calls))
            hit("a_has_recursion", calls[0], fn=fn.name)
            if rec["present"]:
                hit("a_base_case_present", rec["base_line"], fn=fn.name)
            else:
                lines.setdefault("a_base_case_present", _line(fn.node))
                values.setdefault("a_base_case_present", {"line": _line(fn.node), "fn": fn.name})
            if rec["reachable"]:
                hit("a_base_case_static_reachable", rec["base_line"], fn=fn.name)
            else:
                lines.setdefault("a_base_case_static_reachable", rec["base_line"] or _line(fn.node))
                values.setdefault("a_base_case_static_reachable",
                                  {"line": rec["base_line"] or _line(fn.node), "fn": fn.name})
            for form, _, call in rec["shapes"]:
                args = ", ".join(_src(a) for a in (call.args.exprs if call.args else []))
                hit(f"a_rec_arg_form_{form}", call, fn=fn.name, arg=args, param=rec["param"] or "")
            if fn.ret != "void":
                for call in calls:
                    parent = fn.parent.get(id(call))
                    as_stmt = isinstance(parent, c_ast.Compound) or (
                        isinstance(parent, c_ast.If) and call is not parent.cond) or (
                        isinstance(parent, _LOOPS) and call is parent.stmt)
                    if as_stmt:
                        args = ", ".join(_src(a) for a in (call.args.exprs if call.args else []))
                        hit("a_rec_call_discarded", call, fn=fn.name, arg=args)

    if swap_found and loops and feats["a_n_nested_loops"] == 0:
        line = lines.get("a_swap_no_temp") or lines.get("a_swap_with_temp")
        hit("a_single_loop_with_swap", line)

    # ---- entry function only
    returns = [r for r in _walk(entry_fn.body) if isinstance(r, c_ast.Return)]
    if entry_fn.ret != "void" and returns and all(r.expr is not None and _is_number(r.expr) for r in returns):
        hit("a_return_const_only", returns[0], value=_src(returns[0].expr))
    used = set()
    for cur in _walk(entry_fn.body):
        if isinstance(cur, c_ast.ID):
            used.add(cur.name)
    unused = [name for name, _, _ in entry_fn.params if name not in used]
    if not unused:
        hit("a_reads_all_params", _line(entry_fn.node))
    else:
        lines["a_reads_all_params"] = _line(entry_fn.node)
        values["a_reads_all_params"] = {"line": _line(entry_fn.node), "param": unused[0]}

    meta = {
        "parse_ok": True, "lines": lines, "values": values, "entry": entry_fn.name,
        "loops": {
            "main": main.line if main else None,
            "outer": outer.line if outer else None,
            "inner": inner.line if inner else None,
            "all": [lp.line for lp in loops],
            # loops whose body starts on the header line: steps on that line are body steps too
            "inline": [lp.line for lp in loops if _line(next(iter(_direct(lp.body)), None)) == lp.line
                       and not isinstance(lp.body, c_ast.EmptyStatement)],
            "main_var": main.var if main else None,
            "main_cond_vars": list(main.cond_vars) if main else [],
            "selected_by": "none" if main is None else ("trace" if trace else "static"),
        },
    }
    return feats, meta


def _plain_assign(node):
    return isinstance(node, c_ast.Assignment) and node.op == "="


def _is_mid_expr(node):
    """`(x + y) / 2`, `x + (y - x) / 2` or `(y - x) / 2 + x`."""
    if _binop(node, "/") and _int_const(node.right) == 2 and _binop(node.left, "+") \
            and isinstance(node.left.left, c_ast.ID) and isinstance(node.left.right, c_ast.ID):
        return True
    if _binop(node, "+"):
        for base, half in ((node.left, node.right), (node.right, node.left)):
            if isinstance(base, c_ast.ID) and _binop(half, "/") and _int_const(half.right) == 2 \
                    and _binop(half.left, "-") and isinstance(half.left.left, c_ast.ID) \
                    and _is_id(half.left.right, base.name):
                return True
    return False


def _mid_vars(fn):
    if not hasattr(fn, "mid_vars"):
        fn.mid_vars = set()
    return fn.mid_vars
