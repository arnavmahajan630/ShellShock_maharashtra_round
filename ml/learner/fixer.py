"""Verified minimal fixer (ml_plan/03 §7.3 and §4.4). Package R1.

For one class, try the candidate edits in the table's order, at most five, each
at an AST site. The first candidate that passes every test is the result
(``kind`` ``minimal``). If none does, a correct variant that is not the learner's
code is returned (``kind`` ``reference``).

Results are cached by a hash of the source, the problem id and the class, so the
same mutant is not retested. A class stops at the first candidate that passes.
"""
from __future__ import annotations

import copy
import difflib
import hashlib
import re

from pycparser import c_ast, c_generator, c_parser

from ml.c_interp.preprocess import preprocess
from ml.runner import run_tests

MAX_CANDIDATES = 5

# M01 and M08 share one edit list (03 §7.3). Each entry is tried in order.
_M01_M08 = (
    "le_to_lt",
    "lt_to_le",
    "nminus1_to_n",
    "start1_to_0",
    "iplus1_to_i",
    "index_n_to_nminus1",
    "index1_to_0",
)

_RULES = {
    "M01": _M01_M08,
    "M08": _M01_M08,
    "M02": ("append_increment", "dec_to_inc", "hoist_update"),
    "M03": ("hoist_init",),
    "M04": ("float_numerator", "int_literal_to_float", "cast_onto_numerator"),
    "M05": ("init_zero", "init_one", "init_first_cell"),
    "M06": ("assign_to_eq",),
    "M07": ("delete_empty_body",),
    "M10": ("printf_to_return",),
    "D01": ("delete_else",),
    "D02": ("low_mid_plus", "high_mid_minus"),
    "D03": ("swap_with_temp", "save_first_cell"),
    "D04": ("wrap_single_pass", "outer_bound_once"),
    "D05": ("base_le0_0", "base_le0_1", "base_eq0_0"),
    "D06": ("rec_same_to_minus", "rec_plus_to_minus", "rec_mod_to_div"),
    "D07": ("combine_mul", "combine_add", "combine_call"),
    "D08": ("string_to_char", "array_eq_to_loop"),
}

_RULE_TEXT = {
    "le_to_lt": "<= to <",
    "lt_to_le": "< to <=",
    "nminus1_to_n": "n-1 to n",
    "start1_to_0": "start 1 to 0",
    "iplus1_to_i": "a[i+1] to a[i]",
    "index_n_to_nminus1": "a[n] to a[n-1]",
    "index1_to_0": "a[1] to a[0]",
    "append_increment": "i++ on the loop variable",
    "dec_to_inc": "i-- to i++",
    "hoist_update": "move update out of if",
    "hoist_init": "hoist init",
    "float_numerator": "float cast on numerator",
    "int_literal_to_float": "2 to 2.0",
    "cast_onto_numerator": "move cast inside",
    "init_zero": "init = 0",
    "init_one": "init = 1",
    "init_first_cell": "init = a[0]",
    "assign_to_eq": "= to ==",
    "delete_empty_body": "delete semicolon",
    "printf_to_return": "printf to return",
    "delete_else": "delete else branch",
    "low_mid_plus": "low = mid + 1",
    "high_mid_minus": "high = mid - 1",
    "swap_with_temp": "swap with temp",
    "save_first_cell": "save a[0]",
    "wrap_single_pass": "wrap single pass",
    "outer_bound_once": "i < 1 to i < n - 1",
    "base_le0_0": "base n <= 0 return 0",
    "base_le0_1": "base n <= 0 return 1",
    "base_eq0_0": "base n == 0 return 0",
    "rec_same_to_minus": "f(n) to f(n - 1)",
    "rec_plus_to_minus": "f(n + 1) to f(n - 1)",
    "rec_mod_to_div": "f(n % 10) to f(n / 10)",
    "combine_mul": "return y * f(x)",
    "combine_add": "return y + f(x)",
    "combine_call": "return f(x)",
    "string_to_char": "string to char",
    "array_eq_to_loop": "array == to char loop",
}

_LOOPS = (c_ast.For, c_ast.While, c_ast.DoWhile)
_GEN = c_generator.CGenerator()

# (code hash, problem id, class) -> probe dict. Run results are keyed separately
# so two classes do not re-execute the same source.
_probe_cache: dict = {}
_run_cache: dict = {}


def repair(problem, code, cls):
    """The verified fix for ``cls``. See the module docstring for the shape."""
    return probe(problem, code, cls)["fix"]


def probe(problem, code, cls):
    """Run the class's edits.

    Returns ``fix`` (the repair dict), ``base_frac`` (pass fraction of ``code``),
    ``gain`` (best change in pass fraction among the candidates that were run)
    and ``hit`` (True when a candidate passes every test).
    """
    key = (_digest(code), problem.get("problem_id"), cls)
    cached = _probe_cache.get(key)
    if cached is not None:
        return cached
    packed = _probe_uncached(problem, code, cls)
    _probe_cache[key] = packed
    return packed


def clear_cache():
    """Drop the in-process source and run caches. Tests use this between cases."""
    _probe_cache.clear()
    _run_cache.clear()


# ---------------------------------------------------------------- running

def _digest(code):
    return hashlib.sha256(code.encode("utf-8")).hexdigest()


def _run(problem, code):
    key = (problem.get("problem_id"), _digest(code))
    hit = _run_cache.get(key)
    if hit is None:
        hit = run_tests(problem, code)
        _run_cache[key] = hit
    return hit


def _fraction(problem, code):
    tests = _run(problem, code)["tests"]
    total = tests["total"]
    if not total:
        return 0.0
    return tests["passed"] / total


def _passes(problem, code):
    tests = _run(problem, code)["tests"]
    return tests["total"] > 0 and tests["passed"] == tests["total"]


def _probe_uncached(problem, code, cls):
    base = _fraction(problem, code)
    best_gain = 0.0
    for rule, source in _candidates(code, cls):
        frac = _fraction(problem, source)
        best_gain = max(best_gain, frac - base)
        if _passes(problem, source):
            fix = _minimal(code, source, rule)
            return {"fix": fix, "base_frac": base, "gain": frac - base, "hit": True}
    fix = _reference_fix(problem, code)
    return {"fix": fix, "base_frac": base, "gain": best_gain, "hit": False}


def _candidates(code, cls):
    """Up to five sources, table order, each different from ``code``."""
    names = _RULES.get(cls)
    if not names:
        return
    try:
        ast = _parse(code)
    except Exception:
        return
    baseline = _generate(ast)
    seen = set()
    produced = 0
    for name in names:
        maker = _EDIT[name]
        try:
            texts = maker(ast)
        except Exception:
            continue
        for text in texts:
            if text is None or text == baseline or text == code or text in seen:
                continue
            seen.add(text)
            produced += 1
            yield _RULE_TEXT[name], text
            if produced >= MAX_CANDIDATES:
                return


def _minimal(before, after, rule):
    return {
        "kind": "minimal",
        "code": after,
        "changed_lines": changed_lines(before, after),
        "rule": rule,
        "verified": True,
    }


def _reference_fix(problem, code):
    source = _reference_source(problem, code)
    return {
        "kind": "reference",
        "code": source,
        "changed_lines": changed_lines(code, source),
        "rule": None,
        "verified": False,
    }


def _reference_source(problem, code):
    """A correct variant whose text is not the learner's text."""
    variants = list(problem.get("correct_variants") or [])
    stripped = _squash(code)
    for variant in variants:
        if variant != code and _squash(variant) != stripped:
            return variant
    for variant in variants:
        if variant != code:
            return variant
    if variants:
        return variants[0]
    return code


def changed_lines(before, after):
    """1-based line numbers that differ. Insertions are numbered in ``after``."""
    old = before.splitlines()
    new = after.splitlines()
    lines = []
    matcher = difflib.SequenceMatcher(a=old, b=new)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        if tag == "delete":
            lines.extend(range(i1 + 1, i2 + 1))
        else:
            lines.extend(range(j1 + 1, j2 + 1))
    return lines


def _squash(text):
    return re.sub(r"\s+", "", text)


# ---------------------------------------------------------------- tree

def _parse(code):
    return c_parser.CParser().parse(preprocess(code))


def _generate(ast):
    return _GEN.visit(ast)


def _walk(node):
    if node is None:
        return
    yield node
    for _, child in node.children():
        yield from _walk(child)


def _func(ast):
    for ext in getattr(ast, "ext", None) or []:
        if isinstance(ext, c_ast.FuncDef):
            return ext
    raise ValueError("no function")


def _params(fn):
    decl = fn.decl.type
    if isinstance(decl, c_ast.FuncDecl) and decl.args:
        return list(decl.args.params or [])
    return []


def _is_array_param(param):
    return isinstance(param.type, (c_ast.ArrayDecl, c_ast.PtrDecl))


def _array_name(fn):
    for param in _params(fn):
        if _is_array_param(param):
            return param.name
    return None


def _size_name(fn):
    params = _params(fn)
    seen_array = False
    for param in params:
        if _is_array_param(param):
            seen_array = True
            continue
        if seen_array and isinstance(param, c_ast.Decl):
            return param.name
    for param in params:
        if param.name == "n":
            return "n"
    for param in params:
        if not _is_array_param(param):
            return param.name
    return "n"


def _scalar_name(fn):
    for param in _params(fn):
        if not _is_array_param(param):
            return param.name
    return "n"


def _fname(fn):
    return fn.decl.name


def _int_const(node):
    if isinstance(node, c_ast.Constant) and node.type == "int":
        try:
            return int(node.value, 0)
        except ValueError:
            return None
    if isinstance(node, c_ast.UnaryOp) and node.op == "-" and isinstance(node.expr, c_ast.Constant):
        value = _int_const(node.expr)
        return None if value is None else -value
    return None


def _const_int(value):
    if value < 0:
        return c_ast.UnaryOp("-", c_ast.Constant("int", str(-value)))
    return c_ast.Constant("int", str(value))


def _id(name):
    return c_ast.ID(name)


def _float_cast(expr):
    # pycparser's Typename takes (name, quals, align, type). align is unused here.
    typename = c_ast.Typename(None, [], None, c_ast.IdentifierType(["float"]))
    return c_ast.Cast(typename, expr)


def _is_float_cast(node):
    if not isinstance(node, c_ast.Cast):
        return False
    names = _type_names(node.to_type)
    return "float" in names or "double" in names


def _type_names(node):
    found = []
    for item in _walk(node):
        if isinstance(item, c_ast.IdentifierType):
            found.extend(item.names)
    return found


def _loops(ast):
    """Loops, outermost first."""
    found = []

    def walk(node, depth):
        for _, child in node.children():
            if isinstance(child, _LOOPS):
                found.append((depth, child))
                walk(child, depth + 1)
            else:
                walk(child, depth)

    walk(ast, 0)
    found.sort(key=lambda item: item[0])
    return [node for _, node in found]


def _enclosing_loop(ast, node):
    """The nearest loop that contains ``node``, by identity after one walk."""
    owner = {}

    def walk(current, loop):
        owner[id(current)] = loop
        nxt = current if isinstance(current, _LOOPS) else loop
        for _, child in current.children():
            walk(child, nxt)

    walk(ast, None)
    return owner.get(id(node))


def _parent_map(ast):
    parent = {}
    name = {}

    def walk(node):
        for attr, child in node.children():
            parent[id(child)] = node
            name[id(child)] = attr
            walk(child)

    walk(ast)
    return parent, name


def _block_index(attr):
    match = re.fullmatch(r"block_items\[(\d+)\]", attr or "")
    return None if match is None else int(match.group(1))


def _loop_var(loop):
    init = getattr(loop, "init", None)
    decls = []
    if isinstance(init, c_ast.DeclList):
        decls = list(init.decls or [])
    elif isinstance(init, c_ast.Decl):
        decls = [init]
    elif isinstance(init, c_ast.Assignment) and isinstance(init.lvalue, c_ast.ID):
        return init.lvalue.name
    for decl in decls:
        if isinstance(decl, c_ast.Decl) and decl.name:
            return decl.name
    cond = loop.cond
    if isinstance(cond, c_ast.BinaryOp) and isinstance(cond.left, c_ast.ID):
        return cond.left.name
    return None


def _copy_edit(ast, mutate):
    fresh = copy.deepcopy(ast)
    mutate(fresh)
    return _generate(fresh)


def _names_in(node):
    return {item.name for item in _walk(node) if isinstance(item, c_ast.ID)}


def _decl_names(fn):
    return {item.name for item in _walk(fn) if isinstance(item, c_ast.Decl) and item.name}


def _fresh_name(fn, preferred):
    used = _decl_names(fn) | _names_in(fn)
    if preferred not in used:
        return preferred
    for name in ("t", "tmp", "saved", "first", "p"):
        if name not in used:
            return name
    return preferred + "_"


def _snippet_items(source):
    wrapped = "void _snippet(int n, int a[], int b[]) {\n" + source + "\n}"
    ast = _parse(wrapped)
    return copy.deepcopy(_func(ast).body.block_items or [])


def _rel_op(loop):
    cond = loop.cond
    if isinstance(cond, c_ast.BinaryOp):
        return cond.op
    return None


def _bound_minus_one(expr):
    """The left operand of ``expr - 1``, or None."""
    if isinstance(expr, c_ast.BinaryOp) and expr.op == "-" and _int_const(expr.right) == 1:
        return expr.left
    return None


def _is_empty(node):
    return isinstance(node, c_ast.EmptyStatement)


def _controls(ast):
    return [node for node in _walk(ast) if isinstance(node, (c_ast.If, c_ast.For, c_ast.While))]


def _control_body(node):
    if isinstance(node, c_ast.If):
        return node.iftrue
    return node.stmt


def _set_control_body(node, body):
    if isinstance(node, c_ast.If):
        node.iftrue = body
    else:
        node.stmt = body


def _mid_names(fn):
    """Names initialised or assigned from a ``/ 2`` expression (the midpoint)."""
    names = set()
    for node in _walk(fn):
        init = None
        target = None
        if isinstance(node, c_ast.Decl) and node.init is not None:
            target, init = node.name, node.init
        elif isinstance(node, c_ast.Assignment) and node.op == "=" and isinstance(node.lvalue, c_ast.ID):
            target, init = node.lvalue.name, node.rvalue
        if target and _has_div2(init):
            names.add(target)
    return names


def _has_div2(node):
    for item in _walk(node):
        if isinstance(item, c_ast.BinaryOp) and item.op == "/" and _int_const(item.right) == 2:
            return True
    return False


def _self_calls(fn):
    name = _fname(fn)
    found = []
    for node in _walk(fn.body):
        if isinstance(node, c_ast.FuncCall) and isinstance(node.name, c_ast.ID) and node.name.name == name:
            found.append(node)
    return found


def _is_self_call(node, name):
    return isinstance(node, c_ast.FuncCall) and isinstance(node.name, c_ast.ID) and node.name.name == name


def _printf_call(node):
    return isinstance(node, c_ast.FuncCall) and isinstance(node.name, c_ast.ID) and node.name.name == "printf"


def _string_value(node):
    if isinstance(node, c_ast.Constant) and node.type == "string":
        raw = node.value
        if len(raw) >= 2 and raw[0] == '"' and raw[-1] == '"':
            return raw[1:-1]
    return None


def _returns_float(fn):
    names = _type_names(fn.decl)
    return "float" in names or "double" in names


def _uninit_decls(fn):
    """Scalar declarations with no initialiser, excluding for-loop index decls."""
    found = []
    parents, _ = _parent_map(fn)
    for node in _walk(fn.body):
        if not isinstance(node, c_ast.Decl) or node.name is None or node.init is not None:
            continue
        if isinstance(node.type, (c_ast.ArrayDecl, c_ast.PtrDecl)):
            continue
        parent = parents.get(id(node))
        if isinstance(parent, (c_ast.DeclList, c_ast.For)):
            continue
        found.append(node)
    return found


def _outer_decl(fn, name, skip):
    for node in _walk(fn.body):
        if node is skip:
            continue
        if isinstance(node, c_ast.Decl) and node.name == name:
            return node
    return None


# ---------------------------------------------------------------- edits
# Each edit returns a list of reprinted functions. One site per edit: the first
# match, outermost loop first. That keeps a later edit inside the five-candidate
# budget when earlier edits do not apply.


def _edit_le_to_lt(ast):
    return _retarget_bound(ast, "<=", "<")


def _edit_lt_to_le(ast):
    return _retarget_bound(ast, "<", "<=")


def _retarget_bound(ast, src, dst):
    for loop in _loops(ast):
        if isinstance(loop.cond, c_ast.BinaryOp) and loop.cond.op == src:
            ordinal = _loop_ordinal(ast, loop)

            def mutate(fresh, ordinal=ordinal, dst=dst):
                _loops(fresh)[ordinal].cond.op = dst
            return [_copy_edit(ast, mutate)]
    return []


def _loop_ordinal(ast, loop):
    """Index of ``loop`` in outermost-first order. Stable across deepcopy."""
    for ordinal, node in enumerate(_loops(ast)):
        if node is loop:
            return ordinal
    raise ValueError("loop is not in this tree")


def _edit_nminus1_to_n(ast):
    for loop in _loops(ast):
        cond = loop.cond
        if not isinstance(cond, c_ast.BinaryOp):
            continue
        side = "right" if _bound_minus_one(cond.right) is not None else None
        if side is None and _bound_minus_one(cond.left) is not None:
            side = "left"
        if side is None:
            continue
        ordinal = _loop_ordinal(ast, loop)

        def mutate(fresh, ordinal=ordinal, side=side):
            node = _loops(fresh)[ordinal]
            expr = getattr(node.cond, side)
            setattr(node.cond, side, _bound_minus_one(expr))
        return [_copy_edit(ast, mutate)]
    return []


def _edit_start1_to_0(ast):
    for loop in _loops(ast):
        var = _loop_var(loop)
        target = _init_one(loop, var)
        if target is None:
            continue
        ordinal = _loop_ordinal(ast, loop)
        kind = target[0]

        def mutate(fresh, ordinal=ordinal, kind=kind):
            node = _loops(fresh)[ordinal]
            _set_init_zero(_init_one(node, _loop_var(node)), kind)
        return [_copy_edit(ast, mutate)]
    return []


def _init_one(loop, var):
    init = getattr(loop, "init", None)
    decls = []
    if isinstance(init, c_ast.DeclList):
        decls = list(init.decls or [])
    elif isinstance(init, c_ast.Decl):
        decls = [init]
    for decl in decls:
        if isinstance(decl, c_ast.Decl) and decl.name == var and _int_const(decl.init) == 1:
            return ("decl", decl)
    if isinstance(init, c_ast.Assignment) and isinstance(init.lvalue, c_ast.ID):
        if init.lvalue.name == var and _int_const(init.rvalue) == 1:
            return ("assign", init)
    # ``int i = 1; while (...)`` — the assignment or decl sits in the enclosing
    # block, not in the loop header. Handled by a scan of assignments to ``var``.
    return None


def _set_init_zero(found, kind):
    if found is None:
        return
    node = found[1]
    if kind == "decl":
        node.init = _const_int(0)
    else:
        node.rvalue = _const_int(0)


def _edit_start1_decl_before_while(ast):
    """``int i = 1; while (i < n)`` has the 1 outside the loop header."""
    parents, _ = _parent_map(ast)
    for loop in _loops(ast):
        if isinstance(loop, c_ast.For):
            continue
        var = _loop_var(loop)
        if var is None:
            continue
        parent = parents.get(id(loop))
        if not isinstance(parent, c_ast.Compound):
            continue
        for stmt in parent.block_items or []:
            if stmt is loop:
                break
            if isinstance(stmt, c_ast.Decl) and stmt.name == var and _int_const(stmt.init) == 1:
                ordinal = _loop_ordinal(ast, loop)

                def mutate(fresh, ordinal=ordinal, var=var):
                    node = _loops(fresh)[ordinal]
                    parent_map, _ = _parent_map(fresh)
                    block = parent_map.get(id(node))
                    for item in block.block_items or []:
                        if isinstance(item, c_ast.Decl) and item.name == var and _int_const(item.init) == 1:
                            item.init = _const_int(0)
                            return
                return [_copy_edit(ast, mutate)]
    return []


def _edit_iplus1_to_i(ast):
    for node in _walk(ast):
        if not isinstance(node, c_ast.ArrayRef):
            continue
        sub = node.subscript
        if not (isinstance(sub, c_ast.BinaryOp) and sub.op == "+" and isinstance(sub.left, c_ast.ID)
                and _int_const(sub.right) == 1):
            continue
        name = sub.left.name

        def mutate(fresh, name=name):
            for item in _walk(fresh):
                if not isinstance(item, c_ast.ArrayRef):
                    continue
                inner = item.subscript
                if (isinstance(inner, c_ast.BinaryOp) and inner.op == "+" and isinstance(inner.left, c_ast.ID)
                        and inner.left.name == name and _int_const(inner.right) == 1):
                    item.subscript = _id(name)
                    return
        return [_copy_edit(ast, mutate)]
    return []


def _edit_index_n(ast):
    fn = _func(ast)
    size = _size_name(fn)
    for node in _walk(fn):
        if isinstance(node, c_ast.ArrayRef) and isinstance(node.subscript, c_ast.ID) and node.subscript.name == size:
            def mutate(fresh, size=size):
                for item in _walk(_func(fresh)):
                    if isinstance(item, c_ast.ArrayRef) and isinstance(item.subscript, c_ast.ID) and item.subscript.name == size:
                        item.subscript = c_ast.BinaryOp("-", _id(size), _const_int(1))
                        return
            return [_copy_edit(ast, mutate)]
    return []


def _edit_index1(ast):
    """``a[1]`` written as a first-element initialiser, not a loop subscript of ``i``."""
    parents, _ = _parent_map(ast)
    for node in _walk(ast):
        if not isinstance(node, c_ast.ArrayRef) or _int_const(node.subscript) != 1:
            continue
        if _enclosing_loop(ast, node) is not None and not isinstance(parents.get(id(node)), c_ast.Decl):
            # Inside a loop and not the initialiser of a declaration: leave it.
            # A declaration's initialiser can still sit lexically inside a loop
            # (M03). First-element context is a declaration or a bare assignment
            # that is not the loop's own index.
            if not isinstance(parents.get(id(node)), (c_ast.Decl, c_ast.Assignment)):
                continue
        holder = parents.get(id(node))
        if isinstance(holder, c_ast.Assignment) and _enclosing_loop(ast, node) is not None:
            continue
        base = node.name.name if isinstance(node.name, c_ast.ID) else None
        if base is None:
            continue

        def mutate(fresh, base=base):
            parent_map, _ = _parent_map(fresh)
            for item in _walk(fresh):
                if not isinstance(item, c_ast.ArrayRef) or _int_const(item.subscript) != 1:
                    continue
                if not isinstance(item.name, c_ast.ID) or item.name.name != base:
                    continue
                item.subscript = _const_int(0)
                return
        return [_copy_edit(ast, mutate)]
    return []


def _progress_increment(loop, var):
    """A ++ of some name other than ``var`` that is not nested inside an if."""
    parents, _ = _parent_map(loop)
    for node in _walk(loop):
        if not isinstance(node, c_ast.UnaryOp) or node.op not in ("p++", "++"):
            continue
        if not isinstance(node.expr, c_ast.ID) or node.expr.name == var:
            continue
        current = node
        nested = False
        while current is not None and current is not loop:
            current = parents.get(id(current))
            if isinstance(current, c_ast.If):
                nested = True
                break
        if not nested:
            return node
    return None


def _var_modified(loop, var):
    for node in _walk(loop):
        if isinstance(node, c_ast.UnaryOp) and node.op in ("p++", "++", "p--", "--"):
            if isinstance(node.expr, c_ast.ID) and node.expr.name == var:
                return True
        if isinstance(node, c_ast.Assignment) and isinstance(node.lvalue, c_ast.ID) and node.lvalue.name == var:
            return True
    return False


def _edit_append_inc(ast):
    for loop in _loops(ast):
        var = _loop_var(loop)
        if var is None or _var_modified(loop, var):
            continue
        ordinal = _loop_ordinal(ast, loop)
        # m02_wrong_var turns the progress i++ into n++. Appending i++ beside
        # that never finishes, because n grows in step with i. A ++ that is not
        # nested in an if is that progress update; point it back at the loop
        # variable. A ++ inside an if (count++) is the work, and is left alone.
        foreign = _progress_increment(loop, var)
        if foreign is not None:
            def mutate(fresh, ordinal=ordinal, var=var):
                node = _loops(fresh)[ordinal]
                target = _progress_increment(node, var)
                if target is not None:
                    target.expr.name = var
            return [_copy_edit(ast, mutate)]

        def mutate(fresh, ordinal=ordinal, var=var):
            node = _loops(fresh)[ordinal]
            inc = c_ast.UnaryOp("p++", _id(var))
            if isinstance(node.stmt, c_ast.Compound):
                items = list(node.stmt.block_items or [])
                items.append(inc)
                node.stmt.block_items = items
            elif node.stmt is None or _is_empty(node.stmt):
                node.stmt = c_ast.Compound([inc])
            else:
                node.stmt = c_ast.Compound([node.stmt, inc])
        return [_copy_edit(ast, mutate)]
    return []


def _edit_dec_to_inc(ast):
    for loop in _loops(ast):
        if _rel_op(loop) not in ("<", "<="):
            continue
        var = _loop_var(loop)
        ordinal = _loop_ordinal(ast, loop)

        def mutate(fresh, ordinal=ordinal, var=var):
            node = _loops(fresh)[ordinal]
            for item in _walk(node):
                if isinstance(item, c_ast.UnaryOp) and item.op in ("p--", "--"):
                    if isinstance(item.expr, c_ast.ID) and item.expr.name == var:
                        item.op = "p++" if item.op == "p--" else "++"
                        return
        text = _copy_edit(ast, mutate)
        if text != _generate(ast):
            return [text]
    return []


def _edit_hoist_update(ast):
    """Move the loop variable's ``i++`` from inside an ``if`` to the end of the body."""
    parents, _ = _parent_map(ast)
    for loop in _loops(ast):
        var = _loop_var(loop)
        if var is None or not isinstance(loop.stmt, c_ast.Compound):
            continue
        moved = None
        for node in _walk(loop.stmt):
            if not isinstance(node, c_ast.UnaryOp) or node.op not in ("p++", "++"):
                continue
            if not isinstance(node.expr, c_ast.ID) or node.expr.name != var:
                continue
            # The increment statement is the unary itself or an expression
            # statement. In this subset it is a direct block item.
            holder = parents.get(id(node))
            if isinstance(holder, c_ast.Compound) and holder is not loop.stmt:
                moved = node
                break
        if moved is None:
            continue
        ordinal = _loop_ordinal(ast, loop)

        def mutate(fresh, ordinal=ordinal, var=var):
            node = _loops(fresh)[ordinal]
            parent_map, names = _parent_map(fresh)
            for item in list(_walk(node.stmt)):
                if not isinstance(item, c_ast.UnaryOp) or item.op not in ("p++", "++"):
                    continue
                if not isinstance(item.expr, c_ast.ID) or item.expr.name != var:
                    continue
                holder = parent_map.get(id(item))
                if not isinstance(holder, c_ast.Compound) or holder is node.stmt:
                    continue
                attr = names.get(id(item))
                index = _block_index(attr)
                if index is None:
                    continue
                stmt = holder.block_items.pop(index)
                node.stmt.block_items.append(stmt)
                return
        return [_copy_edit(ast, mutate)]
    return []


def _edit_hoist_init(ast):
    """Drop a shadowing in-loop declaration, else move a leading ``t = 0`` out."""
    fn = _func(ast)
    parents, names = _parent_map(ast)
    for node in _walk(fn.body):
        if not isinstance(node, c_ast.Decl) or not node.name or node.init is None:
            continue
        loop = _enclosing_loop(ast, node)
        if loop is None:
            continue
        if _outer_decl(fn, node.name, node) is None:
            continue
        # Shadow of an outer scalar: the inverse of planting the declaration.
        target_name = node.name

        def mutate(fresh, target_name=target_name):
            parent_map, attrs = _parent_map(fresh)
            for item in list(_walk(_func(fresh).body)):
                if not isinstance(item, c_ast.Decl) or item.name != target_name or item.init is None:
                    continue
                if _enclosing_loop(fresh, item) is None:
                    continue
                holder = parent_map.get(id(item))
                index = _block_index(attrs.get(id(item)))
                if isinstance(holder, c_ast.Compound) and index is not None:
                    del holder.block_items[index]
                    return
        return [_copy_edit(ast, mutate)]

    for loop in _loops(ast):
        body = loop.stmt
        if not isinstance(body, c_ast.Compound) or not body.block_items:
            continue
        first = body.block_items[0]
        if not isinstance(first, c_ast.Assignment) or first.op != "=":
            continue
        if not isinstance(first.lvalue, c_ast.ID):
            continue
        if _int_const(first.rvalue) is None:
            continue
        if _outer_decl(fn, first.lvalue.name, first) is None and not _param_named(fn, first.lvalue.name):
            continue
        ordinal = _loop_ordinal(ast, loop)

        def mutate(fresh, ordinal=ordinal):
            node = _loops(fresh)[ordinal]
            if not isinstance(node.stmt, c_ast.Compound) or not node.stmt.block_items:
                return
            stmt = node.stmt.block_items.pop(0)
            parent_map, attrs = _parent_map(fresh)
            holder = parent_map.get(id(node))
            index = _block_index(attrs.get(id(node)))
            if isinstance(holder, c_ast.Compound) and index is not None:
                holder.block_items.insert(index, stmt)
        return [_copy_edit(ast, mutate)]
    return []


def _param_named(fn, name):
    return any(param.name == name for param in _params(fn))


def _edit_float_numerator(ast):
    fn = _func(ast)
    if not _returns_float(fn) and not any(isinstance(node, c_ast.Decl) and "float" in _type_names(node.type)
                                          for node in _walk(fn)):
        # Still try: a float return is the M04 shape. A non-float function has
        # nothing for this edit to mean.
        if not _returns_float(fn):
            return []
    for node in _walk(fn):
        if isinstance(node, c_ast.BinaryOp) and node.op == "/" and not _is_float_cast(node.left):
            if _is_float_cast(node.left) or _already_float(node.left):
                continue

            def mutate(fresh):
                for item in _walk(_func(fresh)):
                    if isinstance(item, c_ast.BinaryOp) and item.op == "/" and not _is_float_cast(item.left):
                        if _already_float(item.left):
                            continue
                        item.left = _float_cast(item.left)
                        return
            return [_copy_edit(ast, mutate)]
    return []


def _already_float(node):
    if isinstance(node, c_ast.Constant) and node.type in ("float", "double"):
        return True
    return _is_float_cast(node)


def _edit_int_literal(ast):
    for node in _walk(ast):
        if not isinstance(node, c_ast.BinaryOp) or node.op not in ("/", "*"):
            continue
        for side in ("left", "right"):
            child = getattr(node, side)
            if isinstance(child, c_ast.Constant) and child.type == "int" and node.op == "/":
                def mutate(fresh):
                    for item in _walk(fresh):
                        if not isinstance(item, c_ast.BinaryOp) or item.op != "/":
                            continue
                        for which in ("left", "right"):
                            leaf = getattr(item, which)
                            if isinstance(leaf, c_ast.Constant) and leaf.type == "int":
                                leaf.type = "double"
                                leaf.value = leaf.value + ".0"
                                return
                return [_copy_edit(ast, mutate)]
    return []


def _edit_cast_inside(ast):
    """``(float)(a / b)`` becomes ``(float)a / b``."""
    for node in _walk(ast):
        if not _is_float_cast(node) or not isinstance(node.expr, c_ast.BinaryOp) or node.expr.op != "/":
            continue

        def mutate(fresh):
            parent_map, _ = _parent_map(fresh)
            for item in list(_walk(fresh)):
                if not _is_float_cast(item) or not isinstance(item.expr, c_ast.BinaryOp) or item.expr.op != "/":
                    continue
                div = item.expr
                div.left = _float_cast(div.left)
                holder = parent_map.get(id(item))
                _replace_child(holder, item, div)
                return
        return [_copy_edit(ast, mutate)]
    return []


def _replace_child(parent, old, new):
    if parent is None:
        return
    for attr, child in list(parent.children()):
        if child is old:
            if attr.startswith("block_items["):
                index = _block_index(attr)
                parent.block_items[index] = new
            else:
                setattr(parent, attr.split("[")[0], new)
            return


def _edit_init(ast, value_kind):
    fn = _func(ast)
    decls = _uninit_decls(fn)
    if not decls:
        return []
    array = _array_name(fn)
    if value_kind == "cell" and array is None:
        return []
    name = decls[0].name

    def mutate(fresh, name=name, value_kind=value_kind, array=array):
        for decl in _uninit_decls(_func(fresh)):
            if decl.name != name:
                continue
            if value_kind == "zero":
                decl.init = _const_int(0)
            elif value_kind == "one":
                decl.init = _const_int(1)
            else:
                decl.init = c_ast.ArrayRef(_id(array), _const_int(0))
            return
    return [_copy_edit(ast, mutate)]


def _edit_assign_to_eq(ast):
    parents, _ = _parent_map(ast)
    for node in _walk(ast):
        if not isinstance(node, c_ast.Assignment) or node.op != "=":
            continue
        holder = parents.get(id(node))
        if not isinstance(holder, (c_ast.If, c_ast.While, c_ast.DoWhile, c_ast.For)):
            continue
        if isinstance(holder, c_ast.For) and holder.cond is not node:
            continue
        if isinstance(holder, (c_ast.If, c_ast.While, c_ast.DoWhile)) and holder.cond is not node:
            continue

        def mutate(fresh):
            parent_map, _ = _parent_map(fresh)
            for item in _walk(fresh):
                if not isinstance(item, c_ast.Assignment) or item.op != "=":
                    continue
                owner = parent_map.get(id(item))
                if isinstance(owner, (c_ast.If, c_ast.While, c_ast.DoWhile)) and owner.cond is item:
                    owner.cond = c_ast.BinaryOp("==", item.lvalue, item.rvalue)
                    return
                if isinstance(owner, c_ast.For) and owner.cond is item:
                    owner.cond = c_ast.BinaryOp("==", item.lvalue, item.rvalue)
                    return
        return [_copy_edit(ast, mutate)]
    return []


def _edit_delete_semi(ast):
    """Reattach the statement that follows an empty ``if`` / ``for`` / ``while`` body.

    The stray semicolon is an ``EmptyStatement`` in the tree (the generator's
    ``if (c); { ... }`` shape), so this is a tree edit, not a coordinate splice.
    """
    parents, names = _parent_map(ast)
    for node in _controls(ast):
        if not _is_empty(_control_body(node)):
            continue
        holder = parents.get(id(node))
        index = _block_index(names.get(id(node)))
        if not isinstance(holder, c_ast.Compound) or index is None:
            continue
        items = holder.block_items or []
        if index + 1 >= len(items):
            continue
        # Match this control by its position among empty-bodied controls.
        ordinal = _empty_ordinal(ast, node)

        def mutate(fresh, ordinal=ordinal):
            parent_map, attrs = _parent_map(fresh)
            target = _empty_controls(fresh)[ordinal]
            block = parent_map.get(id(target))
            index = _block_index(attrs.get(id(target)))
            nxt = block.block_items[index + 1]
            _set_control_body(target, nxt)
            del block.block_items[index + 1]
        return [_copy_edit(ast, mutate)]
    return []


def _empty_controls(ast):
    return [node for node in _controls(ast) if _is_empty(_control_body(node))]


def _empty_ordinal(ast, node):
    return _empty_controls(ast).index(node)


def _edit_printf(ast):
    parents, names = _parent_map(ast)
    for node in _walk(ast):
        if not _printf_call(node):
            continue
        mode = _printf_mode(node)
        if mode is None:
            continue
        holder = parents.get(id(node))
        index = _block_index(names.get(id(node)))
        as_item = isinstance(holder, c_ast.Compound) and index is not None
        as_body = isinstance(holder, (c_ast.If, c_ast.While, c_ast.For)) and _control_body(holder) is node
        if not as_item and not as_body:
            continue
        ordinal = _printf_ordinal(ast, node)

        def mutate(fresh, ordinal=ordinal, mode=mode):
            parent_map, attrs = _parent_map(fresh)
            call = _printf_nodes(fresh)[ordinal]
            expr = _printf_expr(call, mode)
            block = parent_map.get(id(call))
            index = _block_index(attrs.get(id(call)))
            if isinstance(block, c_ast.Compound) and index is not None:
                block.block_items[index] = c_ast.Return(expr)
                nxt = block.block_items[index + 1] if index + 1 < len(block.block_items) else None
                if isinstance(nxt, c_ast.Return) and _int_const(nxt.expr) == 0:
                    del block.block_items[index + 1]
                return
            if isinstance(block, (c_ast.If, c_ast.While, c_ast.For)) and _control_body(block) is call:
                _set_control_body(block, c_ast.Return(expr))
        return [_copy_edit(ast, mutate)]
    return []


def _printf_mode(node):
    args = list(node.args.exprs) if node.args is not None else []
    if len(args) >= 2:
        return "value"
    if len(args) == 1 and _string_value(args[0]) in ("yes", "no"):
        return "bool"
    return None


def _printf_expr(call, mode):
    args = list(call.args.exprs)
    if mode == "value":
        return args[1]
    return _const_int(1 if _string_value(args[0]) == "yes" else 0)


def _printf_nodes(ast):
    return [node for node in _walk(ast) if _printf_call(node)]


def _printf_ordinal(ast, node):
    return _printf_nodes(ast).index(node)


def _edit_delete_else(ast):
    for node in _walk(ast):
        if not isinstance(node, c_ast.If) or node.iffalse is None:
            continue
        if not _is_bug_else(node.iffalse):
            continue

        def mutate(fresh):
            for item in _walk(fresh):
                if isinstance(item, c_ast.If) and item.iffalse is not None and _is_bug_else(item.iffalse):
                    item.iffalse = None
                    return
        return [_copy_edit(ast, mutate)]
    return []


def _is_bug_else(node):
    if isinstance(node, c_ast.Return):
        return _int_const(node.expr) is not None
    if isinstance(node, c_ast.Assignment) and node.op == "=" and _int_const(node.rvalue) == 0:
        return True
    if isinstance(node, c_ast.Compound) and len(node.block_items or []) == 1:
        return _is_bug_else(node.block_items[0])
    return False


def _edit_mid(ast, direction):
    """``name = mid`` becomes ``name = mid + 1`` (low) or ``mid - 1`` (high)."""
    fn = _func(ast)
    mids = _mid_names(fn)
    if not mids:
        return []
    low, high = _window_names(fn)
    for node in _walk(fn):
        if not isinstance(node, c_ast.Assignment) or node.op != "=" or not isinstance(node.lvalue, c_ast.ID):
            continue
        if not isinstance(node.rvalue, c_ast.ID) or node.rvalue.name not in mids:
            continue
        lhs = node.lvalue.name
        if direction == "+" and high and lhs == high:
            continue
        if direction == "-" and low and lhs == low:
            continue
        if direction == "+" and low and lhs != low and high and lhs == high:
            continue
        target = lhs

        def mutate(fresh, target=target, direction=direction, mids=mids):
            for item in _walk(_func(fresh)):
                if not isinstance(item, c_ast.Assignment) or not isinstance(item.lvalue, c_ast.ID):
                    continue
                if item.lvalue.name != target or not isinstance(item.rvalue, c_ast.ID):
                    continue
                if item.rvalue.name not in mids:
                    continue
                item.rvalue = c_ast.BinaryOp(direction, _id(item.rvalue.name), _const_int(1))
                return
        return [_copy_edit(ast, mutate)]
    return []


def _window_names(fn):
    """(low, high) from a ``low <= high`` test, else from initialisers 0 and n-1."""
    for node in _walk(fn):
        if isinstance(node, c_ast.BinaryOp) and node.op in ("<=", "<"):
            if isinstance(node.left, c_ast.ID) and isinstance(node.right, c_ast.ID):
                return node.left.name, node.right.name
    low = high = None
    for node in _walk(fn):
        if isinstance(node, c_ast.Decl) and _int_const(node.init) == 0:
            low = low or node.name
        if isinstance(node, c_ast.Decl) and _bound_minus_one(node.init) is not None:
            high = high or node.name
    return low, high


def _edit_swap_temp(ast):
    fn = _func(ast)
    for compound in [node for node in _walk(fn) if isinstance(node, c_ast.Compound)]:
        items = compound.block_items or []
        for index in range(len(items) - 1):
            first, second = items[index], items[index + 1]
            pair = _overwrite_pair(first, second)
            if pair is None:
                continue
            temp = _fresh_name(fn, "t")

            def mutate(fresh, index=index, temp=temp):
                for block in [node for node in _walk(fresh) if isinstance(node, c_ast.Compound)]:
                    body = block.block_items or []
                    for cursor in range(len(body) - 1):
                        got = _overwrite_pair(body[cursor], body[cursor + 1])
                        if got is None:
                            continue
                        left, right = got
                        decl = _snippet_items(f"int {temp} = a[0];")[0]
                        decl.init = copy.deepcopy(left)
                        body[cursor:cursor + 2] = [
                            decl,
                            c_ast.Assignment("=", copy.deepcopy(left), copy.deepcopy(right)),
                            c_ast.Assignment("=", copy.deepcopy(right), _id(temp)),
                        ]
                        block.block_items = body
                        return
            return [_copy_edit(ast, mutate)]
    return []


def _overwrite_pair(first, second):
    if not isinstance(first, c_ast.Assignment) or not isinstance(second, c_ast.Assignment):
        return None
    if first.op != "=" or second.op != "=":
        return None
    if not isinstance(first.lvalue, c_ast.ArrayRef) or not isinstance(first.rvalue, c_ast.ArrayRef):
        return None
    if not isinstance(second.lvalue, c_ast.ArrayRef) or not isinstance(second.rvalue, c_ast.ArrayRef):
        return None
    if _expr_key(second.lvalue) == _expr_key(first.rvalue) and _expr_key(second.rvalue) == _expr_key(first.lvalue):
        return first.lvalue, first.rvalue
    return None


def _expr_key(node):
    return _GEN.visit(node)


def _edit_save_first(ast):
    fn = _func(ast)
    array = _array_name(fn) or "a"
    parents, names = _parent_map(ast)
    for node in _walk(fn):
        if not isinstance(node, c_ast.Assignment) or node.op != "=":
            continue
        if not isinstance(node.rvalue, c_ast.ArrayRef) or _int_const(node.rvalue.subscript) != 0:
            continue
        if not isinstance(node.lvalue, c_ast.ArrayRef):
            continue
        holder = parents.get(id(node))
        index = _block_index(names.get(id(node)))
        if not isinstance(holder, c_ast.Compound) or index is None or index == 0:
            continue
        # A shift loop should sit somewhere before this store.
        if not any(isinstance(item, _LOOPS) for item in holder.block_items[:index]):
            continue
        temp = _fresh_name(fn, "first")

        def mutate(fresh, array=array, temp=temp):
            parent_map, attrs = _parent_map(fresh)
            for item in _walk(fresh):
                if not isinstance(item, c_ast.Assignment) or not isinstance(item.rvalue, c_ast.ArrayRef):
                    continue
                if _int_const(item.rvalue.subscript) != 0 or not isinstance(item.lvalue, c_ast.ArrayRef):
                    continue
                block = parent_map.get(id(item))
                index = _block_index(attrs.get(id(item)))
                if not isinstance(block, c_ast.Compound) or index is None:
                    continue
                decl = _snippet_items(f"int {temp} = a[0];")[0]
                decl.init = c_ast.ArrayRef(_id(array), _const_int(0))
                # Place the save before the first loop in this block.
                insert_at = 0
                for cursor, stmt in enumerate(block.block_items):
                    if isinstance(stmt, _LOOPS):
                        insert_at = cursor
                        break
                block.block_items.insert(insert_at, decl)
                item.rvalue = _id(temp)
                return
        return [_copy_edit(ast, mutate)]
    return []


def _edit_wrap_pass(ast):
    fn = _func(ast)
    size = _size_name(fn)
    parents, names = _parent_map(ast)
    loops = _loops(ast)
    if not loops:
        return []
    # Prefer a loop that is not already nested, so a one-pass body gets the wrapper.
    target = loops[0]
    for loop in loops:
        if _enclosing_loop(ast, loop) is None:
            target = loop
            break
    holder = parents.get(id(target))
    if not isinstance(holder, c_ast.Compound):
        return []
    ordinal = _loop_ordinal(ast, target)

    def mutate(fresh, ordinal=ordinal, size=size):
        node = _loops(fresh)[ordinal]
        parent_map, attrs = _parent_map(fresh)
        block = parent_map.get(id(node))
        index = _block_index(attrs.get(id(node)))
        if not isinstance(block, c_ast.Compound) or index is None:
            return
        wrapper = _snippet_items(f"for (int p = 0; p < {size} - 1; p++) {{\n  a[0] = a[0];\n}}")[0]
        wrapper.stmt = c_ast.Compound([block.block_items.pop(index)])
        block.block_items.insert(index, wrapper)
    return [_copy_edit(ast, mutate)]


def _edit_outer_once(ast):
    for loop in _loops(ast):
        cond = loop.cond
        if not isinstance(cond, c_ast.BinaryOp) or cond.op != "<":
            continue
        if _int_const(cond.right) != 1:
            continue
        if not any(isinstance(node, _LOOPS) for node in _walk(loop.stmt)):
            continue
        size = _size_name(_func(ast))
        ordinal = _loop_ordinal(ast, loop)

        def mutate(fresh, ordinal=ordinal, size=size):
            node = _loops(fresh)[ordinal]
            node.cond.right = c_ast.BinaryOp("-", _id(size), _const_int(1))
        return [_copy_edit(ast, mutate)]
    return []


def _edit_base(ast, cond_src, value):
    fn = _func(ast)
    if not _self_calls(fn):
        return []
    # Already has a base that returns before any call: don't stack another.
    name = _scalar_name(fn)

    def mutate(fresh, name=name, cond_src=cond_src, value=value):
        body = _func(fresh).body
        items = list(body.block_items or [])
        stmt = _snippet_items(f"if ({cond_src.replace('n', name)}) return {value};")[0]
        # The snippet renames only the source text's n. Rebuild when the param
        # is not n so the condition uses the real parameter.
        if name != "n":
            stmt = _snippet_items(f"if ({name} {cond_src[len('n'):]}) return {value};")[0]
        items.insert(0, stmt)
        body.block_items = items
    text = _copy_edit(ast, mutate)
    return [text]


def _edit_rec_arg(ast, kind):
    fn = _func(ast)
    name = _fname(fn)
    param = _scalar_name(fn)
    for call in _self_calls(fn):
        if call.args is None:
            continue
        for index, arg in enumerate(call.args.exprs):
            if kind == "same" and isinstance(arg, c_ast.ID) and arg.name == param:
                pass
            elif kind == "plus" and isinstance(arg, c_ast.BinaryOp) and arg.op == "+" and _int_const(arg.right) == 1:
                if not isinstance(arg.left, c_ast.ID) or arg.left.name != param:
                    continue
            elif kind == "mod" and isinstance(arg, c_ast.BinaryOp) and arg.op == "%" and isinstance(arg.left, c_ast.ID):
                pass
            else:
                continue
            which = index
            mode = kind

            def mutate(fresh, which=which, mode=mode, name=name, param=param):
                for call in _self_calls(_func(fresh)):
                    if call.args is None or which >= len(call.args.exprs):
                        continue
                    arg = call.args.exprs[which]
                    if mode == "same" and isinstance(arg, c_ast.ID):
                        call.args.exprs[which] = c_ast.BinaryOp("-", _id(arg.name), _const_int(1))
                        return
                    if mode == "plus" and isinstance(arg, c_ast.BinaryOp) and arg.op == "+":
                        arg.op = "-"
                        return
                    if mode == "mod" and isinstance(arg, c_ast.BinaryOp) and arg.op == "%":
                        arg.op = "/"
                        return
            return [_copy_edit(ast, mutate)]
    return []


def _edit_combine(ast, op):
    fn = _func(ast)
    name = _fname(fn)
    parents, names_map = _parent_map(ast)
    for node in _walk(fn.body):
        if not _is_self_call(node, name):
            continue
        holder = parents.get(id(node))
        index = _block_index(names_map.get(id(node)))
        if not isinstance(holder, c_ast.Compound) or index is None:
            continue
        items = holder.block_items or []
        if index + 1 >= len(items) or not isinstance(items[index + 1], c_ast.Return):
            continue
        mode = op

        def mutate(fresh, mode=mode, name=name):
            parent_map, attrs = _parent_map(fresh)
            for item in _walk(_func(fresh).body):
                if not _is_self_call(item, name):
                    continue
                block = parent_map.get(id(item))
                index = _block_index(attrs.get(id(item)))
                if not isinstance(block, c_ast.Compound) or index is None:
                    continue
                if index + 1 >= len(block.block_items) or not isinstance(block.block_items[index + 1], c_ast.Return):
                    continue
                call = block.block_items[index]
                ret = block.block_items[index + 1]
                if mode is None:
                    expr = call
                else:
                    expr = c_ast.BinaryOp(mode, ret.expr, call)
                block.block_items[index:index + 2] = [c_ast.Return(expr)]
                return
        return [_copy_edit(ast, mutate)]
    return []


def _edit_string_to_char(ast):
    changed = False
    for node in _walk(ast):
        if isinstance(node, c_ast.Constant) and _string_value(node) is not None and len(_string_value(node)) == 1:
            changed = True
            break
    if not changed:
        return []

    def mutate(fresh):
        for item in _walk(fresh):
            if not isinstance(item, c_ast.BinaryOp) or item.op not in ("==", "!="):
                continue
            for side in ("left", "right"):
                child = getattr(item, side)
                text = _string_value(child)
                if text is not None and len(text) == 1:
                    setattr(item, side, c_ast.Constant("char", "'" + text + "'"))
    text = _copy_edit(ast, mutate)
    if text == _generate(ast):
        return []
    return [text]


def _edit_array_eq(ast):
    """``if (s == t) return 1; else return 0;`` becomes a character walk.

    03 §7.3 allows the reference fallback when this rewrite does not apply.
    There is no bank site for the operator; the rewrite is still attempted.
    """
    for node in _walk(ast):
        if not isinstance(node, c_ast.If) or not isinstance(node.cond, c_ast.BinaryOp) or node.cond.op != "==":
            continue
        left, right = node.cond.left, node.cond.right
        if not isinstance(left, c_ast.ID) or not isinstance(right, c_ast.ID):
            continue
        if not isinstance(node.iftrue, c_ast.Return) or not isinstance(node.iffalse, c_ast.Return):
            continue
        s, t = left.name, right.name

        def mutate(fresh, s=s, t=t):
            parent_map, attrs = _parent_map(fresh)
            for item in _walk(fresh):
                if not isinstance(item, c_ast.If) or not isinstance(item.cond, c_ast.BinaryOp):
                    continue
                if item.cond.op != "==":
                    continue
                if not isinstance(item.cond.left, c_ast.ID) or item.cond.left.name != s:
                    continue
                block = parent_map.get(id(item))
                index = _block_index(attrs.get(id(item)))
                if not isinstance(block, c_ast.Compound) or index is None:
                    continue
                body = _snippet_items(
                    f"int i = 0;\n"
                    f"while ({s}[i] != '\\0' && {s}[i] == {t}[i]) i++;\n"
                    f"return {s}[i] == {t}[i];"
                )
                block.block_items[index:index + 1] = body
                return
        return [_copy_edit(ast, mutate)]
    return []


def _make(name, fn):
    def run(ast):
        return fn(ast)
    run.__name__ = name
    return run


_EDIT = {
    "le_to_lt": _make("le_to_lt", _edit_le_to_lt),
    "lt_to_le": _make("lt_to_le", _edit_lt_to_le),
    "nminus1_to_n": _make("nminus1_to_n", _edit_nminus1_to_n),
    "start1_to_0": _make("start1_to_0", lambda ast: _edit_start1_to_0(ast) or _edit_start1_decl_before_while(ast)),
    "iplus1_to_i": _make("iplus1_to_i", _edit_iplus1_to_i),
    "index_n_to_nminus1": _make("index_n", _edit_index_n),
    "index1_to_0": _make("index1", _edit_index1),
    "append_increment": _make("append", _edit_append_inc),
    "dec_to_inc": _make("dec", _edit_dec_to_inc),
    "hoist_update": _make("hoist_update", _edit_hoist_update),
    "hoist_init": _make("hoist_init", _edit_hoist_init),
    "float_numerator": _make("float_num", _edit_float_numerator),
    "int_literal_to_float": _make("int_lit", _edit_int_literal),
    "cast_onto_numerator": _make("cast_in", _edit_cast_inside),
    "init_zero": _make("init0", lambda ast: _edit_init(ast, "zero")),
    "init_one": _make("init1", lambda ast: _edit_init(ast, "one")),
    "init_first_cell": _make("inita", lambda ast: _edit_init(ast, "cell")),
    "assign_to_eq": _make("eq", _edit_assign_to_eq),
    "delete_empty_body": _make("semi", _edit_delete_semi),
    "printf_to_return": _make("printf", _edit_printf),
    "delete_else": _make("else", _edit_delete_else),
    "low_mid_plus": _make("low", lambda ast: _edit_mid(ast, "+")),
    "high_mid_minus": _make("high", lambda ast: _edit_mid(ast, "-")),
    "swap_with_temp": _make("swap", _edit_swap_temp),
    "save_first_cell": _make("save", _edit_save_first),
    "wrap_single_pass": _make("wrap", _edit_wrap_pass),
    "outer_bound_once": _make("once", _edit_outer_once),
    "base_le0_0": _make("b0", lambda ast: _edit_base(ast, "n <= 0", 0)),
    "base_le0_1": _make("b1", lambda ast: _edit_base(ast, "n <= 0", 1)),
    "base_eq0_0": _make("b00", lambda ast: _edit_base(ast, "n == 0", 0)),
    "rec_same_to_minus": _make("same", lambda ast: _edit_rec_arg(ast, "same")),
    "rec_plus_to_minus": _make("plus", lambda ast: _edit_rec_arg(ast, "plus")),
    "rec_mod_to_div": _make("mod", lambda ast: _edit_rec_arg(ast, "mod")),
    "combine_mul": _make("mul", lambda ast: _edit_combine(ast, "*")),
    "combine_add": _make("add", lambda ast: _edit_combine(ast, "+")),
    "combine_call": _make("call", lambda ast: _edit_combine(ast, None)),
    "string_to_char": _make("str", _edit_string_to_char),
    "array_eq_to_loop": _make("arr", _edit_array_eq),
}
