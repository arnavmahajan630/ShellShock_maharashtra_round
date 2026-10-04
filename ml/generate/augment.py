"""Style augmentations (03 §3.5.2) and CORRECT near-miss transforms (03 §3.5.3).

Every edit is meant to keep the program's result. ``verify`` re-runs the tests
and drops anything that does not.
"""
from __future__ import annotations

import copy

from pycparser import c_ast, c_generator

from ml.generate.ops_main import parse_ast

_GEN = c_generator.CGenerator()
_BUILTINS = frozenset({
    "printf", "strlen", "fire", "launch", "open_door", "close_door", "scan",
})
_POOLS = (
    ("sum", "s", "tot", "total", "ans", "res"),
    ("i", "j", "k", "idx", "x"),
    ("cnt", "count", "c"),
)
_COMMENTS = (
    "// yaha total add karo",
    "// loop over every element",
    "// check karo condition",
    "// return the answer",
    "// yaha count badhao",
    "// bas itna hi",
    "// running total lives here",
    "// agla element dekho",
)
_INCR_FORMS = ("p++", "++", "+=", "add")


def _walk(node):
    if node is None:
        return
    yield node
    for _, child in node.children():
        yield from _walk(child)


def _fn(ast):
    for node in getattr(ast, "ext", None) or []:
        if isinstance(node, c_ast.FuncDef):
            return node
    return None


def _reprint(ast):
    return _GEN.visit(ast)


def _parse(code):
    return parse_ast(code)


def _const(value):
    return c_ast.Constant("int", str(value))


def _id(name):
    return c_ast.ID(name)


def _is_one(node):
    return isinstance(node, c_ast.Constant) and node.type == "int" and node.value == "1"


def _int_value(node):
    if isinstance(node, c_ast.Constant) and node.type == "int":
        try:
            return int(node.value, 0)
        except ValueError:
            return None
    return None


def _type_names(node):
    seen = 0
    while node is not None and not isinstance(node, c_ast.IdentifierType) and seen < 6:
        if isinstance(node, c_ast.ArrayDecl):
            return ["array"]
        node = getattr(node, "type", None)
        seen += 1
    if isinstance(node, c_ast.IdentifierType):
        return list(node.names or [])
    return []


def _int_decl(decl):
    if not isinstance(decl, c_ast.Decl) or not decl.name:
        return False
    names = _type_names(decl.type)
    return "int" in names or "char" in names


def _items(compound):
    if compound is None:
        return None
    if compound.block_items is None:
        compound.block_items = []
    return compound.block_items


def _scalar_names(fn):
    names = []
    args = fn.decl.type.args if fn.decl is not None else None
    for param in (args.params if args else []):
        if _int_decl(param):
            names.append(param.name)
    for item in (fn.body.block_items if fn.body is not None else None) or []:
        if _int_decl(item):
            names.append(item.name)
    return names


def _declared(fn):
    names = []
    for node in _walk(fn):
        if isinstance(node, c_ast.Decl) and node.name and node is not fn.decl:
            names.append(node.name)
    return names


# ---------------------------------------------------------------- style


def _insert_comment(code, text):
    if text in code:
        return None
    brace = code.find("{")
    if brace < 0:
        return None
    return code[: brace + 1] + "\n  " + text + code[brace + 1 :]


def _indent_more(code, extra):
    pad = " " * extra
    lines = []
    changed = False
    for line in code.split("\n"):
        if line.startswith("  ") or line.startswith("\t"):
            lines.append(pad + line)
            changed = True
        else:
            lines.append(line)
    if not changed:
        return None
    return "\n".join(lines)


def _to_allman(code):
    try:
        text = _reprint(_parse(code))
    except Exception:
        return None
    return None if text.strip() == code.strip() else text


def _to_kr(code):
    base = _to_allman(code) or code
    lines = base.split("\n")
    out = []
    changed = False
    for line in lines:
        if line.strip() == "{" and out:
            out[-1] = out[-1] + " {"
            changed = True
        else:
            out.append(line)
    if not changed:
        return None
    return "\n".join(out)


def _simple_stmt(node):
    return not isinstance(node, (c_ast.If, c_ast.For, c_ast.While, c_ast.DoWhile, c_ast.Compound, c_ast.Switch))


def _to_nobrace(code):
    try:
        ast = _parse(code)
    except Exception:
        return None
    changed = False

    def visit(node):
        nonlocal changed
        if isinstance(node, c_ast.If):
            node.iftrue = _strip(node.iftrue)
            node.iffalse = _strip(node.iffalse)
        elif isinstance(node, (c_ast.For, c_ast.While, c_ast.DoWhile)):
            node.stmt = _strip(node.stmt)
        for _, child in node.children():
            visit(child)

    def _strip(body):
        nonlocal changed
        if (isinstance(body, c_ast.Compound) and body.block_items and len(body.block_items) == 1
                and _simple_stmt(body.block_items[0])):
            changed = True
            return body.block_items[0]
        return body

    visit(ast)
    if not changed:
        return None
    return _reprint(ast)


def _add_parens(code):
    changed = False
    lines = []
    for line in code.split("\n"):
        stripped = line.strip()
        if stripped.startswith("return ") and stripped.endswith(";") and not stripped.startswith("return ("):
            expr = stripped[len("return "): -1].strip()
            if expr and ";" not in expr:
                indent = line[: len(line) - len(line.lstrip())]
                lines.append(f"{indent}return ({expr});")
                changed = True
                continue
        lines.append(line)
    if not changed:
        return None
    return "\n".join(lines)


def _add_unused(code, name):
    try:
        ast = _parse(code)
    except Exception:
        return None
    fn = _fn(ast)
    if fn is None or fn.body is None:
        return None
    if name in _declared(fn) or name == fn.decl.name:
        return None
    decl = c_ast.Decl(
        name, [], [], [], [],
        c_ast.TypeDecl(name, [], None, c_ast.IdentifierType(["int"])),
        _const(0), None,
    )
    items = _items(fn.body)
    items.insert(0, decl)
    return _reprint(ast)


def _add_debug(code):
    try:
        ast = _parse(code)
    except Exception:
        return None
    fn = _fn(ast)
    if fn is None or fn.body is None:
        return None
    names = _scalar_names(fn)
    arg = _id(names[0]) if names else _const(0)
    call = c_ast.FuncCall(_id("printf"), c_ast.ExprList([
        c_ast.Constant("string", r'"debug=true %d\n"'),
        arg,
    ]))
    _items(fn.body).insert(0, call)
    return _reprint(ast)


def _rename(code, salt):
    try:
        ast = _parse(code)
    except Exception:
        return None
    fn = _fn(ast)
    if fn is None:
        return None
    names = []
    seen = set()
    for name in _declared(fn):
        if name not in seen and name != fn.decl.name and name not in _BUILTINS:
            seen.add(name)
            names.append(name)
    used = set(names) | set(_BUILTINS) | {fn.decl.name}
    mapping = {}
    eligible = [name for name in names if any(name in pool for pool in _POOLS)]
    if not eligible:
        return None
    # salt picks which eligible names move, and which free pool name they take.
    chosen = eligible if salt % 3 == 2 else eligible[: 1 + (salt % max(len(eligible), 1))]
    for offset, name in enumerate(chosen):
        pool = next(pool for pool in _POOLS if name in pool)
        options = [item for item in pool if item not in used]
        if not options:
            continue
        pick = options[(salt + offset) % len(options)]
        mapping[name] = pick
        used.add(pick)
    if not mapping:
        return None

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
    return _reprint(ast)


def _incr_name(node):
    if isinstance(node, c_ast.UnaryOp) and node.op in ("p++", "++", "p--", "--") and isinstance(node.expr, c_ast.ID):
        if node.op in ("p++", "++"):
            return node.expr.name, "+"
        return node.expr.name, "-"
    if isinstance(node, c_ast.Assignment) and isinstance(node.lvalue, c_ast.ID):
        if node.op in ("+=", "-=") and _is_one(node.rvalue):
            return node.lvalue.name, "+" if node.op == "+=" else "-"
        if node.op == "=" and isinstance(node.rvalue, c_ast.BinaryOp) and node.rvalue.op in ("+", "-"):
            if (isinstance(node.rvalue.left, c_ast.ID) and node.rvalue.left.name == node.lvalue.name
                    and _is_one(node.rvalue.right)):
                return node.lvalue.name, "+" if node.rvalue.op == "+" else "-"
    return None


def _make_incr(name, form, sign):
    if sign == "-" and form in ("p++", "++", "+=", "add"):
        form = {"p++": "p--", "++": "--", "+=": "-=", "add": "sub"}[form]
    if form == "p++":
        return c_ast.UnaryOp("p++", _id(name))
    if form == "++":
        return c_ast.UnaryOp("++", _id(name))
    if form == "p--":
        return c_ast.UnaryOp("p--", _id(name))
    if form == "--":
        return c_ast.UnaryOp("--", _id(name))
    if form == "+=":
        return c_ast.Assignment("+=", _id(name), _const(1))
    if form == "-=":
        return c_ast.Assignment("-=", _id(name), _const(1))
    op = "+" if sign == "+" else "-"
    return c_ast.Assignment("=", _id(name), c_ast.BinaryOp(op, _id(name), _const(1)))


def _incr(code, salt):
    try:
        ast = _parse(code)
    except Exception:
        return None
    form = _INCR_FORMS[salt % len(_INCR_FORMS)]
    changed = False

    def convert(node):
        nonlocal changed
        found = _incr_name(node)
        if found is None:
            return node
        name, sign = found
        made = _make_incr(name, form, sign)
        if _GEN.visit(made) == _GEN.visit(node):
            return node
        changed = True
        return made

    def visit(node):
        if isinstance(node, c_ast.Compound) and node.block_items:
            node.block_items = [convert(item) for item in node.block_items]
            for item in node.block_items:
                visit(item)
            return
        if isinstance(node, c_ast.For) and node.next is not None:
            if isinstance(node.next, c_ast.ExprList):
                node.next.exprs = [convert(item) for item in node.next.exprs]
            else:
                node.next = convert(node.next)
        for _, child in node.children():
            visit(child)

    visit(ast)
    if not changed:
        return None
    return _reprint(ast)


def _decl_hoist(code):
    try:
        ast = _parse(code)
    except Exception:
        return None
    fn = _fn(ast)
    if fn is None or fn.body is None:
        return None
    target = None
    for node in _walk(fn.body):
        if isinstance(node, c_ast.For) and isinstance(node.init, c_ast.DeclList):
            decls = [d for d in (node.init.decls or []) if isinstance(d, c_ast.Decl) and d.name and d.init is not None]
            if len(decls) == 1:
                target = (node, decls[0])
                break
    if target is None:
        return None
    loop, decl = target
    name = decl.name
    others = [n for n in _declared(fn) if n == name]
    if len(others) != 1:
        return None
    hoisted = copy.deepcopy(decl)
    hoisted.init = None
    loop.init = c_ast.Assignment("=", _id(name), decl.init)
    _items(fn.body).insert(0, hoisted)
    return _reprint(ast)


def style_variants(code, limit, rng):
    """Up to ``limit`` semantic-preserving variants. Each item is ``(aug names, code)``."""
    specs = []
    for text in _COMMENTS:
        specs.append((["comments"], lambda c, text=text: _insert_comment(c, text)))
    for extra in (1, 2, 4):
        specs.append((["spacing"], lambda c, extra=extra: _indent_more(c, extra)))
    specs.append((["brace_style"], _to_kr))
    specs.append((["brace_style"], _to_allman))
    specs.append((["brace_style"], _to_nobrace))
    specs.append((["redundant_parens"], _add_parens))
    for name in ("unused_a", "unused_b", "tmp_pad"):
        specs.append((["unused_var"], lambda c, name=name: _add_unused(c, name)))
    specs.append((["debug_printf"], _add_debug))
    for salt in range(3):
        specs.append((["rename"], lambda c, salt=salt: _rename(c, salt)))
        specs.append((["incr_form"], lambda c, salt=salt: _incr(c, salt)))
    specs.append((["decl_hoist"], _decl_hoist))
    rng.shuffle(specs)
    out = []
    seen = {code}
    for names, fn in specs:
        if len(out) >= limit:
            break
        try:
            nxt = fn(code)
        except Exception:
            nxt = None
        if not nxt or nxt in seen:
            continue
        seen.add(nxt)
        out.append((list(names), nxt))
    if len(out) < limit:
        for names, nxt in list(out):
            if len(out) >= limit:
                break
            try:
                combo = _indent_more(nxt, 2)
            except Exception:
                combo = None
            if combo and combo not in seen:
                seen.add(combo)
                merged = list(names)
                if "spacing" not in merged:
                    merged.append("spacing")
                out.append((merged, combo))
    return out[:limit]


# ---------------------------------------------------------------- near-misses (CORRECT, source E)


def _edit_nth(code, which, matcher):
    """Apply ``matcher(node) -> replacement or None`` to the ``which``-th hit. ``which`` < 0 edits every hit."""
    try:
        ast = _parse(code)
    except Exception:
        return None
    seen = 0
    changed = False

    def visit(node):
        nonlocal seen, changed
        replacement = matcher(node)
        if replacement is not None:
            if which < 0 or seen == which:
                changed = True
                if which >= 0:
                    return replacement
            seen += 1
            if which >= 0 and seen > which:
                return node
        for name, child in node.children():
            new_child = visit(child)
            if new_child is not child:
                _set_child(node, name, new_child)
        return node

    try:
        visit(ast)
    except Exception:
        return None
    if not changed:
        return None
    try:
        return _reprint(ast)
    except Exception:
        return None


def _set_child(node, name, child):
    if "[" in name:
        attr, index = name[:-1].split("[")
        getattr(node, attr)[int(index)] = child
    else:
        setattr(node, name, child)


def _nth_loop(code, which, edit):
    try:
        ast = _parse(code)
    except Exception:
        return None
    loops = [node for node in _walk(ast) if isinstance(node, (c_ast.For, c_ast.While, c_ast.DoWhile))]
    if which >= len(loops):
        return None
    if not edit(loops[which]):
        return None
    try:
        return _reprint(ast)
    except Exception:
        return None


def _plain_lt(cond):
    return (isinstance(cond, c_ast.BinaryOp) and cond.op == "<"
            and isinstance(cond.left, c_ast.ID) and isinstance(cond.right, c_ast.ID))


def _nm_le_nminus1(code, which):
    def edit(loop):
        if not _plain_lt(loop.cond):
            return False
        var, bound = loop.cond.left.name, loop.cond.right.name
        loop.cond = c_ast.BinaryOp(
            "<=", _id(var), c_ast.BinaryOp("-", _id(bound), _const(1)),
        )
        return True
    return _nth_loop(code, which, edit)


def _for_init_name(loop):
    init = loop.init
    if isinstance(init, c_ast.DeclList):
        for decl in init.decls or []:
            if isinstance(decl, c_ast.Decl) and decl.name and _int_value(decl.init) is not None:
                return decl.name, decl, "decl"
    if isinstance(init, c_ast.Assignment) and init.op == "=" and isinstance(init.lvalue, c_ast.ID):
        if _int_value(init.rvalue) is not None:
            return init.lvalue.name, init, "assign"
    return None, None, None


def _mentions(node, name):
    if isinstance(node, c_ast.ID) and node.name == name:
        return True
    return any(_mentions(child, name) for _, child in (node.children() if node is not None else []))


def _nm_one_based(code, which):
    def edit(loop):
        if not isinstance(loop, c_ast.For) or not _plain_lt(loop.cond):
            return False
        var, holder, kind = _for_init_name(loop)
        if var is None or loop.cond.left.name != var or _int_value(holder.init if kind == "decl" else holder.rvalue) != 0:
            return False
        if _incr_name(loop.next) != (var, "+"):
            return False
        exact = other = 0
        refs = []
        for node in _walk(loop.stmt):
            if isinstance(node, c_ast.ArrayRef):
                if isinstance(node.subscript, c_ast.ID) and node.subscript.name == var:
                    exact += 1
                    refs.append(node)
                elif _mentions(node.subscript, var):
                    other += 1
        if exact == 0 or other:
            return False
        if kind == "decl":
            holder.init = _const(1)
        else:
            holder.rvalue = _const(1)
        loop.cond.op = "<="
        for ref in refs:
            ref.subscript = c_ast.BinaryOp("-", _id(var), _const(1))
        return True
    return _nth_loop(code, which, edit)


def _nm_countdown(code, which):
    def edit(loop):
        if not isinstance(loop, c_ast.For) or not _plain_lt(loop.cond):
            return False
        var, holder, kind = _for_init_name(loop)
        if var is None or loop.cond.left.name != var:
            return False
        if _int_value(holder.init if kind == "decl" else holder.rvalue) != 0:
            return False
        if _incr_name(loop.next) != (var, "+"):
            return False
        bound = loop.cond.right.name
        start = c_ast.BinaryOp("-", _id(bound), _const(1))
        if kind == "decl":
            holder.init = start
        else:
            holder.rvalue = start
        loop.cond = c_ast.BinaryOp(">=", _id(var), _const(0))
        loop.next = _make_incr(var, "p--", "-")
        return True
    return _nth_loop(code, which, edit)


def _nm_decl_split(code, which):
    try:
        ast = _parse(code)
    except Exception:
        return None
    fn = _fn(ast)
    if fn is None:
        return None
    found = []

    def scan(node):
        if isinstance(node, c_ast.Compound) and node.block_items:
            for index, item in enumerate(node.block_items):
                if isinstance(item, c_ast.Decl) and item.init is not None and _int_decl(item):
                    found.append((node, index, item))
                else:
                    scan(item)
        elif node is not None:
            for _, child in node.children():
                if not isinstance(child, c_ast.DeclList):
                    scan(child)

    scan(fn.body)
    if which >= len(found):
        return None
    compound, index, decl = found[which]
    init = decl.init
    decl.init = None
    compound.block_items.insert(index + 1, c_ast.Assignment("=", _id(decl.name), init))
    return _reprint(ast)


_FLIP = {"<": ">", "<=": ">=", ">": "<", ">=": "<=", "==": "==", "!=": "!="}


def _nm_yoda(code, which):
    try:
        ast = _parse(code)
    except Exception:
        return None
    ifs = [node for node in _walk(ast) if isinstance(node, c_ast.If)]
    hits = []
    for node in ifs:
        cond = node.cond
        if not isinstance(cond, c_ast.BinaryOp) or cond.op not in _FLIP:
            continue
        left_c = isinstance(cond.left, c_ast.Constant)
        right_c = isinstance(cond.right, c_ast.Constant)
        if left_c == right_c:
            continue
        hits.append(node)
    if which >= len(hits):
        return None
    cond = hits[which].cond
    if isinstance(cond.right, c_ast.Constant):
        cond.left, cond.right = cond.right, cond.left
        cond.op = _FLIP[cond.op]
    else:
        cond.left, cond.right = cond.right, cond.left
        cond.op = _FLIP[cond.op]
    return _reprint(ast)


def _nm_unbraced(code, which):
    try:
        ast = _parse(code)
    except Exception:
        return None
    hosts = []
    for node in _walk(ast):
        if isinstance(node, c_ast.If) and isinstance(node.iftrue, c_ast.Compound):
            hosts.append((node, "iftrue"))
        if isinstance(node, (c_ast.For, c_ast.While)) and isinstance(node.stmt, c_ast.Compound):
            hosts.append((node, "stmt"))
    good = []
    for node, attr in hosts:
        body = getattr(node, attr)
        if body.block_items and len(body.block_items) == 1 and _simple_stmt(body.block_items[0]):
            good.append((node, attr, body.block_items[0]))
    if which >= len(good):
        return None
    node, attr, stmt = good[which]
    setattr(node, attr, stmt)
    return _reprint(ast)


def _nm_debug(code, which):
    return _add_debug(code) if which == 0 else None


def _nm_early_break(code, which):
    def edit(loop):
        if not _plain_lt(loop.cond) or not isinstance(loop.stmt, c_ast.Compound):
            return False
        var, bound = loop.cond.left.name, loop.cond.right.name
        test = c_ast.BinaryOp(
            ">=",
            c_ast.BinaryOp("+", _id(var), _const(1)),
            _id(bound),
        )
        _items(loop.stmt).append(c_ast.If(test, c_ast.Break(), None))
        return True
    return _nth_loop(code, which, edit)


def _nm_redundant(code, which):
    try:
        ast = _parse(code)
    except Exception:
        return None
    fn = _fn(ast)
    if fn is None or fn.body is None or which != 0:
        return None
    names = _scalar_names(fn)
    if not names:
        return None
    name = names[0]
    stmt = c_ast.Assignment(
        "=", _id(name), c_ast.BinaryOp("+", _id(name), _const(0)),
    )
    _items(fn.body).insert(0, stmt)
    return _reprint(ast)


def _is_char0(node):
    if not isinstance(node, c_ast.Constant) or node.type != "char":
        return False
    value = node.value or ""
    return "0" in value and "'" in value


def _nm_term(code, which):
    def edit(loop):
        cond = loop.cond
        if isinstance(cond, c_ast.BinaryOp) and cond.op == "!=" and isinstance(cond.left, c_ast.ArrayRef) and _is_char0(cond.right):
            loop.cond = cond.left
            return True
        if isinstance(cond, c_ast.ArrayRef):
            loop.cond = c_ast.BinaryOp("!=", cond, c_ast.Constant("char", r"'\0'"))
            return True
        return False
    return _nth_loop(code, which, edit)


def _same_expr(a, b):
    try:
        return _GEN.visit(a) == _GEN.visit(b)
    except Exception:
        return False


def _nm_swap_order(code, which):
    try:
        ast = _parse(code)
    except Exception:
        return None
    hits = []
    for node in _walk(ast):
        if not isinstance(node, c_ast.Compound) or not node.block_items:
            continue
        items = node.block_items
        for index in range(len(items) - 2):
            a, b, c = items[index: index + 3]
            if not all(isinstance(x, c_ast.Assignment) and x.op == "=" for x in (a, b, c)):
                continue
            if not isinstance(a.lvalue, c_ast.ID) or not isinstance(c.rvalue, c_ast.ID):
                continue
            if a.lvalue.name != c.rvalue.name:
                continue
            if not _same_expr(a.rvalue, b.lvalue) or not _same_expr(b.rvalue, c.lvalue):
                continue
            hits.append((items, index, a, b, c))
    if which >= len(hits):
        return None
    items, index, a, b, c = hits[which]
    temp = a.lvalue.name
    src_a = copy.deepcopy(a.rvalue)
    src_c = copy.deepcopy(c.lvalue)
    items[index] = c_ast.Assignment("=", _id(temp), copy.deepcopy(b.rvalue))
    items[index + 1] = c_ast.Assignment("=", src_c, src_a)
    items[index + 2] = c_ast.Assignment("=", copy.deepcopy(b.lvalue), _id(temp))
    return _reprint(ast)


def _nm_mid(code, which):
    try:
        ast = _parse(code)
    except Exception:
        return None
    hits = []
    for node in _walk(ast):
        if not isinstance(node, c_ast.BinaryOp) or node.op != "/":
            continue
        if not _is_one(node.right) and not (isinstance(node.right, c_ast.Constant) and node.right.value == "2"):
            continue
        left = node.left
        if isinstance(left, c_ast.BinaryOp) and left.op == "+" and isinstance(left.left, c_ast.ID) and isinstance(left.right, c_ast.ID):
            hits.append(node)
    if which >= len(hits):
        return None
    node = hits[which]
    low, high = node.left.left.name, node.left.right.name
    node.op = "+"
    node.left = _id(low)
    node.right = c_ast.BinaryOp(
        "/",
        c_ast.BinaryOp("-", _id(high), _id(low)),
        _const(2),
    )
    return _reprint(ast)


def _nm_base(code, which):
    try:
        ast = _parse(code)
    except Exception:
        return None
    hits = []
    for node in _walk(ast):
        if not isinstance(node, c_ast.If) or not isinstance(node.cond, c_ast.BinaryOp):
            continue
        cond = node.cond
        if not isinstance(cond.left, c_ast.ID):
            continue
        value = _int_value(cond.right)
        if value is None:
            continue
        if not ((cond.op == "==" and value == 0) or (cond.op == "<=" and value == 0) or (cond.op == "<" and value == 1)):
            continue
        ret = node.iftrue
        if isinstance(ret, c_ast.Compound) and ret.block_items:
            ret = ret.block_items[-1]
        if isinstance(ret, c_ast.Return) and _int_value(ret.expr) == 1:
            hits.append(node)
    if which >= len(hits):
        return None
    cond = hits[which].cond
    cond.op = "<="
    cond.right = _const(1)
    return _reprint(ast)


def _nm_float(code, which):
    try:
        ast = _parse(code)
    except Exception:
        return None
    hits = []
    for node in _walk(ast):
        if isinstance(node, c_ast.BinaryOp) and node.op == "/" and isinstance(node.left, c_ast.Cast):
            if not isinstance(node.right, c_ast.Cast):
                hits.append(node)
    if which >= len(hits):
        return None
    node = hits[which]
    node.right = c_ast.Cast(copy.deepcopy(node.left.to_type), node.right)
    return _reprint(ast)


def _is_minus_one(node):
    if isinstance(node, c_ast.BinaryOp) and node.op == "-" and isinstance(node.left, c_ast.ID) and _int_value(node.right) == 1:
        return node.left.name
    return None


def _nm_bubble_bound(code, which):
    def edit(loop):
        cond = loop.cond
        if not isinstance(cond, c_ast.BinaryOp) or cond.op != "<" or not isinstance(cond.left, c_ast.ID):
            return False
        right = cond.right
        if not isinstance(right, c_ast.BinaryOp) or right.op != "-":
            return False
        bound = _is_minus_one(right.left)
        if bound is None or not isinstance(right.right, c_ast.ID):
            return False
        cond.right = c_ast.BinaryOp("-", _id(bound), _const(1))
        return True
    return _nth_loop(code, which, edit)


def _nm_unused_array(code, which):
    if which != 0:
        return None
    try:
        ast = _parse(code)
    except Exception:
        return None
    fn = _fn(ast)
    if fn is None or fn.body is None:
        return None
    if "pad_cells" in _declared(fn):
        return None
    decl = c_ast.Decl(
        "pad_cells", [], [], [], [],
        c_ast.ArrayDecl(
            c_ast.TypeDecl("pad_cells", [], None, c_ast.IdentifierType(["int"])),
            _const(1), [],
        ),
        None, None,
    )
    assign = c_ast.Assignment(
        "=",
        c_ast.ArrayRef(_id("pad_cells"), _const(0)),
        _const(0),
    )
    items = _items(fn.body)
    items.insert(0, decl)
    items.insert(1, assign)
    return _reprint(ast)


_NEAR = (
    ("nm_le_nminus1", _nm_le_nminus1),
    ("nm_one_based", _nm_one_based),
    ("nm_countdown", _nm_countdown),
    ("nm_decl_split", _nm_decl_split),
    ("nm_yoda", _nm_yoda),
    ("nm_unbraced", _nm_unbraced),
    ("nm_debug_return", _nm_debug),
    ("nm_early_break", _nm_early_break),
    ("nm_redundant", _nm_redundant),
    ("nm_term", _nm_term),
    ("nm_swap_order", _nm_swap_order),
    ("nm_mid", _nm_mid),
    ("nm_base_le1", _nm_base),
    ("nm_float_cast", _nm_float),
    ("nm_bubble_bound", _nm_bubble_bound),
    ("nm_unused_array", _nm_unused_array),
)


def near_misses(code):
    """CORRECT-shaped rewrites of one correct variant. Tests decide which survive."""
    found = []
    seen = set()
    for name, fn in _NEAR:
        for which in range(6):
            try:
                text = fn(code, which)
            except Exception:
                text = None
            if not text or text in seen or text.strip() == code.strip():
                if text is None and which == 0:
                    break
                if text is None:
                    break
                continue
            seen.add(text)
            found.append((name, text))
    return found
