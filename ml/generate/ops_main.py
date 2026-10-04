"""Main-game mutation operators (ml_plan/03 §3.5.1, package C1).

Each operator finds sites on a parsed syntax tree and edits that tree. ``apply``
reprints with ``pycparser.c_generator.CGenerator``. A ``BinaryOp`` coordinate is the
left operand, so nothing here edits source text by column.

``sites`` returns indexes into a fresh parse. ``apply`` parses again and edits that
one index, so one site cannot leak into the next.

DSA-surface operators (``amb_pair_bound``, ``m08_mirror``, ``m01_half_bound``,
``m02_pointer_stuck``, ``m10_print_bool``) and D01–D08 belong to package C2.
"""
from __future__ import annotations

import copy
import re

from pycparser import c_ast, c_generator, c_parser

from ml.c_interp.preprocess import preprocess

_REL = {"<", "<=", ">", ">=", "==", "!="}
_FLIP = {"<": ">", "<=": ">=", ">": "<", ">=": "<=", "==": "==", "!=": "!="}
_INC = {"p++": 1, "++": 1, "p--": -1, "--": -1}
_LOOPS = (c_ast.For, c_ast.While, c_ast.DoWhile)


class Op:
    """One catalogue operator. ``labels`` is the class or the AMB pair."""

    def __init__(self, op_id, labels, test_only=False):
        self.op_id = op_id
        self.labels = tuple(labels)
        self.test_only = test_only

    def __repr__(self):
        return f"Op({self.op_id})"


class _Tree:
    """Parent and attribute of every node, from one parse."""

    def __init__(self, ast):
        self.ast = ast
        self.parent = {}
        self.attr = {}
        self._index(ast)

    def _index(self, node):
        for name, child in node.children():
            self.parent[id(child)] = node
            self.attr[id(child)] = name
            self._index(child)

    def inside(self, node, ancestor):
        cur = node
        while cur is not None and cur is not ancestor:
            cur = self.parent.get(id(cur))
        return cur is ancestor


class _Loop:
    """One for / while / do-while, with the condition read as ``var op bound``."""

    def __init__(self, node, fn, tree):
        self.node, self.fn, self.tree = node, fn, tree
        self.kind = "for" if isinstance(node, c_ast.For) else "while"
        self.body = node.stmt
        self.next = node.next if isinstance(node, c_ast.For) else None
        self.cond_ids = _ids(node.cond)
        var, init = _for_init(node, self.cond_ids)
        self.mods = _mods(self.next) + _mods(self.body)
        if var is None:
            var = _guess_var(node.cond, self.cond_ids, self.mods)
        self.var = var
        if init is None and var is not None and not (isinstance(node, c_ast.For) and node.init is not None):
            init = _last_value(fn, node, var)
        self.init = init
        self.zero = init if _int_const(init) == 0 else None
        self.init_form = _init_form(init, var)
        self.rels = []
        if var is not None and node.cond is not None:
            for rel in _relations(node.cond):
                norm, on_left, bound = _norm_rel(rel, var)
                if norm is None:
                    continue
                self.rels.append((rel, norm, on_left, bound, _bound_form(bound, var)))
        self.indexes = _indexes(self.body, var)
        self.cond_mods = [m for m in self.mods if m[0] == var] if var else []

    def has(self, norm, form):
        return any(r[1] == norm and r[4] == form for r in self.rels)


def parse_ast(code):
    """pycparser tree of ``code`` after the interpreter's preprocess."""
    return c_parser.CParser().parse(preprocess(code))


def _functions(ast):
    for ext in getattr(ast, "ext", None) or []:
        if isinstance(ext, c_ast.FuncDef):
            yield ext


def _walk(node):
    if node is None:
        return
    yield node
    for _, child in node.children():
        yield from _walk(child)


def _loops(fn, tree):
    found = []

    def visit(node):
        for _, child in node.children():
            if isinstance(child, _LOOPS):
                found.append(_Loop(child, fn, tree))
            visit(child)

    if fn.body is not None:
        visit(fn.body)
    return found


def _is_id(node, name=None):
    return isinstance(node, c_ast.ID) and (name is None or node.name == name)


def _int_const(node):
    sign = 1
    while isinstance(node, c_ast.UnaryOp) and node.op in ("+", "-"):
        if node.op == "-":
            sign = -sign
        node = node.expr
    if isinstance(node, c_ast.Constant) and node.type in ("int", "long int", "unsigned int"):
        try:
            return sign * int(node.value.rstrip("uUlL"), 0)
        except ValueError:
            return None
    return None


def _float_const(node):
    if isinstance(node, c_ast.Constant) and node.type in ("float", "double"):
        return node
    return None


def _type_names(node):
    seen = 0
    while node is not None and not isinstance(node, c_ast.IdentifierType) and seen < 8:
        node = getattr(node, "type", None)
        seen += 1
    if isinstance(node, c_ast.IdentifierType):
        return list(node.names or [])
    return []


def _is_float_names(names):
    return "float" in names or "double" in names


def _is_void_fn(fn):
    return "void" in _type_names(fn.decl.type)


def _returns_float(fn):
    return _is_float_names(_type_names(fn.decl.type))


def _is_array_decl(decl):
    return isinstance(getattr(decl, "type", None), c_ast.ArrayDecl)


def _is_float_decl(decl):
    return _is_float_names(_type_names(decl.type))


def _is_float_cast(node):
    return isinstance(node, c_ast.Cast) and _is_float_names(_type_names(node.to_type))


def _params(fn):
    scalars, arrays, floats = [], [], set()
    args = fn.decl.type.args
    for param in (args.params if args else []):
        if not isinstance(param, c_ast.Decl) or not param.name:
            continue
        if _is_array_decl(param):
            arrays.append(param.name)
        else:
            scalars.append(param.name)
            if _is_float_decl(param):
                floats.add(param.name)
    return scalars, arrays, floats


def _float_names(fn):
    _, _, floats = _params(fn)
    names = set(floats)
    for node in _walk(fn.body):
        if isinstance(node, c_ast.Decl) and node.name and _is_float_decl(node) and not _is_array_decl(node):
            names.add(node.name)
    return names


def _ids(node):
    return [n.name for n in _walk(node) if isinstance(n, c_ast.ID)]


def _relations(cond):
    if isinstance(cond, c_ast.BinaryOp):
        if cond.op in _REL:
            return [cond]
        if cond.op in ("&&", "||"):
            return _relations(cond.left) + _relations(cond.right)
    return []


def _norm_rel(rel, var):
    if _is_id(rel.left, var):
        return rel.op, True, rel.right
    if _is_id(rel.right, var):
        return _FLIP[rel.op], False, rel.left
    return None, False, None


def _bound_form(bound, var):
    if isinstance(bound, c_ast.ID) and bound.name != var:
        return "n"
    if (isinstance(bound, c_ast.BinaryOp) and bound.op == "-"
            and isinstance(bound.left, c_ast.ID) and bound.left.name != var
            and _int_const(bound.right) == 1):
        return "n_minus_1"
    if _int_const(bound) is not None:
        return "const"
    return "other"


def _init_form(expr, var):
    if _int_const(expr) == 0:
        return "0"
    if _int_const(expr) == 1:
        return "1"
    if isinstance(expr, c_ast.ID) and (var is None or expr.name != var):
        return "n"
    return "other"


def _for_init(node, cond_ids):
    if not isinstance(node, c_ast.For) or node.init is None:
        return None, None
    init = node.init
    if isinstance(init, c_ast.DeclList):
        for decl in init.decls or []:
            if decl.name:
                return decl.name, decl.init
        return None, None
    items = init.exprs if isinstance(init, c_ast.ExprList) else [init]
    assigns = [a for a in items if isinstance(a, c_ast.Assignment) and a.op == "=" and _is_id(a.lvalue)]
    for assign in assigns:
        if assign.lvalue.name in cond_ids:
            return assign.lvalue.name, assign.rvalue
    if assigns:
        return assigns[0].lvalue.name, assigns[0].rvalue
    return None, None


def _guess_var(cond, cond_ids, mods):
    updated = {name for name, _, _ in mods}
    for rel in _relations(cond):
        if isinstance(rel.left, c_ast.ID) and (rel.left.name in updated or rel.left.name in cond_ids):
            return rel.left.name
        if isinstance(rel.right, c_ast.ID) and rel.right.name in updated:
            return rel.right.name
    return cond_ids[0] if cond_ids else None


def _last_value(fn, stop, var):
    found = None

    def visit(node):
        nonlocal found
        if node is stop:
            return True
        if isinstance(node, c_ast.Decl) and node.name == var and not _is_array_decl(node):
            found = node.init
        elif isinstance(node, c_ast.Assignment) and node.op == "=" and _is_id(node.lvalue, var):
            found = node.rvalue
        for _, child in node.children():
            if visit(child):
                return True
        return False

    if fn.body is not None:
        visit(fn.body)
    return found


def _indexes(body, var):
    if not var:
        return False
    for node in _walk(body):
        if isinstance(node, c_ast.ArrayRef) and _is_id(node.subscript, var):
            return True
    return False


def _mods(node):
    """Scalar updates under ``node``: (name, direction, ast node)."""
    found = []
    for cur in _walk(node):
        if isinstance(cur, c_ast.UnaryOp) and cur.op in _INC and _is_id(cur.expr):
            found.append((cur.expr.name, _INC[cur.op], cur))
        elif isinstance(cur, c_ast.Assignment) and _is_id(cur.lvalue):
            name, rv = cur.lvalue.name, cur.rvalue
            if cur.op in ("+=", "-="):
                found.append((name, 1 if cur.op == "+=" else -1, cur))
            elif cur.op in ("*=", "/=", "%="):
                found.append((name, 1 if cur.op == "*=" else -1, cur))
            elif cur.op == "=" and isinstance(rv, c_ast.BinaryOp) and rv.op in ("+", "-", "*", "/"):
                left_self = _is_id(rv.left, name)
                right_self = _is_id(rv.right, name) and rv.op in ("+", "*")
                if left_self or right_self:
                    found.append((name, -1 if rv.op == "-" else 1, cur))
    return found


def _mod_kind(mods, name):
    """'acc' (running total / product) or 'counter' (++ / += 1), else None."""
    kinds = set()
    for mod_name, _, node in mods:
        if mod_name != name:
            continue
        if isinstance(node, c_ast.UnaryOp) and "++" in node.op:
            kinds.add("counter")
        elif isinstance(node, c_ast.Assignment) and node.op == "+=":
            kinds.add("counter" if _int_const(node.rvalue) == 1 else "acc")
        elif isinstance(node, c_ast.Assignment) and node.op == "=" and isinstance(node.rvalue, c_ast.BinaryOp):
            rv = node.rvalue
            other = rv.right if _is_id(rv.left, name) else rv.left
            if rv.op == "+" and _int_const(other) == 1:
                kinds.add("counter")
            else:
                kinds.add("acc")
        else:
            kinds.add("acc")
    if "acc" in kinds:
        return "acc"
    if "counter" in kinds:
        return "counter"
    return None


def _direct(stmt):
    if isinstance(stmt, c_ast.Compound):
        return list(stmt.block_items or [])
    if stmt is None:
        return []
    return [stmt]


def _progress_stmt(stmt, name):
    if isinstance(stmt, c_ast.UnaryOp) and stmt.op in _INC and _is_id(stmt.expr, name):
        return True
    if isinstance(stmt, c_ast.Assignment) and _is_id(stmt.lvalue, name):
        return True
    return False


def _set_rel(rel, on_left, norm_op):
    rel.op = norm_op if on_left else _FLIP[norm_op]


def _replace(tree, node, new):
    parent = tree.parent[id(node)]
    name = tree.attr[id(node)]
    match = re.fullmatch(r"(\w+)\[(\d+)\]", name)
    if match:
        getattr(parent, match.group(1))[int(match.group(2))] = new
    else:
        setattr(parent, name, new)


def _expand_stmt(tree, node, nodes):
    """Replace ``node`` with one or more statements."""
    parent = tree.parent[id(node)]
    name = tree.attr[id(node)]
    match = re.fullmatch(r"block_items\[(\d+)\]", name)
    if match:
        block = parent.block_items
        index = int(match.group(1))
        block[index:index + 1] = nodes
        return
    if len(nodes) == 1:
        setattr(parent, name, nodes[0])
    else:
        setattr(parent, name, c_ast.Compound(nodes))


def _delete_stmt(tree, stmt):
    """Remove a statement. A sole body becomes ``{}``, never a bare ``;`` (that is M07)."""
    parent = tree.parent[id(stmt)]
    name = tree.attr[id(stmt)]
    match = re.fullmatch(r"(\w+)\[(\d+)\]", name)
    if match:
        del getattr(parent, match.group(1))[int(match.group(2))]
        return
    setattr(parent, name, c_ast.Compound([]))


def _as_compound(loop):
    stmt = loop.node.stmt
    if isinstance(stmt, c_ast.Compound):
        if stmt.block_items is None:
            stmt.block_items = []
        return stmt
    compound = c_ast.Compound([] if stmt is None else [stmt])
    loop.node.stmt = compound
    return compound


def _in_for_init(tree, node):
    cur = node
    while id(cur) in tree.parent:
        parent = tree.parent[id(cur)]
        if isinstance(parent, c_ast.For) and tree.attr[id(cur)] == "init":
            return True
        cur = parent
    return False


def _in_loop_body(tree, node):
    """True when ``node`` sits in a loop body, not in its init / condition / update."""
    cur = node
    while id(cur) in tree.parent:
        parent = tree.parent[id(cur)]
        attr = tree.attr[id(cur)]
        if isinstance(parent, _LOOPS):
            return attr == "stmt"
        cur = parent
    return False


def _under_attr(tree, node, kinds, attr_name):
    """Nearest ancestor in ``kinds`` whose link to this node is ``attr_name``."""
    cur = node
    while id(cur) in tree.parent:
        parent = tree.parent[id(cur)]
        if tree.attr[id(cur)] == attr_name and isinstance(parent, kinds):
            return parent
        cur = parent
    return None


def _controller(tree, node):
    return _under_attr(tree, node, (c_ast.If, c_ast.For, c_ast.While, c_ast.DoWhile), "cond")


def _in_if_body(tree, node, stop):
    cur = node
    while id(cur) in tree.parent and cur is not stop:
        parent = tree.parent[id(cur)]
        if isinstance(parent, c_ast.If) and tree.attr[id(cur)] in ("iftrue", "iffalse"):
            return True
        cur = parent
    return False


def _const_int(value):
    return c_ast.Constant("int", str(value))


def _id(name):
    return c_ast.ID(name)


def _printf_d(expr):
    return c_ast.FuncCall(_id("printf"), c_ast.ExprList([
        c_ast.Constant("string", '"%d"'),
        expr,
    ]))


def _is_printf(node):
    return isinstance(node, c_ast.FuncCall) and _is_id(node.name, "printf")


def _is_return0(node):
    return isinstance(node, c_ast.Return) and _int_const(node.expr) == 0


def _clear_coords(node):
    for cur in _walk(node):
        if hasattr(cur, "coord"):
            cur.coord = None


def _decl_copy(decl):
    copied = copy.deepcopy(decl)
    _clear_coords(copied)
    return copied


def _scalar_decls(fn):
    for node in _walk(fn.body):
        if isinstance(node, c_ast.Decl) and node.name and not _is_array_decl(node):
            if not isinstance(node.type, c_ast.TypeDecl):
                # keep ordinary scalars; skip function prototypes
                if isinstance(node.type, c_ast.FuncDecl):
                    continue
            yield node


def _cond_vars(loops):
    return {lp.var for lp in loops if lp.var}


def _other_scalar(fn, avoid, prefer):
    scalars, _, _ = _params(fn)
    names = [name for name in scalars if name not in avoid]
    for node in _walk(fn.body):
        if isinstance(node, c_ast.Decl) and node.name and not _is_array_decl(node):
            if node.name not in avoid and node.name not in names:
                names.append(node.name)
    for name in names:
        if name in prefer:
            return name
    return names[0] if names else None


def _skip_literal(tree, node):
    """Loop bounds, indexes and 0/1 are not OTHER-constant sites."""
    value = _int_const(node)
    if value is None or value in (0, 1):
        return True
    cur = node
    while id(cur) in tree.parent:
        parent = tree.parent[id(cur)]
        attr = tree.attr[id(cur)]
        if isinstance(parent, _LOOPS) and attr in ("cond", "init", "next"):
            return True
        if isinstance(parent, c_ast.ArrayRef) and attr == "subscript":
            return True
        cur = parent
    return False


def _literal_edit(value):
    """42 → 24 (digit reverse); a single digit 7 → 6."""
    number = _int_const(value) if not isinstance(value, int) else value
    text = str(abs(number))
    if len(text) == 1:
        changed = number - 1 if number > 0 else number + 1
        return str(changed)
    reversed_digits = text[::-1].lstrip("0") or "0"
    if reversed_digits == text:
        return str(number + 1)
    return ("-" if number < 0 else "") + reversed_digits


def _expr_kind(node, float_names):
    if isinstance(node, c_ast.Cast) and _is_float_cast(node):
        return "float"
    if isinstance(node, c_ast.Constant):
        return "float" if node.type in ("float", "double") else "int"
    if isinstance(node, c_ast.ID):
        return "float" if node.name in float_names else "int"
    if isinstance(node, c_ast.BinaryOp):
        if _expr_kind(node.left, float_names) == "float" or _expr_kind(node.right, float_names) == "float":
            return "float"
        return "int"
    if isinstance(node, c_ast.UnaryOp) and node.op in ("+", "-"):
        return _expr_kind(node.expr, float_names)
    return "int"


def _flows_to_float(tree, node, fn, float_names):
    cur = node
    while id(cur) in tree.parent:
        parent = tree.parent[id(cur)]
        if isinstance(parent, c_ast.Cast) and _is_float_cast(parent):
            return True
        if isinstance(parent, c_ast.Return) and _returns_float(fn):
            return True
        if isinstance(parent, c_ast.Decl) and parent.name in float_names:
            return True
        if isinstance(parent, c_ast.Assignment) and _is_id(parent.lvalue) and parent.lvalue.name in float_names:
            return True
        cur = parent
    return False


def _rw(node, out):
    """Uses in evaluation order: ('r'|'w'|'d', name, decl or None)."""
    if node is None:
        return
    if isinstance(node, c_ast.ID):
        out.append(("r", node.name, None))
    elif isinstance(node, c_ast.Assignment):
        _rw(node.rvalue, out)
        if node.op == "=" and _is_id(node.lvalue):
            out.append(("w", node.lvalue.name, None))
        else:
            _rw(node.lvalue, out)
    elif isinstance(node, c_ast.Decl):
        _rw(node.init, out)
        if node.name and not _is_array_decl(node) and not isinstance(node.type, c_ast.FuncDecl):
            out.append(("d", node.name, node))
    elif isinstance(node, c_ast.FuncCall):
        _rw(node.args, out)
    elif isinstance(node, c_ast.ArrayRef):
        _rw(node.subscript, out)
    elif isinstance(node, c_ast.For):
        for part in (node.init, node.cond, node.stmt, node.next):
            _rw(part, out)
    elif isinstance(node, c_ast.DoWhile):
        _rw(node.stmt, out)
        _rw(node.cond, out)
    elif isinstance(node, (c_ast.Typename, c_ast.TypeDecl, c_ast.IdentifierType, c_ast.DeclList)):
        if isinstance(node, c_ast.DeclList):
            for decl in node.decls or []:
                _rw(decl, out)
    else:
        for _, child in node.children():
            _rw(child, out)


def read_before_assign(fn):
    """True when some scalar is read before it is assigned (M05)."""
    events = []
    _rw(fn.body, events)
    state = {}
    for kind, name, decl in events:
        if kind == "d":
            state[name] = "set" if decl is not None and decl.init is not None else "uninit"
        elif kind == "w":
            state[name] = "set"
        elif kind == "r" and state.get(name) == "uninit":
            return True
    return False


def _printf_shapes(fn):
    """(has a printf immediately followed by ``return 0``, has some other printf statement)."""
    paired = bare = False

    def scan(node):
        nonlocal paired, bare
        if isinstance(node, c_ast.Compound):
            items = node.block_items or []
            for index, item in enumerate(items):
                if not _is_printf(item):
                    continue
                nxt = items[index + 1] if index + 1 < len(items) else None
                if _is_return0(nxt):
                    paired = True
                else:
                    bare = True
            for item in items:
                scan(item)
        elif isinstance(node, c_ast.If):
            for branch in (node.iftrue, node.iffalse):
                if _is_printf(branch):
                    bare = True
                elif branch is not None:
                    scan(branch)
        elif isinstance(node, _LOOPS):
            if _is_printf(node.stmt):
                bare = True
            elif node.stmt is not None:
                scan(node.stmt)

    if fn.body is not None:
        scan(fn.body)
    return paired, bare


def _semi_for(tree, loop):
    """``for (...) { body }`` → declarations hoisted, ``for (...);`` and then ``body``.

    Hoisting ``for (int i = …)`` keeps ``i`` in scope for the spliced body. The insertion
    index is computed here: the parent map still has the for-loop's old slot.
    """
    body = loop.node.stmt
    parent = tree.parent.get(id(loop.node))
    name = tree.attr.get(id(loop.node), "")
    match = re.fullmatch(r"block_items\[(\d+)\]", name)
    if parent is None or match is None:
        return
    index = int(match.group(1))
    hoisted = 0
    if isinstance(loop.node.init, c_ast.DeclList):
        decls = list(loop.node.init.decls or [])
        loop.node.init = None
        parent.block_items[index:index] = decls
        hoisted = len(decls)
    loop.node.stmt = c_ast.EmptyStatement()
    if body is not None:
        parent.block_items.insert(index + hoisted + 1, body)


def _splice_after(tree, node, body):
    """Make ``node``'s body empty and place the old body after ``node`` in its block."""
    parent = tree.parent.get(id(node))
    name = tree.attr.get(id(node))
    match = re.fullmatch(r"block_items\[(\d+)\]", name or "")
    if parent is None or not match or body is None:
        return False
    index = int(match.group(1))
    parent.block_items.insert(index + 1, body)
    return True


# ---------------------------------------------------------------- finders

def _each_loop(ast):
    tree = _Tree(ast)
    pairs = []
    for fn in _functions(ast):
        pairs.append((fn, tree, _loops(fn, tree)))
    return pairs


def _find_m01_le(ast):
    hits = []
    for _, _, loops in _each_loop(ast):
        for lp in loops:
            if lp.kind != "for" or lp.indexes:
                continue
            for rel, norm, on_left, _, form in lp.rels:
                if norm == "<" and form == "n":
                    hits.append(lambda rel=rel, on_left=on_left: _set_rel(rel, on_left, "<="))
    return hits


def _find_m01_nminus1(ast):
    hits = []
    for _, _, loops in _each_loop(ast):
        for lp in loops:
            for rel, norm, on_left, bound, form in lp.rels:
                if norm == "<" and form == "n":
                    hits.append(lambda rel=rel, on_left=on_left, bound=bound: _minus_one(rel, on_left, bound))
    return hits


def _minus_one(rel, on_left, bound):
    rewritten = c_ast.BinaryOp("-", bound, _const_int(1))
    if on_left:
        rel.right = rewritten
    else:
        rel.left = rewritten


def _find_m01_start1(ast):
    hits = []
    for _, _, loops in _each_loop(ast):
        for lp in loops:
            if lp.indexes or lp.zero is None:
                continue
            hits.append(lambda zero=lp.zero: _set_const(zero, 1))
    return hits


def _set_const(node, number):
    node.value = str(number)


def _find_m01_countdown(ast):
    hits = []
    for _, _, loops in _each_loop(ast):
        for lp in loops:
            for rel, norm, on_left, bound, form in lp.rels:
                if norm == ">" and form == "const" and _int_const(bound) == 0:
                    hits.append(lambda rel=rel, on_left=on_left: _set_rel(rel, on_left, ">="))
    return hits


def _find_m01_while_le(ast):
    hits = []
    for _, _, loops in _each_loop(ast):
        for lp in loops:
            if lp.kind != "while" or lp.indexes:
                continue
            for rel, norm, on_left, _, form in lp.rels:
                if norm == "<" and form == "n":
                    hits.append(lambda rel=rel, on_left=on_left: _set_rel(rel, on_left, "<="))
    return hits


def _find_m08_index_n(ast):
    hits = []
    for fn, tree, loops in _each_loop(ast):
        loop_vars = _cond_vars(loops)
        for node in _walk(fn.body):
            if not isinstance(node, c_ast.ArrayRef):
                continue
            sub = node.subscript
            if not (isinstance(sub, c_ast.BinaryOp) and sub.op == "-" and isinstance(sub.left, c_ast.ID)
                    and _int_const(sub.right) == 1 and sub.left.name not in loop_vars):
                continue
            hits.append(lambda node=node, name=sub.left.name: _set_subscript(node, _id(name)))
    return hits


def _set_subscript(ref, expr):
    ref.subscript = expr


def _find_m08_onebased(ast):
    hits = []
    for _, _, loops in _each_loop(ast):
        for lp in loops:
            if not lp.indexes or lp.zero is None:
                continue
            chosen = [rel for rel in lp.rels if rel[1] == "<" and rel[4] == "n"]
            if not chosen:
                continue
            hits.append(lambda zero=lp.zero, chosen=list(chosen): _one_based(zero, chosen))
    return hits


def _one_based(zero, chosen):
    zero.value = "1"
    for rel, _, on_left, _, _ in chosen:
        _set_rel(rel, on_left, "<=")


def _find_m08_iplus1(ast):
    hits = []
    for _, tree, loops in _each_loop(ast):
        for lp in loops:
            if not lp.var:
                continue
            for node in _walk(lp.body):
                if isinstance(node, c_ast.ArrayRef) and _is_id(node.subscript, lp.var):
                    if not tree.inside(node, lp.body):
                        continue
                    hits.append(lambda node=node, var=lp.var: _set_subscript(
                        node, c_ast.BinaryOp("+", _id(var), _const_int(1))))
    return hits


def _find_m08_first1(ast):
    hits = []
    for fn, _, _ in _each_loop(ast):
        for node in _walk(fn.body):
            if isinstance(node, c_ast.ArrayRef) and _int_const(node.subscript) == 0:
                const = node.subscript
                while isinstance(const, c_ast.UnaryOp):
                    const = const.expr
                hits.append(lambda const=const: _set_const(const, 1))
    return hits


def _find_amb_le(ast):
    hits = []
    for _, _, loops in _each_loop(ast):
        for lp in loops:
            if not lp.indexes or lp.init_form == "1":
                continue
            for rel, norm, on_left, _, form in lp.rels:
                if norm == "<" and form == "n":
                    hits.append(lambda rel=rel, on_left=on_left: _set_rel(rel, on_left, "<="))
    return hits


def _find_amb_start1(ast):
    hits = []
    for _, _, loops in _each_loop(ast):
        for lp in loops:
            if lp.indexes and lp.zero is not None:
                hits.append(lambda zero=lp.zero: _set_const(zero, 1))
    return hits


def _find_m02_drop(ast):
    hits = []
    for _, tree, loops in _each_loop(ast):
        for lp in loops:
            if not lp.var:
                continue
            if isinstance(lp.node, c_ast.For) and lp.next is not None and any(
                    name == lp.var for name, _, _ in _mods(lp.next)):
                hits.append(lambda node=lp.node: setattr(node, "next", None))
                continue
            stmts = [stmt for stmt in _direct(lp.body) if _progress_stmt(stmt, lp.var)]
            if stmts:
                target = stmts[-1]
                hits.append(lambda tree=tree, target=target: _delete_stmt(tree, target))
    return hits


def _find_m02_reverse(ast):
    hits = []
    for _, _, loops in _each_loop(ast):
        for lp in loops:
            if not lp.var or not any(norm in ("<", "<=") for _, norm, _, _, _ in lp.rels):
                continue
            for name, _, node in lp.mods:
                if name == lp.var and isinstance(node, c_ast.UnaryOp) and node.op in ("p++", "++"):
                    hits.append(lambda node=node: _flip_inc(node))
    return hits


def _flip_inc(node):
    node.op = {"p++": "p--", "++": "--"}[node.op]


def _find_m02_in_branch(ast):
    hits = []
    for _, tree, loops in _each_loop(ast):
        for lp in loops:
            if not isinstance(lp.node, c_ast.While) or not lp.var:
                continue
            if not isinstance(lp.body, c_ast.Compound):
                continue
            items = lp.body.block_items or []
            ifs = [stmt for stmt in items if isinstance(stmt, c_ast.If)]
            if not ifs:
                continue
            for stmt in items:
                if isinstance(stmt, c_ast.If) or not _progress_stmt(stmt, lp.var):
                    continue
                if _in_if_body(tree, stmt, lp.node):
                    continue
                hits.append(lambda stmt=stmt, iff=ifs[0], items=items: _move_into_if(stmt, iff, items))
    return hits


def _move_into_if(stmt, iff, items):
    items.remove(stmt)
    if isinstance(iff.iftrue, c_ast.Compound):
        if iff.iftrue.block_items is None:
            iff.iftrue.block_items = []
        iff.iftrue.block_items.append(stmt)
    elif iff.iftrue is None:
        iff.iftrue = stmt
    else:
        iff.iftrue = c_ast.Compound([iff.iftrue, stmt])


def _find_m02_wrong_var(ast):
    hits = []
    for fn, _, loops in _each_loop(ast):
        for lp in loops:
            if not isinstance(lp.node, c_ast.While) or not lp.var:
                continue
            other = _other_scalar(fn, {lp.var}, set(lp.cond_ids))
            if other is None:
                continue
            for name, _, node in lp.mods:
                if name == lp.var and isinstance(node, c_ast.UnaryOp) and "++" in node.op and _is_id(node.expr):
                    hits.append(lambda expr=node.expr, other=other: setattr(expr, "name", other))
    return hits


def _find_m03_decl(ast):
    hits = []
    for fn, tree, loops in _each_loop(ast):
        guarded = _cond_vars(loops)
        body_mods = _mods(fn.body)
        for decl in _scalar_decls(fn):
            if _int_const(decl.init) is None or decl.name in guarded:
                continue
            if _in_for_init(tree, decl) or _in_loop_body(tree, decl):
                continue
            if _mod_kind(body_mods, decl.name) is None:
                continue
            for lp in loops:
                if decl.name == lp.var:
                    continue
                if any(name == decl.name for name, _, _ in _mods(lp.body)):
                    hits.append(lambda decl=decl, lp=lp: _plant_decl(decl, lp))
                    break
    return hits


def _plant_decl(decl, loop):
    compound = _as_compound(loop)
    compound.block_items.insert(0, _decl_copy(decl))


def _find_m03_assign(ast):
    hits = []
    for fn, _, loops in _each_loop(ast):
        guarded = _cond_vars(loops)
        for lp in loops:
            seen = set()
            for name, _, _ in _mods(lp.body):
                if name in seen or name in guarded or name == lp.var:
                    continue
                seen.add(name)
                hits.append(lambda lp=lp, name=name: _plant_zero(lp, name))
    return hits


def _plant_zero(loop, name):
    compound = _as_compound(loop)
    compound.block_items.insert(0, c_ast.Assignment("=", _id(name), _const_int(0)))


def _cast_divisions(fn):
    found = []
    for node in _walk(fn.body):
        if isinstance(node, c_ast.BinaryOp) and node.op == "/":
            if _is_float_cast(node.left) or _is_float_cast(node.right):
                found.append(node)
    return found


def _unwrap_cast(node):
    if _is_float_cast(node):
        return node.expr, node.to_type
    return node, None


def _find_m04_drop_cast(ast):
    hits = []
    for fn in _functions(ast):
        for div in _cast_divisions(fn):
            hits.append(lambda div=div: _strip_div_casts(div))
    return hits


def _strip_div_casts(div):
    div.left, _ = _unwrap_cast(div.left)
    div.right, _ = _unwrap_cast(div.right)


def _find_m04_cast_late(ast):
    hits = []
    for fn, tree, _ in _each_loop(ast):
        for div in _cast_divisions(fn):
            hits.append(lambda tree=tree, div=div: _wrap_div(tree, div))
    return hits


def _wrap_div(tree, div):
    left, to_type = _unwrap_cast(div.left)
    right, other = _unwrap_cast(div.right)
    fresh = c_ast.BinaryOp("/", left, right)
    _replace(tree, div, c_ast.Cast(to_type or other, fresh))


def _find_m04_int_literal(ast):
    hits = []
    for fn in _functions(ast):
        for node in _walk(fn.body):
            if not isinstance(node, c_ast.BinaryOp) or node.op not in ("/", "*"):
                continue
            for side in (node.left, node.right):
                const = _float_const(side)
                if const is None:
                    continue
                hits.append(lambda const=const: _float_to_int(const))
    return hits


def _float_to_int(const):
    try:
        number = int(float(const.value.rstrip("fFlL")))
    except ValueError:
        number = 0
    const.type = "int"
    const.value = str(number)


def _find_m05(kind):
    def find(ast):
        hits = []
        for fn, tree, loops in _each_loop(ast):
            guarded = _cond_vars(loops)
            mods = _mods(fn.body)
            for decl in _scalar_decls(fn):
                if decl.name in guarded or _in_for_init(tree, decl):
                    continue
                if kind == "max":
                    if not (isinstance(decl.init, c_ast.ArrayRef) and _int_const(decl.init.subscript) == 0):
                        continue
                else:
                    if _int_const(decl.init) != 0 or _mod_kind(mods, decl.name) != kind:
                        continue
                hits.append(lambda decl=decl: setattr(decl, "init", None))
        return hits
    return find


def _eq_to_assign(eq):
    left, right = eq.left, eq.right
    if _is_id(left) and isinstance(right, (c_ast.ID, c_ast.Constant)):
        return c_ast.Assignment("=", left, right)
    if _is_id(right) and isinstance(left, c_ast.Constant):
        return c_ast.Assignment("=", right, left)
    return None


def _find_m06(want):
    def find(ast):
        hits = []
        tree = _Tree(ast)
        for fn in _functions(ast):
            for node in _walk(fn.body):
                if not isinstance(node, c_ast.BinaryOp) or node.op not in ("==", "!="):
                    continue
                owner = _controller(tree, node)
                if want == "if" and not isinstance(owner, c_ast.If):
                    continue
                if want == "while" and not isinstance(owner, (c_ast.While, c_ast.DoWhile)):
                    continue
                if want == "while" and node.op not in ("==", "!="):
                    continue
                if want == "if" and node.op != "==":
                    continue
                replacement = _eq_to_assign(node)
                if replacement is None:
                    continue
                hits.append(lambda tree=tree, node=node, replacement=replacement: _replace(tree, node, replacement))
        return hits
    return find


def _find_m07_if(ast):
    hits = []
    tree = _Tree(ast)
    for fn in _functions(ast):
        for node in _walk(fn.body):
            if isinstance(node, c_ast.If) and node.iftrue is not None and not isinstance(node.iftrue, c_ast.EmptyStatement):
                if tree.attr.get(id(node), "").startswith("block_items"):
                    hits.append(lambda tree=tree, node=node: _semi_if(tree, node))
    return hits


def _semi_if(tree, node):
    body = node.iftrue
    node.iftrue = c_ast.EmptyStatement()
    _splice_after(tree, node, body)


def _find_m07_for(ast):
    hits = []
    for _, tree, loops in _each_loop(ast):
        for lp in loops:
            if lp.kind != "for" or lp.body is None or isinstance(lp.body, c_ast.EmptyStatement):
                continue
            if not str(tree.attr.get(id(lp.node), "")).startswith("block_items"):
                continue
            hits.append(lambda tree=tree, lp=lp: _semi_for(tree, lp))
    return hits




def _find_m07_while(ast):
    hits = []
    for _, tree, loops in _each_loop(ast):
        for lp in loops:
            if lp.kind != "while" or lp.body is None or isinstance(lp.body, c_ast.EmptyStatement):
                continue
            if not str(tree.attr.get(id(lp.node), "")).startswith("block_items"):
                continue
            hits.append(lambda tree=tree, node=lp.node: _semi_while(tree, node))
    return hits


def _semi_while(tree, node):
    body = node.stmt
    node.stmt = c_ast.EmptyStatement()
    _splice_after(tree, node, body)


def _find_m10(with_return0):
    def find(ast):
        hits = []
        tree = _Tree(ast)
        for fn in _functions(ast):
            if _is_void_fn(fn):
                continue
            for node in _walk(fn.body):
                if not isinstance(node, c_ast.Return) or node.expr is None:
                    continue
                if with_return0 and _int_const(node.expr) == 0:
                    continue
                hits.append(lambda tree=tree, node=node, with_return0=with_return0: _print_return(tree, node, with_return0))
        return hits
    return find


def _print_return(tree, node, with_return0):
    printed = _printf_d(node.expr)
    if with_return0:
        _expand_stmt(tree, node, [printed, c_ast.Return(_const_int(0))])
    else:
        _expand_stmt(tree, node, [printed])


def _find_oth_swap(ast):
    hits = []
    tree = _Tree(ast)
    for fn in _functions(ast):
        for node in _walk(fn.body):
            if not isinstance(node, c_ast.BinaryOp) or node.op != "-":
                continue
            if _under_attr(tree, node, _LOOPS, "cond") is not None:
                continue
            if _under_attr(tree, node, (c_ast.For,), "next") is not None:
                continue
            parent = tree.parent.get(id(node))
            if isinstance(parent, c_ast.ArrayRef) and tree.attr.get(id(node)) == "subscript":
                continue
            hits.append(lambda node=node: _swap(node))
    return hits


def _swap(node):
    node.left, node.right = node.right, node.left


def _find_oth_wrong_op(ast):
    hits = []
    for fn in _functions(ast):
        for node in _walk(fn.body):
            if isinstance(node, c_ast.Assignment) and node.op == "+=":
                hits.append(lambda node=node: setattr(node, "op", "-="))
            elif isinstance(node, c_ast.Assignment) and node.op == "*=":
                hits.append(lambda node=node: setattr(node, "op", "+="))
            elif isinstance(node, c_ast.BinaryOp) and node.op == "*" and _is_accumulator_mul(node):
                hits.append(lambda node=node: setattr(node, "op", "+"))
    return hits


def _is_accumulator_mul(node):
    parent_ok = isinstance(node.left, c_ast.ID) or isinstance(node.right, c_ast.ID)
    return parent_ok


def _find_oth_wrong_const(ast):
    hits = []
    tree = _Tree(ast)
    for fn in _functions(ast):
        for node in _walk(fn.body):
            if isinstance(node, c_ast.Constant) and not _skip_literal(tree, node):
                hits.append(lambda node=node: setattr(node, "value", _literal_edit(node)))
    return hits


def _find_oth_wrong_var(ast):
    hits = []
    for fn, tree, loops in _each_loop(ast):
        scalars, _, _ = _params(fn)
        for lp in loops:
            if not lp.var:
                continue
            prefer = set(lp.cond_ids) - {lp.var}
            replacement = None
            for name in scalars:
                if name != lp.var and name in prefer:
                    replacement = name
                    break
            if replacement is None:
                for name in scalars:
                    if name != lp.var:
                        replacement = name
                        break
            if replacement is None:
                continue
            for node in _walk(lp.body):
                if isinstance(node, c_ast.ArrayRef) and _is_id(node.subscript, lp.var) and tree.inside(node, lp.body):
                    hits.append(lambda tree=tree, node=node, replacement=replacement: _replace(tree, node, _id(replacement)))
    return hits


def _find_oth_flip(ast):
    hits = []
    tree = _Tree(ast)
    for fn in _functions(ast):
        for node in _walk(fn.body):
            if not isinstance(node, c_ast.BinaryOp) or node.op != ">":
                continue
            if not isinstance(_controller(tree, node), c_ast.If):
                continue
            hits.append(lambda node=node: setattr(node, "op", "<"))
    return hits


def _find_oth_drop(ast):
    hits = []
    for fn, tree, loops in _each_loop(ast):
        for node in _walk(fn.body):
            if not isinstance(node, (c_ast.Assignment, c_ast.FuncCall, c_ast.UnaryOp)):
                continue
            parent = tree.parent.get(id(node))
            attr = tree.attr.get(id(node), "")
            if not isinstance(parent, c_ast.Compound) or not attr.startswith("block_items"):
                continue
            if isinstance(node, c_ast.UnaryOp) and node.op not in _INC and node.op not in ("+", "-"):
                continue
            if any(lp.var and tree.inside(node, lp.node) and _progress_stmt(node, lp.var) for lp in loops):
                continue
            hits.append(lambda tree=tree, node=node: _delete_stmt(tree, node))
    return hits


def _find_u1(ast):
    hits = []
    tree = _Tree(ast)
    for fn in _functions(ast):
        for node in _walk(fn.body):
            if not isinstance(node, c_ast.BinaryOp) or node.op != "&&":
                continue
            left, right = node.left, node.right
            if not (isinstance(left, c_ast.BinaryOp) and isinstance(right, c_ast.BinaryOp)):
                continue
            if left.op != "<" or right.op != "<":
                continue
            if not (_is_id(left.right) and _is_id(right.left) and left.right.name == right.left.name):
                continue
            hits.append(lambda tree=tree, node=node: _chain(tree, node))
    return hits


def _chain(tree, node):
    _replace(tree, node, c_ast.BinaryOp("<", node.left, node.right.right))


def _eq_parts(node):
    if not isinstance(node, c_ast.BinaryOp) or node.op != "==":
        return None, None
    if _is_id(node.left) and isinstance(node.right, c_ast.Constant):
        return node.left.name, node.right
    if _is_id(node.right) and isinstance(node.left, c_ast.Constant):
        return node.right.name, node.left
    return None, None


def _find_u2(ast):
    hits = []
    for fn in _functions(ast):
        for node in _walk(fn.body):
            if not isinstance(node, c_ast.BinaryOp) or node.op != "||":
                continue
            left_name, _ = _eq_parts(node.left)
            right_name, const = _eq_parts(node.right)
            if left_name and left_name == right_name and const is not None:
                hits.append(lambda node=node, const=const: setattr(node, "right", const))
    return hits


_FINDERS = {
    "m01_le": _find_m01_le,
    "m01_nminus1": _find_m01_nminus1,
    "m01_start1_count": _find_m01_start1,
    "m01_countdown_ge0": _find_m01_countdown,
    "m01_while_le": _find_m01_while_le,
    "m08_index_n": _find_m08_index_n,
    "m08_onebased": _find_m08_onebased,
    "m08_iplus1": _find_m08_iplus1,
    "m08_first1": _find_m08_first1,
    "amb_le_array": _find_amb_le,
    "amb_start1_array": _find_amb_start1,
    "m02_drop_update": _find_m02_drop,
    "m02_reverse": _find_m02_reverse,
    "m02_update_in_branch": _find_m02_in_branch,
    "m02_wrong_var": _find_m02_wrong_var,
    "m03_decl_in_loop": _find_m03_decl,
    "m03_assign_in_loop": _find_m03_assign,
    "m04_drop_cast": _find_m04_drop_cast,
    "m04_int_literal": _find_m04_int_literal,
    "m04_cast_late": _find_m04_cast_late,
    "m05_drop_init_acc": _find_m05("acc"),
    "m05_drop_init_counter": _find_m05("counter"),
    "m05_drop_init_max": _find_m05("max"),
    "m06_if_assign": _find_m06("if"),
    "m06_while_assign": _find_m06("while"),
    "m07_if_semi": _find_m07_if,
    "m07_for_semi": _find_m07_for,
    "m07_while_semi": _find_m07_while,
    "m10_printf_noreturn": _find_m10(False),
    "m10_printf_return0": _find_m10(True),
    "oth_swap_operands": _find_oth_swap,
    "oth_wrong_op": _find_oth_wrong_op,
    "oth_wrong_const": _find_oth_wrong_const,
    "oth_wrong_var": _find_oth_wrong_var,
    "oth_flip_branch_rel": _find_oth_flip,
    "oth_drop_stmt": _find_oth_drop,
    "u1_chained": _find_u1,
    "u2_or_chain": _find_u2,
}

_LABELS = {
    "m01_le": ("M01",),
    "m01_nminus1": ("M01",),
    "m01_start1_count": ("M01",),
    "m01_countdown_ge0": ("M01",),
    "m01_while_le": ("M01",),
    "m08_index_n": ("M08",),
    "m08_onebased": ("M08",),
    "m08_iplus1": ("M08",),
    "m08_first1": ("M08",),
    "amb_le_array": ("M01", "M08"),
    "amb_start1_array": ("M01", "M08"),
    "m02_drop_update": ("M02",),
    "m02_reverse": ("M02",),
    "m02_update_in_branch": ("M02",),
    "m02_wrong_var": ("M02",),
    "m03_decl_in_loop": ("M03",),
    "m03_assign_in_loop": ("M03",),
    "m04_drop_cast": ("M04",),
    "m04_int_literal": ("M04",),
    "m04_cast_late": ("M04",),
    "m05_drop_init_acc": ("M05",),
    "m05_drop_init_counter": ("M05",),
    "m05_drop_init_max": ("M05",),
    "m06_if_assign": ("M06",),
    "m06_while_assign": ("M06",),
    "m07_if_semi": ("M07",),
    "m07_for_semi": ("M07",),
    "m07_while_semi": ("M07",),
    "m10_printf_noreturn": ("M10",),
    "m10_printf_return0": ("M10",),
    "oth_swap_operands": ("OTHER",),
    "oth_wrong_op": ("OTHER",),
    "oth_wrong_const": ("OTHER",),
    "oth_wrong_var": ("OTHER",),
    "oth_flip_branch_rel": ("OTHER",),
    "oth_drop_stmt": ("OTHER",),
    "u1_chained": ("U1",),
    "u2_or_chain": ("U2",),
}

_TEST_ONLY = {"u1_chained", "u2_or_chain"}
_OPS = None


def all_ops():
    """Every main-game operator, catalogue order. U1 and U2 are ``test_only``."""
    global _OPS
    if _OPS is None:
        _OPS = [Op(op_id, _LABELS[op_id], op_id in _TEST_ONLY) for op_id in _FINDERS]
    return list(_OPS)


def sites(op, code):
    """Indexes where ``op`` applies. Empty when the pattern is absent or the file does not parse."""
    try:
        ast = parse_ast(code)
    except Exception:
        return []
    finder = _FINDERS.get(op.op_id)
    if finder is None:
        return []
    return list(range(len(finder(ast))))


def apply(op, code, site):
    """Reprint of ``code`` with ``sites(op, code)[site]`` edited. Parses a fresh tree first."""
    ast = parse_ast(code)
    edits = _FINDERS[op.op_id](ast)
    edits[site]()
    return c_generator.CGenerator().visit(ast)
