"""DSA mutation operators (ml_plan/03 §3.5.1).

Each operator reparses the function, edits one syntax-tree site, and reprints
with ``pycparser.c_generator``. A site is an index into ``sites()``: ``apply``
parses again, so an earlier edit cannot leak into the next one.
"""
import copy

from pycparser import c_ast, c_generator, c_parser

from ml.c_interp.preprocess import preprocess


class Op:
    def __init__(self, op_id, labels, test_only=False):
        self.op_id = op_id
        self.labels = tuple(labels)
        self.test_only = test_only

    def __repr__(self):
        return f"Op({self.op_id})"


def all_ops():
    return list(_OPS)


def sites(op, code):
    """Indexes of places ``op`` can edit. Empty when the shape is not in ``code``."""
    return list(range(len(_collect(op.op_id, _parse(code)))))


def apply(op, code, site):
    """Return the whole function after editing ``sites(op, code)[site]``."""
    ast = _parse(code)
    found = _collect(op.op_id, ast)
    found[site]()
    return c_generator.CGenerator().visit(ast)


# ---------------------------------------------------------------- catalogue

_OPS = [
    Op("amb_pair_bound", ("M01", "M08")),
    Op("m08_mirror", ("M08",)),
    Op("m01_half_bound", ("M01",)),
    Op("m02_pointer_stuck", ("M02",)),
    Op("m10_print_bool", ("M10",)),
    Op("d01_else_return", ("D01",)),
    Op("d01_else_flag_reset", ("D01",)),
    Op("d02_low_mid", ("D02",)),
    Op("d02_high_mid", ("D02",)),
    Op("d03_drop_temp", ("D03",)),
    Op("d03_rotate_no_save", ("D03",)),
    Op("d04_drop_outer", ("D04",)),
    Op("d04_outer_once", ("D04",)),
    Op("d05_drop_base", ("D05",)),
    Op("d05_unreachable_base", ("D05",)),
    Op("d06_same_arg", ("D06",)),
    Op("d06_grow_arg", ("D06",)),
    Op("d07_discard", ("D07",)),
    Op("d08_str_literal", ("D08",)),
    Op("d08_array_eq", ("D08",)),
]

_COLLECT = {}


def _collector(op_id):
    def wrap(fn):
        _COLLECT[op_id] = fn
        return fn
    return wrap


def _collect(op_id, ast):
    return _COLLECT[op_id](ast)


# ---------------------------------------------------------------- tree helpers

def _parse(code):
    return c_parser.CParser().parse(preprocess(code))


def _fn(ast):
    return next(node for node in ast.ext if isinstance(node, c_ast.FuncDef))


def _fname(fn):
    return fn.decl.name


def _walk(node):
    yield node
    for _, child in node.children():
        yield from _walk(child)


def _const(node):
    """Integer value of a constant, including the unary minus pycparser uses for ``-1``."""
    if isinstance(node, c_ast.Constant) and node.type == "int":
        try:
            return int(node.value, 0)
        except ValueError:
            return None
    if isinstance(node, c_ast.UnaryOp) and node.op == "-" and isinstance(node.expr, c_ast.Constant):
        value = _const(node.expr)
        return None if value is None else -value
    return None


def _int_node(value):
    if value < 0:
        return c_ast.UnaryOp("-", c_ast.Constant("int", str(-value)))
    return c_ast.Constant("int", str(value))


def _clone(node):
    return copy.deepcopy(node)


def _same(a, b):
    gen = c_generator.CGenerator()
    return gen.visit(a) == gen.visit(b)


def _parents(root):
    parent = {}

    def walk(node, up):
        parent[id(node)] = up
        for _, child in node.children():
            walk(child, node)

    walk(root, None)
    return parent


def _params(fn):
    params = fn.decl.type.args.params if fn.decl.type.args else []
    return [(p.name, isinstance(p.type, c_ast.ArrayDecl)) for p in params]


def _scalar_params(fn):
    return [name for name, is_array in _params(fn) if not is_array]


def _replace_stmt(node, parents, new_nodes):
    up = parents[id(node)]
    if isinstance(up, c_ast.Compound):
        items = up.block_items
        index = items.index(node)
        items[index:index + 1] = new_nodes
        return
    if len(new_nodes) == 1:
        repl = new_nodes[0]
    elif not new_nodes:
        repl = c_ast.EmptyStatement()
    else:
        repl = c_ast.Compound(new_nodes)
    if isinstance(up, c_ast.If):
        if up.iftrue is node:
            up.iftrue = repl
        else:
            up.iffalse = repl
    elif isinstance(up, (c_ast.For, c_ast.While, c_ast.DoWhile)):
        up.stmt = repl


def _delete_stmt(node, parents):
    up = parents[id(node)]
    if isinstance(up, c_ast.Compound):
        up.block_items.remove(node)
        return
    _replace_stmt(node, parents, [])


def _loops(node):
    for child in _walk(node):
        if isinstance(child, (c_ast.For, c_ast.While, c_ast.DoWhile)):
            yield child


def _contains_loop(node):
    for child in _walk(node):
        if child is not node and isinstance(child, (c_ast.For, c_ast.While, c_ast.DoWhile)):
            return True
    return False


def _under(node, parents, kind, stop=None):
    cur = parents.get(id(node))
    while cur is not None and cur is not stop:
        if isinstance(cur, kind):
            return True
        cur = parents.get(id(cur))
    return False


def _is_self_call(node, fname):
    return isinstance(node, c_ast.FuncCall) and isinstance(node.name, c_ast.ID) and node.name.name == fname


def _has_call(node, fname):
    return any(_is_self_call(child, fname) for child in _walk(node))


def _has_return(node):
    return any(isinstance(child, c_ast.Return) for child in _walk(node))


def _minus_one(expr):
    """Expression with one ``- 1`` removed, or None. ``(n - 1) - i`` becomes ``n - i``."""
    if isinstance(expr, c_ast.BinaryOp) and expr.op == "-" and _const(expr.right) == 1:
        return expr.left
    if isinstance(expr, c_ast.BinaryOp):
        left = _minus_one(expr.left)
        if left is not None:
            return c_ast.BinaryOp(expr.op, left, expr.right)
        right = _minus_one(expr.right)
        if right is not None:
            return c_ast.BinaryOp(expr.op, expr.left, right)
    return None


def _plus_one_index(node, var):
    """True when some ``a[var + 1]`` appears under ``node``."""
    for child in _walk(node):
        if not isinstance(child, c_ast.ArrayRef) or not isinstance(child.subscript, c_ast.BinaryOp):
            continue
        sub = child.subscript
        if sub.op != "+":
            continue
        if isinstance(sub.left, c_ast.ID) and sub.left.name == var and _const(sub.right) == 1:
            return True
        if isinstance(sub.right, c_ast.ID) and sub.right.name == var and _const(sub.left) == 1:
            return True
    return False


def _mirror_parts(sub):
    """``(base - 1) - index`` → (base, index)."""
    if not (isinstance(sub, c_ast.BinaryOp) and sub.op == "-" and isinstance(sub.right, c_ast.ID)):
        return None
    left = sub.left
    if isinstance(left, c_ast.BinaryOp) and left.op == "-" and _const(left.right) == 1:
        return left.left, sub.right
    return None


def _half_dividend(expr):
    if isinstance(expr, c_ast.BinaryOp) and expr.op == "/" and _const(expr.right) == 2:
        return expr.left
    return None


def _id_step(node, name):
    """``name + 1``, ``name - 1``, ``name += 1`` or ``name++`` / ``name--``."""
    if isinstance(node, c_ast.UnaryOp) and node.op in ("p++", "++", "p--", "--"):
        return isinstance(node.expr, c_ast.ID) and node.expr.name == name
    if isinstance(node, c_ast.Assignment) and isinstance(node.lvalue, c_ast.ID) and node.lvalue.name == name:
        if node.op in ("+=", "-="):
            return True
        if node.op == "=" and isinstance(node.rvalue, c_ast.BinaryOp) and node.rvalue.op in ("+", "-"):
            rhs = node.rvalue
            return isinstance(rhs.left, c_ast.ID) and rhs.left.name == name and _const(rhs.right) == 1
    return False


def _bare_index(node, name):
    for child in _walk(node):
        if isinstance(child, c_ast.ArrayRef) and isinstance(child.subscript, c_ast.ID) and child.subscript.name == name:
            return True
    return False


def _array_base(node):
    if isinstance(node, c_ast.ArrayRef) and isinstance(node.name, c_ast.ID):
        return node.name.name
    return None


def _const_index(node, value):
    return isinstance(node, c_ast.ArrayRef) and _const(node.subscript) == value


def _mid_names(fn):
    """Names initialised from an expression that divides by 2 (the midpoint)."""
    names = set()
    for node in _walk(fn):
        init = None
        name = None
        if isinstance(node, c_ast.Decl) and node.init is not None:
            name, init = node.name, node.init
        elif isinstance(node, c_ast.Assignment) and isinstance(node.lvalue, c_ast.ID):
            name, init = node.lvalue.name, node.rvalue
        if init is not None and any(
            isinstance(child, c_ast.BinaryOp) and child.op == "/" and _const(child.right) == 2
            for child in _walk(init)
        ):
            names.add(name)
    return names


def _mid_offset(expr, mids, op):
    if (isinstance(expr, c_ast.BinaryOp) and expr.op == op and isinstance(expr.left, c_ast.ID)
            and expr.left.name in mids and _const(expr.right) == 1):
        return expr.left.name
    return None


def _saved_cell(stmt):
    if isinstance(stmt, c_ast.Decl) and isinstance(stmt.init, c_ast.ArrayRef):
        return stmt.name, stmt.init
    if (isinstance(stmt, c_ast.Assignment) and stmt.op == "=" and isinstance(stmt.lvalue, c_ast.ID)
            and isinstance(stmt.rvalue, c_ast.ArrayRef)):
        return stmt.lvalue.name, stmt.rvalue
    return None


def _array_pair(stmt):
    if (isinstance(stmt, c_ast.Assignment) and stmt.op == "=" and isinstance(stmt.lvalue, c_ast.ArrayRef)
            and isinstance(stmt.rvalue, c_ast.ArrayRef)):
        return stmt.lvalue, stmt.rvalue
    return None


def _from_temp(stmt, temp):
    if (isinstance(stmt, c_ast.Assignment) and stmt.op == "=" and isinstance(stmt.lvalue, c_ast.ArrayRef)
            and isinstance(stmt.rvalue, c_ast.ID) and stmt.rvalue.name == temp):
        return stmt.lvalue
    return None


def _compounds(node):
    for child in _walk(node):
        if isinstance(child, c_ast.Compound):
            yield child


def _for_init_stmts(node):
    init = getattr(node, "init", None)
    if init is None:
        return []
    if isinstance(init, c_ast.DeclList):
        return list(init.decls)
    return [init]


def _body_stmts(node):
    body = node.stmt
    if isinstance(body, c_ast.Compound):
        return list(body.block_items or [])
    return [body]


def _bound_is_id_minus_one(cond):
    right = cond.right if isinstance(cond, c_ast.BinaryOp) else None
    return (isinstance(cond, c_ast.BinaryOp) and cond.op == "<" and isinstance(right, c_ast.BinaryOp)
            and right.op == "-" and isinstance(right.left, c_ast.ID) and _const(right.right) == 1)


def _param_bin(expr, params, op, k=None):
    if not (isinstance(expr, c_ast.BinaryOp) and expr.op == op and isinstance(expr.left, c_ast.ID)):
        return None
    if expr.left.name not in params:
        return None
    value = _const(expr.right)
    if value is None or (k is not None and value != k):
        return None
    return expr.left.name


def _flag_one(node):
    """Name assigned the constant 1 under ``node``, if that is all the branch does."""
    if isinstance(node, c_ast.Assignment) and node.op == "=" and isinstance(node.lvalue, c_ast.ID):
        if _const(node.rvalue) == 1:
            return node.lvalue.name
        return None
    if isinstance(node, c_ast.Compound):
        names = [_flag_one(stmt) for stmt in (node.block_items or [])]
        names = [name for name in names if name]
        if len(names) == 1 and len(node.block_items or []) == 1:
            return names[0]
    return None


def _failure_value(fn):
    for node in _walk(fn):
        if isinstance(node, c_ast.Return) and _const(node.expr) == -1:
            return -1
    return 0


def _char_to_string(node):
    if not isinstance(node, c_ast.Constant) or node.type != "char":
        return None
    raw = node.value
    if len(raw) >= 2 and raw[0] == "'" and raw[-1] == "'":
        return c_ast.Constant("string", '"' + raw[1:-1] + '"')
    return None


# ---------------------------------------------------------------- operators

@_collector("amb_pair_bound")
def _amb_pair_bound(ast):
    """``j < n - 1`` (or ``j < n - 1 - i``) → drop the ``- 1``, when the body reads ``a[j + 1]``."""
    found = []
    for loop in _loops(ast):
        cond = loop.cond
        if not (isinstance(cond, c_ast.BinaryOp) and cond.op == "<" and isinstance(cond.left, c_ast.ID)):
            continue
        if _minus_one(cond.right) is None or not _plus_one_index(loop.stmt, cond.left.name):
            continue

        def mutate(cond=cond):
            cond.right = _minus_one(cond.right)

        found.append(mutate)
    return found


@_collector("m08_mirror")
def _m08_mirror(ast):
    """``a[n - 1 - i]`` → ``a[n - i]``."""
    found = []
    for node in _walk(ast):
        if not isinstance(node, c_ast.ArrayRef):
            continue
        parts = _mirror_parts(node.subscript)
        if parts is None:
            continue
        base, index = parts

        def mutate(node=node, base=base, index=index):
            node.subscript = c_ast.BinaryOp("-", base, index)

        found.append(mutate)
    return found


@_collector("m01_half_bound")
def _m01_half_bound(ast):
    """``i < n / 2`` → ``i < n`` (a reverse then walks the whole array and undoes itself)."""
    found = []
    for loop in _loops(ast):
        cond = loop.cond
        if not (isinstance(cond, c_ast.BinaryOp) and cond.op == "<"):
            continue
        dividend = _half_dividend(cond.right)
        if dividend is None:
            continue

        def mutate(cond=cond, dividend=dividend):
            cond.right = dividend

        found.append(mutate)
    return found


@_collector("m02_pointer_stuck")
def _m02_pointer_stuck(ast):
    """Delete ``i++`` or ``j--`` in one branch of a loop whose test is ``i < j``."""
    found = []
    parents = _parents(ast)
    for loop in _loops(ast):
        cond = loop.cond
        if not (isinstance(cond, c_ast.BinaryOp) and cond.op == "<" and isinstance(cond.left, c_ast.ID)
                and isinstance(cond.right, c_ast.ID) and cond.left.name != cond.right.name):
            continue
        names = {cond.left.name, cond.right.name}
        if not all(_bare_index(loop, name) for name in names):
            continue
        for node in _walk(loop.stmt):
            if not any(_id_step(node, name) for name in names):
                continue
            if not _under(node, parents, c_ast.If, stop=loop):
                continue

            def mutate(node=node, parents=parents):
                _delete_stmt(node, parents)

            found.append(mutate)
    return found


@_collector("m10_print_bool")
def _m10_print_bool(ast):
    """``return 1`` / ``return 0`` → ``printf("yes")`` / ``printf("no")``."""
    found = []
    parents = _parents(ast)
    for node in _walk(ast):
        if not isinstance(node, c_ast.Return) or _const(node.expr) not in (0, 1):
            continue
        word = "yes" if _const(node.expr) == 1 else "no"

        def mutate(node=node, parents=parents, word=word):
            call = c_ast.FuncCall(
                c_ast.ID("printf"),
                c_ast.ExprList([c_ast.Constant("string", '"' + word + '"')]),
            )
            _replace_stmt(node, parents, [call])

        found.append(mutate)
    return found


@_collector("d01_else_return")
def _d01_else_return(ast):
    """Add ``else return -1`` (or ``else return 0``) to a returning ``if`` inside a loop."""
    found = []
    fn = _fn(ast)
    parents = _parents(ast)
    value = _failure_value(fn)
    for node in _walk(fn):
        if not isinstance(node, c_ast.If) or node.iffalse is not None:
            continue
        if not _under(node, parents, (c_ast.For, c_ast.While, c_ast.DoWhile)):
            continue
        if not _has_return(node.iftrue):
            continue

        def mutate(node=node, value=value):
            node.iffalse = c_ast.Return(_int_node(value))

        found.append(mutate)
    return found


@_collector("d01_else_flag_reset")
def _d01_else_flag_reset(ast):
    """Add ``else found = 0`` to an ``if`` inside a loop that sets the flag to 1."""
    found = []
    parents = _parents(ast)
    for node in _walk(_fn(ast)):
        if not isinstance(node, c_ast.If) or node.iffalse is not None or _has_return(node):
            continue
        if not _under(node, parents, (c_ast.For, c_ast.While, c_ast.DoWhile)):
            continue
        name = _flag_one(node.iftrue)
        if name is None:
            continue

        def mutate(node=node, name=name):
            node.iffalse = c_ast.Assignment("=", c_ast.ID(name), _int_node(0))

        found.append(mutate)
    return found


def _d02(ast, op):
    mids = _mid_names(_fn(ast))
    found = []
    for node in _walk(ast):
        if not (isinstance(node, c_ast.Assignment) and node.op == "=" and isinstance(node.lvalue, c_ast.ID)):
            continue
        mid = _mid_offset(node.rvalue, mids, op)
        if mid is None or mid == node.lvalue.name:
            continue

        def mutate(node=node, mid=mid):
            node.rvalue = c_ast.ID(mid)

        found.append(mutate)
    return found


@_collector("d02_low_mid")
def _d02_low_mid(ast):
    """``low = mid + 1`` → ``low = mid``."""
    return _d02(ast, "+")


@_collector("d02_high_mid")
def _d02_high_mid(ast):
    """``high = mid - 1`` → ``high = mid``."""
    return _d02(ast, "-")


@_collector("d03_drop_temp")
def _d03_drop_temp(ast):
    """``t = a[i]; a[i] = a[j]; a[j] = t`` → ``a[i] = a[j]; a[j] = a[i]``."""
    found = []
    for comp in _compounds(ast):
        items = comp.block_items or []
        for index in range(len(items) - 2):
            saved = _saved_cell(items[index])
            pair = _array_pair(items[index + 1])
            if saved is None or pair is None:
                continue
            temp, cell = saved
            back = _from_temp(items[index + 2], temp)
            if back is None:
                continue
            dest, src = pair
            # The temp must be holding one of the two cells the middle assignment exchanges.
            if not (_same(cell, dest) or _same(cell, src)):
                continue
            if not (_same(back, dest) or _same(back, src)):
                continue

            def mutate(items=items, index=index, dest=dest, src=src):
                first = c_ast.Assignment("=", _clone(dest), _clone(src))
                second = c_ast.Assignment("=", _clone(src), _clone(dest))
                items[index:index + 3] = [first, second]

            found.append(mutate)
    return found


@_collector("d03_rotate_no_save")
def _d03_rotate_no_save(ast):
    """Drop ``int first = a[0]`` and write ``a[n - 1] = a[0]`` after the shift."""
    found = []
    fn = _fn(ast)
    items = fn.body.block_items or []
    for index, stmt in enumerate(items):
        saved = _saved_cell(stmt)
        if saved is None or not _const_index(saved[1], 0):
            continue
        temp, cell = saved
        use = None
        for later in items[index + 1:]:
            if _from_temp(later, temp) is not None:
                use = later
        if use is None:
            continue

        def mutate(items=items, stmt=stmt, use=use, cell=cell):
            use.rvalue = _clone(cell)
            items.remove(stmt)

        found.append(mutate)
    return found


@_collector("d04_drop_outer")
def _d04_drop_outer(ast):
    """Remove the outer loop and keep one inner pass, with the outer index left at 0."""
    found = []
    parents = _parents(ast)
    for loop in _loops(ast):
        if not _contains_loop(loop.stmt):
            continue

        def mutate(loop=loop, parents=parents):
            stmts = _body_stmts(loop)
            if isinstance(loop, c_ast.For):
                stmts = _for_init_stmts(loop) + stmts
            _replace_stmt(loop, parents, stmts)

        found.append(mutate)
    return found


@_collector("d04_outer_once")
def _d04_outer_once(ast):
    """Outer bound ``i < n - 1`` → ``i < 1``."""
    found = []
    for loop in _loops(ast):
        if not _contains_loop(loop.stmt) or not _bound_is_id_minus_one(loop.cond):
            continue

        def mutate(loop=loop):
            loop.cond.right = _int_node(1)

        found.append(mutate)
    return found


@_collector("d05_drop_base")
def _d05_drop_base(ast):
    """Delete a base-case ``if`` that returns without calling this function."""
    fn = _fn(ast)
    fname = _fname(fn)
    if not _has_call(fn, fname):
        return []
    found = []
    parents = _parents(ast)
    for node in _walk(fn):
        if not isinstance(node, c_ast.If):
            continue
        if not _has_return(node.iftrue) or _has_call(node.iftrue, fname):
            continue

        def mutate(node=node, parents=parents):
            _delete_stmt(node, parents)

        found.append(mutate)
    return found


@_collector("d05_unreachable_base")
def _d05_unreachable_base(ast):
    """``n <= …`` → ``n == 0`` and recursive ``n - 1`` → ``n - 2`` (odd n never hits the base)."""
    fn = _fn(ast)
    fname = _fname(fn)
    base = None
    for node in _walk(fn):
        if isinstance(node, c_ast.If) and _has_return(node.iftrue) and not _has_call(node.iftrue, fname):
            base = node
            break
    if base is None or not isinstance(base.cond, c_ast.BinaryOp) or not isinstance(base.cond.left, c_ast.ID):
        return []
    param = base.cond.left.name
    slots = []
    for node in _walk(fn):
        if not _is_self_call(node, fname) or node.args is None:
            continue
        for index, arg in enumerate(node.args.exprs):
            if _param_bin(arg, {param}, "-", 1):
                slots.append((node, index))
    if not slots:
        return []

    def mutate(base=base, param=param, slots=slots):
        base.cond = c_ast.BinaryOp("==", c_ast.ID(param), _int_node(0))
        for call, index in slots:
            call.args.exprs[index] = c_ast.BinaryOp("-", c_ast.ID(param), _int_node(2))

    return [mutate]


def _d06_slots(ast, kind):
    fn = _fn(ast)
    fname = _fname(fn)
    params = set(_scalar_params(fn))
    found = []
    for node in _walk(fn):
        if not _is_self_call(node, fname) or node.args is None:
            continue
        for index, arg in enumerate(node.args.exprs):
            if kind == "same-minus" and _param_bin(arg, params, "-", 1):
                name = arg.left.name

                def mutate(node=node, index=index, name=name):
                    node.args.exprs[index] = c_ast.ID(name)

                found.append(mutate)
            elif kind == "same-div" and _param_bin(arg, params, "/"):
                name = arg.left.name
                k = _const(arg.right)

                def mutate(node=node, index=index, name=name, k=k):
                    node.args.exprs[index] = c_ast.BinaryOp("%", c_ast.ID(name), _int_node(k))

                found.append(mutate)
            elif kind == "grow" and _param_bin(arg, params, "-", 1):
                name = arg.left.name

                def mutate(node=node, index=index, name=name):
                    node.args.exprs[index] = c_ast.BinaryOp("+", c_ast.ID(name), _int_node(1))

                found.append(mutate)
    return found


@_collector("d06_same_arg")
def _d06_same_arg(ast):
    """``f(n - 1)`` → ``f(n)`` and ``f(n / 10)`` → ``f(n % 10)``."""
    return _d06_slots(ast, "same-minus") + _d06_slots(ast, "same-div")


@_collector("d06_grow_arg")
def _d06_grow_arg(ast):
    """``f(n - 1)`` → ``f(n + 1)``."""
    return _d06_slots(ast, "grow")


@_collector("d07_discard")
def _d07_discard(ast):
    """``return n * f(n - 1)`` → ``f(n - 1); return n`` (also ``+``)."""
    fn = _fn(ast)
    fname = _fname(fn)
    parents = _parents(ast)
    found = []
    for node in _walk(fn):
        if not isinstance(node, c_ast.Return) or not isinstance(node.expr, c_ast.BinaryOp):
            continue
        expr = node.expr
        if expr.op not in ("+", "*"):
            continue
        call = local = None
        if _is_self_call(expr.left, fname) and not _has_call(expr.right, fname):
            call, local = expr.left, expr.right
        elif _is_self_call(expr.right, fname) and not _has_call(expr.left, fname):
            call, local = expr.right, expr.left
        if call is None:
            continue

        def mutate(node=node, parents=parents, call=call, local=local):
            _replace_stmt(node, parents, [call, c_ast.Return(local)])

        found.append(mutate)
    return found


@_collector("d08_str_literal")
def _d08_str_literal(ast):
    """``s[i] == 'a'`` → ``s[i] == "a"``."""
    found = []
    for node in _walk(ast):
        if not isinstance(node, c_ast.BinaryOp) or node.op != "==":
            continue
        for side in ("left", "right"):
            if _char_to_string(getattr(node, side)) is None:
                continue

            def mutate(node=node, side=side):
                setattr(node, side, _char_to_string(getattr(node, side)))

            found.append(mutate)
    return found


@_collector("d08_array_eq")
def _d08_array_eq(ast):
    """A char-by-char loop comparing two arrays becomes ``if (s == t)``."""
    found = []
    parents = _parents(ast)
    for loop in _loops(ast):
        pair = None
        for node in _walk(loop.stmt):
            if not isinstance(node, c_ast.BinaryOp) or node.op not in ("==", "!="):
                continue
            left, right = _array_base(node.left), _array_base(node.right)
            if left and right and left != right:
                pair = (left, right)
                break
        if pair is None:
            continue
        left, right = pair

        def mutate(loop=loop, parents=parents, left=left, right=right):
            cond = c_ast.BinaryOp("==", c_ast.ID(left), c_ast.ID(right))
            stmt = c_ast.If(cond, c_ast.Return(_int_node(1)), c_ast.Return(_int_node(0)))
            _replace_stmt(loop, parents, [stmt])

        found.append(mutate)
    return found
