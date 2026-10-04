"""Static half of the DSA verifier (ml_plan/03 §3.5.4).

Trace events (``depth_cap_hit``, a multiset change, ``discarded_call_value``) are
checked by the generator, not here. ``holds`` is true when the code already has
the shape the operator is supposed to leave behind.
"""
from pycparser import c_ast, c_generator, c_parser

from ml.c_interp.preprocess import preprocess


def holds(op_id: str, code: str) -> bool:
    """Static half of 03 §3.5.4.

    D01: else-return or else-flag assign inside the match if.
    D02: low = mid or high = mid with no ±1.
    D03: consecutive cross-assignments without a temp.
    D04: outer loop removed or bound i < 1.
    D05: no base-case if that returns without a self-call.
    D06: recursive argument is the same parameter or parameter+1 or n%10 where n/10 was.
    D07: self-call used as a statement.
    D08: == with a string literal or between two array names.
    Trace events (depth_cap_hit, multiset change) are C3's job; do not require them inside holds().
    """
    try:
        ast = c_parser.CParser().parse(preprocess(code))
    except Exception:
        return False
    check = _CHECKS.get(op_id)
    if check is None:
        return False
    return check(ast)


def _fn(ast):
    return next(node for node in ast.ext if isinstance(node, c_ast.FuncDef))


def _fname(fn):
    return fn.decl.name


def _walk(node):
    yield node
    for _, child in node.children():
        yield from _walk(child)


def _const(node):
    if isinstance(node, c_ast.Constant) and node.type == "int":
        try:
            return int(node.value, 0)
        except ValueError:
            return None
    if isinstance(node, c_ast.UnaryOp) and node.op == "-" and isinstance(node.expr, c_ast.Constant):
        value = _const(node.expr)
        return None if value is None else -value
    return None


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
    return [(node.name, isinstance(node.type, c_ast.ArrayDecl)) for node in params]


def _loops(node):
    for child in _walk(node):
        if isinstance(child, (c_ast.For, c_ast.While, c_ast.DoWhile)):
            yield child


def _nested(node):
    for loop in _loops(node):
        if any(inner is not loop and isinstance(inner, (c_ast.For, c_ast.While, c_ast.DoWhile))
               for inner in _walk(loop.stmt)):
            return True
    return False


def _is_self(node, fname):
    return isinstance(node, c_ast.FuncCall) and isinstance(node.name, c_ast.ID) and node.name.name == fname


def _has_call(node, fname):
    return any(_is_self(child, fname) for child in _walk(node))


def _has_return(node):
    return any(isinstance(child, c_ast.Return) for child in _walk(node))


def _inside_loop(node, parents):
    cur = parents.get(id(node))
    while cur is not None:
        if isinstance(cur, (c_ast.For, c_ast.While, c_ast.DoWhile)):
            return True
        cur = parents.get(id(cur))
    return False


def _flag_const(node):
    if isinstance(node, c_ast.Assignment) and node.op == "=" and isinstance(node.lvalue, c_ast.ID):
        if _const(node.rvalue) is not None:
            return node.lvalue.name
        return None
    if isinstance(node, c_ast.Compound):
        for stmt in node.block_items or []:
            name = _flag_const(stmt)
            if name:
                return name
    return None


def _minus_const(node):
    return isinstance(node, c_ast.BinaryOp) and node.op == "-" and _const(node.right) == 1


def _mid_names(fn):
    names = set()
    for node in _walk(fn):
        init = name = None
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


def _compounds(node):
    for child in _walk(node):
        if isinstance(child, c_ast.Compound):
            yield child


def _array_assign(stmt):
    if (isinstance(stmt, c_ast.Assignment) and stmt.op == "=" and isinstance(stmt.lvalue, c_ast.ArrayRef)
            and isinstance(stmt.rvalue, c_ast.ArrayRef)):
        return stmt.lvalue, stmt.rvalue
    return None


def _writes(node, name):
    for child in _walk(node):
        if isinstance(child, c_ast.UnaryOp) and child.op in ("p++", "++", "p--", "--"):
            if isinstance(child.expr, c_ast.ID) and child.expr.name == name:
                return True
        if isinstance(child, c_ast.Assignment) and isinstance(child.lvalue, c_ast.ID) and child.lvalue.name == name:
            return True
    return False


def _scalar_params(fn):
    return {name for name, is_array in _params(fn) if not is_array}


def _param_plus(expr, params):
    return (isinstance(expr, c_ast.BinaryOp) and expr.op == "+" and isinstance(expr.left, c_ast.ID)
            and expr.left.name in params and _const(expr.right) == 1)


def _param_mod(expr, params):
    return (isinstance(expr, c_ast.BinaryOp) and expr.op == "%" and isinstance(expr.left, c_ast.ID)
            and expr.left.name in params and _const(expr.right) is not None)


def _param_minus(expr, params, k):
    return (isinstance(expr, c_ast.BinaryOp) and expr.op == "-" and isinstance(expr.left, c_ast.ID)
            and expr.left.name in params and _const(expr.right) == k)


def _plain_diff(node):
    return (isinstance(node, c_ast.BinaryOp) and node.op == "-"
            and isinstance(node.left, c_ast.ID) and isinstance(node.right, c_ast.ID))


def _mirror_sub(node):
    """``(name - 1) - index`` still present, the shape a half-bound reverse keeps."""
    if not (isinstance(node, c_ast.BinaryOp) and node.op == "-" and isinstance(node.right, c_ast.ID)):
        return False
    left = node.left
    return isinstance(left, c_ast.BinaryOp) and left.op == "-" and _const(left.right) == 1


def _plus1(node, var):
    for child in _walk(node):
        if not isinstance(child, c_ast.ArrayRef) or not isinstance(child.subscript, c_ast.BinaryOp):
            continue
        sub = child.subscript
        if sub.op != "+":
            continue
        if isinstance(sub.left, c_ast.ID) and sub.left.name == var and _const(sub.right) == 1:
            return True
    return False


def _moved(node, names):
    found = set()
    for child in _walk(node):
        if isinstance(child, c_ast.UnaryOp) and child.op in ("p++", "++", "p--", "--"):
            if isinstance(child.expr, c_ast.ID) and child.expr.name in names:
                found.add(child.expr.name)
        if isinstance(child, c_ast.Assignment) and isinstance(child.lvalue, c_ast.ID) and child.lvalue.name in names:
            if child.op in ("+=", "-=") or (
                child.op == "=" and isinstance(child.rvalue, c_ast.BinaryOp)
                and child.rvalue.op in ("+", "-") and isinstance(child.rvalue.left, c_ast.ID)
                and child.rvalue.left.name == child.lvalue.name and _const(child.rvalue.right) == 1
            ):
                found.add(child.lvalue.name)
    return found


def _cond_eq_zero(cond):
    if not isinstance(cond, c_ast.BinaryOp) or cond.op != "==":
        return False
    if isinstance(cond.left, c_ast.ID) and _const(cond.right) == 0:
        return True
    if isinstance(cond.right, c_ast.ID) and _const(cond.left) == 0:
        return True
    return False


def _base_ifs(fn, fname):
    found = []
    for node in _walk(fn):
        if isinstance(node, c_ast.If) and _has_return(node.iftrue) and not _has_call(node.iftrue, fname):
            found.append(node)
    return found


def _shift(dest, src):
    """``a[i] = a[i + 1]`` or ``a[i - 1] = a[i]``."""
    if (isinstance(src, c_ast.BinaryOp) and src.op == "+" and _const(src.right) == 1
            and _same(dest, src.left)):
        return True
    if (isinstance(dest, c_ast.BinaryOp) and dest.op == "-" and _const(dest.right) == 1
            and _same(src, dest.left)):
        return True
    return False


def _has_shift(fn):
    for node in _walk(fn):
        pair = _array_assign(node)
        if pair and _shift(pair[0].subscript, pair[1].subscript):
            return True
    return False


def _saved_a0(fn):
    for node in _walk(fn):
        if isinstance(node, c_ast.Decl) and isinstance(node.init, c_ast.ArrayRef) and _const(node.init.subscript) == 0:
            return True
    return False


def _assigns_a0(fn):
    for node in _walk(fn):
        if isinstance(node, c_ast.Assignment) and isinstance(node.rvalue, c_ast.ArrayRef) and _const(node.rvalue.subscript) == 0:
            return True
    return False


def _has_cross(ast):
    for comp in _compounds(ast):
        items = comp.block_items or []
        for left, right in zip(items, items[1:]):
            a, b = _array_assign(left), _array_assign(right)
            if a and b and _same(a[0], b[1]) and _same(a[1], b[0]):
                return True
    return False


def _unsaved_rotate(ast):
    fn = _fn(ast)
    return _has_shift(fn) and _assigns_a0(fn) and not _saved_a0(fn)


def _zero_decls(fn):
    names = []
    for stmt in fn.body.block_items or []:
        if isinstance(stmt, c_ast.Decl) and _const(stmt.init) == 0:
            names.append(stmt.name)
    return names


def _array_move(fn):
    return any(_array_assign(node) for node in _walk(fn))


def _assigned_inside_loop(fn, name):
    for loop in _loops(fn):
        if _writes(loop.stmt, name):
            return True
    return False


def _bound_lt_one(ast):
    for loop in _loops(ast):
        cond = loop.cond
        if not (isinstance(cond, c_ast.BinaryOp) and cond.op == "<" and _const(cond.right) == 1):
            continue
        if any(isinstance(inner, (c_ast.For, c_ast.While, c_ast.DoWhile)) for inner in _walk(loop.stmt)):
            return True
    return False


def _unwrapped_pass(ast):
    """One pass left behind: a frozen ``i = 0`` and an array move, no outer loop around it."""
    if _nested(ast):
        return False
    fn = _fn(ast)
    if not _array_move(fn):
        return False
    frozen = [name for name in _zero_decls(fn) if not _assigned_inside_loop(fn, name)]
    return bool(frozen)


def _discarded_call(fn):
    fname = _fname(fn)

    def walk(node):
        if isinstance(node, c_ast.Compound):
            for stmt in node.block_items or []:
                if _is_self(stmt, fname) or walk(stmt):
                    return True
        elif isinstance(node, (c_ast.For, c_ast.While, c_ast.DoWhile)):
            return walk(node.stmt)
        elif isinstance(node, c_ast.If):
            for branch in (node.iftrue, node.iffalse):
                if branch is None:
                    continue
                if _is_self(branch, fname) or walk(branch):
                    return True
        return False

    return walk(fn.body)


def _d01(ast):
    parents = _parents(ast)
    for node in _walk(ast):
        if not isinstance(node, c_ast.If) or node.iffalse is None:
            continue
        if not _inside_loop(node, parents):
            continue
        then_match = _has_return(node.iftrue) or _flag_const(node.iftrue)
        else_bug = _has_return(node.iffalse) or _flag_const(node.iffalse)
        if then_match and else_bug:
            return True
    return False


def _d02(ast):
    mids = _mid_names(_fn(ast))
    if not mids:
        return False
    for node in _walk(ast):
        if (isinstance(node, c_ast.Assignment) and node.op == "=" and isinstance(node.lvalue, c_ast.ID)
                and isinstance(node.rvalue, c_ast.ID) and node.rvalue.name in mids
                and node.lvalue.name != node.rvalue.name):
            return True
    return False


def _d03(ast):
    return _has_cross(ast) or _unsaved_rotate(ast)


def _d04(ast):
    return _bound_lt_one(ast) or _unwrapped_pass(ast)


def _d05(ast):
    """No base that returns without a self-call, or a base ``n == 0`` whose step is ``n - 2``."""
    fn = _fn(ast)
    fname = _fname(fn)
    if not _has_call(fn, fname):
        return False
    bases = _base_ifs(fn, fname)
    if not bases:
        return True
    scalars = _scalar_params(fn)
    if not all(_cond_eq_zero(base.cond) for base in bases):
        return False
    for node in _walk(fn):
        if not _is_self(node, fname) or node.args is None:
            continue
        for arg in node.args.exprs:
            if _param_minus(arg, scalars, 2):
                return True
    return False


def _d06(ast):
    fn = _fn(ast)
    fname = _fname(fn)
    scalars = _scalar_params(fn)
    for node in _walk(fn):
        if not _is_self(node, fname) or node.args is None:
            continue
        for arg in node.args.exprs:
            if isinstance(arg, c_ast.ID) and arg.name in scalars:
                return True
            if _param_plus(arg, scalars) or _param_mod(arg, scalars):
                return True
    return False


def _d07(ast):
    return _discarded_call(_fn(ast))


def _d08(ast):
    fn = _fn(ast)
    arrays = {name for name, is_array in _params(fn) if is_array}
    for node in _walk(fn):
        if not isinstance(node, c_ast.BinaryOp) or node.op not in ("==", "!="):
            continue
        if (isinstance(node.left, c_ast.Constant) and node.left.type == "string") or (
            isinstance(node.right, c_ast.Constant) and node.right.type == "string"
        ):
            return True
        if isinstance(node.left, c_ast.ID) and isinstance(node.right, c_ast.ID):
            if node.left.name in arrays and node.right.name in arrays:
                return True
    return False


def _amb(ast):
    for loop in _loops(ast):
        cond = loop.cond
        if not (isinstance(cond, c_ast.BinaryOp) and cond.op == "<" and isinstance(cond.left, c_ast.ID)):
            continue
        if any(_minus_const(child) for child in _walk(cond.right)):
            continue
        if _plus1(loop.stmt, cond.left.name):
            return True
    return False


def _m08(ast):
    return any(isinstance(node, c_ast.ArrayRef) and _plain_diff(node.subscript) for node in _walk(ast))


def _m01(ast):
    for loop in _loops(ast):
        cond = loop.cond
        if not (isinstance(cond, c_ast.BinaryOp) and cond.op == "<" and isinstance(cond.left, c_ast.ID)
                and isinstance(cond.right, c_ast.ID)):
            continue
        if any(isinstance(node, c_ast.ArrayRef) and _mirror_sub(node.subscript) for node in _walk(loop.stmt)):
            return True
    return False


def _bare_index(node, name):
    """``name`` is itself the subscript of some array (a pointer, not a length)."""
    for child in _walk(node):
        if isinstance(child, c_ast.ArrayRef) and isinstance(child.subscript, c_ast.ID) and child.subscript.name == name:
            return True
    return False


def _m02(ast):
    for loop in _loops(ast):
        cond = loop.cond
        if not (isinstance(cond, c_ast.BinaryOp) and cond.op == "<" and isinstance(cond.left, c_ast.ID)
                and isinstance(cond.right, c_ast.ID) and cond.left.name != cond.right.name):
            continue
        names = {cond.left.name, cond.right.name}
        # ``i < n`` moves only i. A two-pointer loop indexes with both names.
        if not all(_bare_index(loop, name) for name in names):
            continue
        if len(_moved(loop.stmt, names)) == 1:
            return True
    return False


def _m10(ast):
    for node in _walk(ast):
        if not (isinstance(node, c_ast.FuncCall) and isinstance(node.name, c_ast.ID) and node.name.name == "printf"):
            continue
        args = node.args.exprs if node.args else []
        if len(args) == 1 and isinstance(args[0], c_ast.Constant) and args[0].value in ('"yes"', '"no"'):
            return True
    return False


_CHECKS = {
    "amb_pair_bound": _amb,
    "m08_mirror": _m08,
    "m01_half_bound": _m01,
    "m02_pointer_stuck": _m02,
    "m10_print_bool": _m10,
    "d01_else_return": _d01,
    "d01_else_flag_reset": _d01,
    "d02_low_mid": _d02,
    "d02_high_mid": _d02,
    "d03_drop_temp": _d03,
    "d03_rotate_no_save": _d03,
    "d04_drop_outer": _d04,
    "d04_outer_once": _d04,
    "d05_drop_base": _d05,
    "d05_unreachable_base": _d05,
    "d06_same_arg": _d06,
    "d06_grow_arg": _d06,
    "d07_discard": _d07,
    "d08_str_literal": _d08,
    "d08_array_eq": _d08,
}
