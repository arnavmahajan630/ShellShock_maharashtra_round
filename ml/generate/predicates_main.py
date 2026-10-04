"""Structural predicates for main-game operators (ml_plan/03 §3.5.4, package C1).

``holds(op_id, code)`` is true when the mutant's syntax tree shows that operator's
class. Trace evidence (a failing test, an out-of-bounds read, a step cap) is left
to package C3. ``oth_*`` is always false: OTHER is whatever is left after every
class predicate misses.
"""
from __future__ import annotations

from pycparser import c_ast

from ml.generate.ops_main import (
    _Tree,
    _direct,
    _each_loop,
    _expr_kind,
    _float_names,
    _flows_to_float,
    _functions,
    _in_if_body,
    _in_loop_body,
    _int_const,
    _is_float_cast,
    _is_id,
    _is_void_fn,
    _mods,
    _params,
    _printf_shapes,
    _walk,
    parse_ast,
    read_before_assign,
)


def holds(op_id: str, code: str) -> bool:
    """Static structural predicate from 03 §3.5.4 (AST of the mutant, not a regex on the original text).

    Trace evidence is C3's job; here only the structure.
    M07: EmptyStatement body at the edited site.
    M05: a variable is read before assignment.
    M06: Assignment as an if/while condition.
    AMB: the loop bound/start changed AND the body indexes an array with that variable.
    OTHER: do not implement a class predicate that would make holds() true for a real class;
    holds() for oth_* is False.
    """
    if op_id.startswith("oth_"):
        return False
    pred = _PREDS.get(op_id)
    if pred is None:
        return False
    try:
        ast = parse_ast(code)
    except Exception:
        return False
    return bool(pred(ast))


def _loops(ast):
    for _, _, loops in _each_loop(ast):
        yield from loops


def _m01_le(ast):
    # ``i < n`` became ``i <= n`` on a 0-based count. ``i = 1; i <= n`` is a correct count.
    return any(
        lp.kind == "for" and not lp.indexes and lp.init_form == "0" and lp.has("<=", "n")
        for lp in _loops(ast)
    )


def _m01_nminus1(ast):
    return any(lp.has("<", "n_minus_1") for lp in _loops(ast))


def _m01_start1(ast):
    # Init became 1 and the bound is still strict ``<``. ``i = 1; i <= n`` is a correct count.
    return any(
        lp.init_form == "1" and not lp.indexes and any(norm == "<" for _, norm, _, _, _ in lp.rels)
        for lp in _loops(ast)
    )


def _m01_countdown(ast):
    for lp in _loops(ast):
        for _, norm, _, bound, form in lp.rels:
            if norm == ">=" and form == "const" and _int_const(bound) == 0:
                return True
    return False


def _m01_while_le(ast):
    return any(
        lp.kind == "while" and not lp.indexes and lp.init_form == "0" and lp.has("<=", "n")
        for lp in _loops(ast)
    )


def _m08_index_n(ast):
    for fn, _, loops in _each_loop(ast):
        scalars, _, _ = _params(fn)
        loop_vars = {lp.var for lp in loops if lp.var}
        for node in _walk(fn.body):
            if isinstance(node, c_ast.ArrayRef) and _is_id(node.subscript):
                if node.subscript.name in scalars and node.subscript.name not in loop_vars:
                    return True
    return False


def _m08_onebased(ast):
    return any(lp.indexes and lp.init_form == "1" and lp.has("<=", "n") for lp in _loops(ast))


def _plus_one(node):
    if not isinstance(node, c_ast.BinaryOp) or node.op != "+":
        return False
    return ((_is_id(node.left) and _int_const(node.right) == 1)
            or (_is_id(node.right) and _int_const(node.left) == 1))


def _m08_iplus1(ast):
    for fn in _functions(ast):
        for node in _walk(fn.body):
            if isinstance(node, c_ast.ArrayRef) and _plus_one(node.subscript):
                return True
    return False


def _m08_first1(ast):
    for fn in _functions(ast):
        for node in _walk(fn.body):
            if isinstance(node, c_ast.ArrayRef) and _int_const(node.subscript) == 1:
                return True
    return False


def _amb_le(ast):
    # Bound became ``<= n`` and the body still indexes with the loop variable.
    # Init stays off 1, which is the one-based loop (m08_onebased) instead.
    return any(lp.indexes and lp.init_form != "1" and lp.has("<=", "n") for lp in _loops(ast))


def _amb_start1(ast):
    # Start became 1, the bound is still strict, and the body indexes with that variable.
    return any(
        lp.indexes and lp.init_form == "1" and any(norm == "<" for _, norm, _, _, _ in lp.rels)
        for lp in _loops(ast)
    )


def _m02_drop(ast):
    return any(lp.var and lp.rels and not lp.cond_mods for lp in _loops(ast))


def _m02_reverse(ast):
    for lp in _loops(ast):
        if not any(norm in ("<", "<=") for _, norm, _, _, _ in lp.rels):
            continue
        if any(name == lp.var and direction < 0 for name, direction, _ in lp.cond_mods):
            return True
    return False


def _m02_in_branch(ast):
    for lp in _loops(ast):
        if lp.kind != "while" or not lp.var or not lp.cond_mods:
            continue
        if all(_in_if_body(lp.tree, node, lp.node) for _, _, node in lp.cond_mods):
            return True
    return False


def _m02_wrong_var(ast):
    for lp in _loops(ast):
        if lp.kind != "while" or not lp.var or lp.cond_mods:
            continue
        for name, _, node in _mods(lp.node):
            if name != lp.var and isinstance(node, c_ast.UnaryOp) and "++" in node.op:
                return True
    return False


def _m03_decl(ast):
    for _, tree, loops in _each_loop(ast):
        for lp in loops:
            for node in _walk(lp.body):
                if not isinstance(node, c_ast.Decl) or not node.name or node.name == lp.var:
                    continue
                if _int_const(node.init) is None or not _in_loop_body(tree, node):
                    continue
                if any(name == node.name for name, _, _ in _mods(lp.body)):
                    return True
    return False


def _m03_assign(ast):
    for _, _, loops in _each_loop(ast):
        for lp in loops:
            mods = _mods(lp.body)
            for stmt in _direct(lp.body):
                if not isinstance(stmt, c_ast.Assignment) or stmt.op != "=" or not _is_id(stmt.lvalue):
                    continue
                if _int_const(stmt.rvalue) is None or stmt.lvalue.name == lp.var:
                    continue
                name = stmt.lvalue.name
                if any(mod_name == name and mod_node is not stmt for mod_name, _, mod_node in mods):
                    return True
    return False


def _m04_drop(ast):
    tree = _Tree(ast)
    for fn in _functions(ast):
        floats = _float_names(fn)
        for node in _walk(fn.body):
            if not isinstance(node, c_ast.BinaryOp) or node.op != "/":
                continue
            if _expr_kind(node.left, floats) != "int" or _expr_kind(node.right, floats) != "int":
                continue
            parent = tree.parent.get(id(node))
            if isinstance(parent, c_ast.Cast) and _is_float_cast(parent):
                continue
            if _flows_to_float(tree, node, fn, floats):
                return True
    return False


def _m04_cast_late(ast):
    for fn in _functions(ast):
        for node in _walk(fn.body):
            if _is_float_cast(node) and isinstance(node.expr, c_ast.BinaryOp) and node.expr.op == "/":
                return True
    return False


def _m04_int_literal(ast):
    tree = _Tree(ast)
    for fn in _functions(ast):
        floats = _float_names(fn)
        for node in _walk(fn.body):
            if not isinstance(node, c_ast.BinaryOp) or node.op not in ("/", "*"):
                continue
            if _int_const(node.left) is None and _int_const(node.right) is None:
                continue
            if _expr_kind(node.left, floats) != "int" or _expr_kind(node.right, floats) != "int":
                continue
            if _flows_to_float(tree, node, fn, floats):
                return True
    return False


def _m05(ast):
    return any(read_before_assign(fn) for fn in _functions(ast))


def _has_assign(cond):
    if isinstance(cond, c_ast.Assignment) and cond.op == "=":
        return True
    if isinstance(cond, c_ast.BinaryOp) and cond.op in ("&&", "||"):
        return _has_assign(cond.left) or _has_assign(cond.right)
    if isinstance(cond, c_ast.UnaryOp) and cond.op == "!":
        return _has_assign(cond.expr)
    return False


def _m06_if(ast):
    for fn in _functions(ast):
        for node in _walk(fn.body):
            if isinstance(node, c_ast.If) and _has_assign(node.cond):
                return True
    return False


def _m06_while(ast):
    for fn in _functions(ast):
        for node in _walk(fn.body):
            if isinstance(node, (c_ast.While, c_ast.DoWhile)) and _has_assign(node.cond):
                return True
    return False


def _empty_body(kinds, attr):
    def pred(ast):
        for fn in _functions(ast):
            for node in _walk(fn.body):
                if isinstance(node, kinds) and isinstance(getattr(node, attr), c_ast.EmptyStatement):
                    return True
        return False
    return pred


def _m10(paired_ok):
    def pred(ast):
        for fn in _functions(ast):
            if _is_void_fn(fn):
                continue
            paired, bare = _printf_shapes(fn)
            if paired_ok and paired:
                return True
            if not paired_ok and bare:
                return True
        return False
    return pred


def _u1(ast):
    for fn in _functions(ast):
        for node in _walk(fn.body):
            if (isinstance(node, c_ast.BinaryOp) and node.op == "<"
                    and isinstance(node.left, c_ast.BinaryOp) and node.left.op == "<"):
                return True
    return False


def _u2(ast):
    for fn in _functions(ast):
        for node in _walk(fn.body):
            if not isinstance(node, c_ast.BinaryOp) or node.op != "||":
                continue
            sides = (node.left, node.right)
            if any(isinstance(side, c_ast.Constant) for side in sides) and any(
                    isinstance(side, c_ast.BinaryOp) and side.op == "==" for side in sides):
                return True
    return False


_PREDS = {
    "m01_le": _m01_le,
    "m01_nminus1": _m01_nminus1,
    "m01_start1_count": _m01_start1,
    "m01_countdown_ge0": _m01_countdown,
    "m01_while_le": _m01_while_le,
    "m08_index_n": _m08_index_n,
    "m08_onebased": _m08_onebased,
    "m08_iplus1": _m08_iplus1,
    "m08_first1": _m08_first1,
    "amb_le_array": _amb_le,
    "amb_start1_array": _amb_start1,
    "m02_drop_update": _m02_drop,
    "m02_reverse": _m02_reverse,
    "m02_update_in_branch": _m02_in_branch,
    "m02_wrong_var": _m02_wrong_var,
    "m03_decl_in_loop": _m03_decl,
    "m03_assign_in_loop": _m03_assign,
    "m04_drop_cast": _m04_drop,
    "m04_int_literal": _m04_int_literal,
    "m04_cast_late": _m04_cast_late,
    "m05_drop_init_acc": _m05,
    "m05_drop_init_counter": _m05,
    "m05_drop_init_max": _m05,
    "m06_if_assign": _m06_if,
    "m06_while_assign": _m06_while,
    "m07_if_semi": _empty_body(c_ast.If, "iftrue"),
    "m07_for_semi": _empty_body(c_ast.For, "stmt"),
    "m07_while_semi": _empty_body((c_ast.While, c_ast.DoWhile), "stmt"),
    "m10_printf_noreturn": _m10(False),
    "m10_printf_return0": _m10(True),
    "u1_chained": _u1,
    "u2_or_chain": _u2,
}
