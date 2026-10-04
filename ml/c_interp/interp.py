"""Evaluator for the frozen C subset (ml_plan/03 §2.1, §2.2, §2.5).

The pycparser tree is walked once per source file. Each node becomes a Python closure
(one `_x_<Node>` / `_s_<Node>` method per node type builds it), and running a test calls
those closures. Types are static in C, so int and float paths are chosen while walking.

A running function has a frame: a Python list with one slot per parameter and local.
The run state (step counter, recorded steps, events, counters) lives on the Program and
is reset before every test; a lock keeps two threads from running the same Program at once.

Signals between statements are small ints, not exceptions: 0/None = carry on, BRK, CNT, RET.
`Halt` is the only exception: step cap, depth cap and division by zero.
"""
import operator
import re
import sys
import threading
from operator import itemgetter

from pycparser import c_ast, c_parser

from ml.contracts.subset import DEPTH_CAP, GARBAGE, MAX_RECORDED_STEPS, STEP_CAP

from .preprocess import SourceError, brace_blocks, preprocess
from .values import (
    FGARBAGE, INT_MAX, INT_MIN, NOTYET, UNINIT, Arr, FormatError, fmt_num, format_arg, json_num,
    parse_char, parse_float, parse_format, parse_int, parse_string, wrap32,
)

BRK, CNT, RET = 1, 2, 3
STRICT = False      # tests set this: an interpreter bug then raises instead of ending the run as runtime_error

# 100 nested C calls need a few thousand Python frames. Python-to-Python calls do not use
# the C stack on Python 3.11+, so a higher limit is safe. Set once, never lowered.
_PY_FRAMES = 20000
if sys.getrecursionlimit() < _PY_FRAMES:
    sys.setrecursionlimit(_PY_FRAMES)

EFFECT_NAMES = ["fire", "launch", "door_open", "door_closed", "scan",
                "read_cell", "read_void", "write_cell", "compare", "call", "ret"]
WORLD_CALLS = {"fire": ("fire", 0), "launch": ("launch", 0), "open_door": ("door_open", 0),
               "close_door": ("door_closed", 0), "scan": ("scan", 1)}

_INT_WORDS = {"int", "long", "short", "signed", "unsigned", "char", "_Bool"}
_CMP = {"<": operator.lt, "<=": operator.le, ">": operator.gt, ">=": operator.ge,
        "==": operator.eq, "!=": operator.ne}
_ARITH = {"+", "-", "*", "/", "%"}
_BITS = {"&", "|", "^", "<<", ">>"}

# Library calls that are outside the subset: name -> key of subset.REJECTED.
_REJECTED_CALLS = {name: "malloc" for name in ("malloc", "calloc", "realloc", "free")}
_REJECTED_CALLS.update({name: "scanf" for name in ("scanf", "sscanf", "fscanf", "gets", "fgets", "getchar")})
_REJECTED_CALLS.update({name: "string_h" for name in (
    "strcmp", "strncmp", "strcpy", "strncpy", "strcat", "strncat", "strchr", "strrchr", "strstr",
    "strtok", "strdup", "memset", "memcpy", "memmove", "memcmp")})


class Halt(Exception):
    """Stops the running test. status: "timeout" or "runtime_error"."""

    def __init__(self, status, line):
        super().__init__(status)
        self.status = status
        self.line = line


class _Var:
    __slots__ = ("name", "slot", "type", "kind", "fn", "size")

    def __init__(self, name, slot, ctype, kind, fn):
        self.name, self.slot, self.type, self.kind, self.fn = name, slot, ctype, kind, fn
        self.size = None            # arrays whose size is known before the run (for sizeof)


class _Fn:
    __slots__ = ("name", "rtype", "params", "node", "line", "endline", "blank", "disp", "run",
                 "dups", "order", "decls", "nslots")

    def __init__(self, name, rtype, node, line):
        self.name, self.rtype, self.node, self.line = name, rtype, node, line
        self.params = []
        self.endline = line
        self.blank = []         # a fresh frame, copied on every call
        self.disp = []          # (name, slot, current_slot): what goes into steps[].vars
        self.dups = {}          # name declared more than once -> slot holding the live slot number
        self.order = []         # scalar names in declaration order
        self.decls = []         # scalar locals in declaration order
        self.nslots = 0
        self.run = None


class _Scope:
    __slots__ = ("vars", "restore")

    def __init__(self):
        self.vars = {}
        self.restore = []       # (current_slot, outer_slot) to put back when the scope ends


def _line(node):
    return node.coord.line if node is not None and node.coord is not None else 0


def _is_arr(ctype):
    return ctype.startswith("arr:")


def _is_string(node):
    return isinstance(node, c_ast.Constant) and node.type == "string"


def _mentions(node, name):
    """Does the expression read a variable called `name`?"""
    if isinstance(node, c_ast.ID):
        return node.name == name
    return any(_mentions(child, name) for _, child in node.children())


def _parse_error(exc, text, parser):
    message = str(exc)
    m = re.match(r"^[^:]*:(\d+):(\d+):\s*(.*)$", message, re.S)
    if not m:
        # pycparser gave no position: use the token it stopped at, else the last line
        reason = message.strip(" :").lower() or "syntax error"
        line = max(1, len(text.rstrip().split("\n")))
        try:
            token = parser._peek()
            if token is not None:
                line = token.lineno
                reason += f" near `{token.value}`"
        except Exception:
            pass
        return SourceError("parse_error", line, reason)
    line, reason = int(m.group(1)), m.group(3).strip()
    if reason.startswith("before:"):
        reason = "unexpected `" + reason[len("before:"):].strip() + "`"
    return SourceError("parse_error", line, reason)


def _find_rejected(ast):
    """First construct outside the subset (03 §2.1 'Rejected'), or None."""
    found = []

    def note(node, key):
        found.append((_line(node), len(found), key))

    def walk(node):
        kind = node.__class__.__name__
        if kind == "PtrDecl":
            note(node, "pointer")
        elif kind == "UnaryOp" and node.op == "&":
            note(node, "address_of")
        elif kind == "UnaryOp" and node.op == "*":
            note(node, "pointer")
        elif kind in ("Struct", "Union", "StructRef"):
            note(node, "struct")
        elif kind in ("Goto", "Label"):
            note(node, "goto")
        elif kind in ("Switch", "Case", "Default"):
            note(node, "switch")
        elif kind == "ArrayDecl" and isinstance(node.type, c_ast.ArrayDecl):
            note(node, "multi_dim_array")
        elif kind == "ArrayRef" and isinstance(node.name, c_ast.ArrayRef):
            note(node, "multi_dim_array")
        elif kind == "FuncCall" and isinstance(node.name, c_ast.ID) and node.name.name in _REJECTED_CALLS:
            note(node, _REJECTED_CALLS[node.name.name])
        elif kind in ("Typedef", "Enum", "CompoundLiteral", "NamedInitializer"):
            note(node, kind.lower())
        for _, child in node.children():
            walk(child)

    walk(ast)
    if not found:
        return None
    line, _, key = min(found)
    return SourceError("unsupported", line, f"`{key}` is outside the supported C subset", key)


class Program:
    """A parsed and prepared learner file. `run_test` runs one call of one function."""

    def __init__(self, code, forbid=()):
        self.lock = threading.Lock()
        self.forbid = set(forbid or ())
        text = preprocess(code)
        parser = c_parser.CParser()
        try:
            ast = parser.parse(text)
        except RecursionError:
            raise SourceError("parse_error", 1, "expression nested too deeply") from None
        except Exception as exc:        # ParseError, and the AssertionError pycparser raises on a stray `}`
            raise _parse_error(exc, text, parser) from None
        rejected = _find_rejected(ast)
        if rejected:
            raise rejected

        # ---- run state, reset by run_test
        self.n = 0                  # executed statements in this test
        self.rec = False            # recording steps?
        self.steps = []
        self.eff = []               # effects waiting for the next recorded step
        self.ev = []                # events waiting for the next recorded step
        self.events = []            # every event of this test, each with "test"
        self.test = 0
        self.depth = 0
        self.maxdepth = 0
        self.retval = None
        self.frame = None
        self.disp = ()
        self.out = []
        self.truncated = False
        self.max_steps = MAX_RECORDED_STEPS
        self.cnt = dict.fromkeys(EFFECT_NAMES, 0)
        self.L = []                 # iterations per loop statement
        self.BT = []                # times each `if` was true
        self.BF = []                # times each `if` was false
        self.G = []                 # global variables

        # ---- compile state
        self.loop_lines = []
        self.branch_lines = []
        self.fns = {}
        self.fn = None
        self.scopes = [_Scope()]
        self.loop_depth = 0
        self.kval = {}              # closure of a constant -> its value
        self.global_inits = []
        self._build(ast, text)
        self.global_blank = [0] * len(self.G)

    # ================================================================ building

    def _build(self, ast, text):
        defs = [node for node in ast.ext if isinstance(node, c_ast.FuncDef)]
        blocks = brace_blocks(text)
        for node in defs:
            fn = self._signature(node)
            if fn.name in self.fns:
                raise SourceError("parse_error", fn.line, f"function `{fn.name}` is defined twice")
            start = _line(node.body)
            for k, (open_line, close_line) in enumerate(blocks):
                if open_line == start:
                    fn.endline = close_line
                    del blocks[k]
                    break
            self.fns[fn.name] = fn
        gfn = _Fn("<globals>", "void", None, 0)
        for node in ast.ext:
            if isinstance(node, c_ast.Decl) and not isinstance(node.type, c_ast.FuncDecl):
                self.fn = gfn
                action = self._decl_action(node, is_global=True)
                self.fn = None
                if action:
                    self.global_inits.append(action)
            elif not isinstance(node, (c_ast.FuncDef, c_ast.Decl, c_ast.Pragma)):
                raise SourceError("unsupported", _line(node), "only functions and variables may be at file level",
                                  node.__class__.__name__.lower())
        self.G.extend([0] * gfn.nslots)
        for node in defs:
            self._function(self.fns[node.decl.name])

    def _type_names(self, node, what):
        """IdentifierType names under a TypeDecl; anything else is outside the subset."""
        if isinstance(node, c_ast.TypeDecl) and isinstance(node.type, c_ast.IdentifierType):
            return node.type.names
        raise SourceError("unsupported", _line(node), f"this kind of {what} is not supported", "pointer")

    def _scalar_type(self, names, line, allow_void=False):
        words = set(names)
        if "void" in words:
            if allow_void:
                return "void"
            raise SourceError("parse_error", line, "a variable cannot be `void`")
        if "float" in words or "double" in words:
            return "float"
        if words <= _INT_WORDS:
            return "int"
        raise SourceError("unsupported", line, f"type `{' '.join(names)}` is not supported", " ".join(names))

    def _signature(self, node):
        decl = node.decl
        line = _line(decl)
        ftype = decl.type
        rtype = self._scalar_type(self._type_names(ftype.type, "return type"), line, allow_void=True)
        fn = _Fn(decl.name, rtype, node, line)
        params = ftype.args.params if ftype.args else []
        for p in params:
            if isinstance(p, c_ast.EllipsisParam):
                raise SourceError("unsupported", line, "variadic functions are not supported", "...")
            if isinstance(p, c_ast.Typename):
                names = self._type_names(p.type, "parameter")
                if "void" in names and len(params) == 1:
                    break
                raise SourceError("parse_error", line, "parameter needs a name")
            if isinstance(p, c_ast.ID):
                raise SourceError("parse_error", line, f"parameter `{p.name}` needs a type")
            if isinstance(p.type, c_ast.ArrayDecl):
                names = self._type_names(p.type.type, "array parameter")
                elem = "char" if "char" in names else self._scalar_type(names, line)
                ctype = "arr:" + elem
            else:
                ctype = self._scalar_type(self._type_names(p.type, "parameter"), line)
            fn.params.append(_Var(p.name, len(fn.params), ctype, "param", fn))
        return fn

    def _function(self, fn):
        self.fn = fn
        scope = _Scope()
        self.scopes.append(scope)
        fn.nslots = len(fn.params)
        # names declared more than once in this function need a "which one is live" slot
        counts = {}
        for var in fn.params:
            if not _is_arr(var.type):
                counts[var.name] = counts.get(var.name, 0) + 1

        def count(node):
            if isinstance(node, c_ast.Decl) and isinstance(node.type, c_ast.TypeDecl):
                counts[node.name] = counts.get(node.name, 0) + 1
            for _, child in node.children():
                count(child)

        count(fn.node.body)
        for name, times in counts.items():
            if times > 1:
                fn.dups[name] = fn.nslots
                fn.nslots += 1
        for var in fn.params:
            if var.name in scope.vars:
                raise SourceError("parse_error", fn.line, f"parameter `{var.name}` is declared twice")
            scope.vars[var.name] = var
            if not _is_arr(var.type):
                fn.order.append(var.name)
        body = self._block_items(fn.node.body.block_items or [])
        self.scopes.pop()
        self.fn = None

        blank = [UNINIT] * fn.nslots
        for cur in fn.dups.values():
            blank[cur] = NOTYET                 # no variable of that name is live yet
        for var in fn.params:
            if var.name in fn.dups:
                blank[fn.dups[var.name]] = var.slot
        fn.blank = blank
        first = {}
        for var in fn.decls:
            first.setdefault(var.name, var.slot)
        for var in fn.params:
            first.setdefault(var.name, var.slot)
        fn.disp[:] = [(name, first[name], fn.dups.get(name)) for name in fn.order]

        M = self
        cnt = self.cnt
        name, rtype, endline = fn.name, fn.rtype, fn.endline
        garbage = FGARBAGE if rtype == "float" else GARBAGE

        def run(f):
            if body(f) == RET:
                return
            # fell off the end of the function: an implicit return at the closing brace
            if rtype == "void":
                M.retval = None
            else:
                M.event("missing_return", endline)
                M.retval = garbage
            cnt["ret"] += 1
            if M.rec:
                M.eff.append("ret:%s:%d:%s" % (name, M.depth, fmt_num(M.retval)))
                M.record(endline)

        fn.run = run

    # ---------------------------------------------------------------- scopes and variables

    def _lookup(self, name, line):
        for scope in reversed(self.scopes):
            var = scope.vars.get(name)
            if var is not None:
                return var
        if name in self.fns or name in WORLD_CALLS or name in ("printf", "strlen"):
            raise SourceError("unsupported", line, f"`{name}` is a function, not a value", "pointer")
        raise SourceError("parse_error", line, f"`{name}` is not declared")

    def _declare(self, name, ctype, line, is_global=False):
        fn = self.fn
        scope = self.scopes[-1]
        if name in scope.vars:
            # Real C rejects this too: a parameter and the function's own top-level block
            # share one scope, so redeclaring the parameter's name there is illegal — same
            # error GCC gives ("redeclaration of 'n' with no linkage"). Letting it through
            # silently clobbered the parameter's value with the new local's, which produced
            # a confusing, hard-to-classify bug (every test behaving as if the argument were
            # ignored) instead of a clear rejection.
            raise SourceError("parse_error", line, f"`{name}` is declared twice")
        if is_global:
            var = _Var(name, fn.nslots, ctype, "global", None)
            fn.nslots += 1
            scope.vars[name] = var
            return var
        var = _Var(name, fn.nslots, ctype, "local", fn)
        fn.nslots += 1
        if not _is_arr(ctype):
            if name in fn.dups:
                outer = None
                for outer_scope in reversed(self.scopes):
                    found = outer_scope.vars.get(name)
                    if found is not None:
                        outer = found if found.fn is fn and not _is_arr(found.type) else None
                        break
                if outer is not None:
                    scope.restore.append((fn.dups[name], outer.slot))
            if name not in fn.order:
                fn.order.append(name)
            fn.decls.append(var)
        scope.vars[name] = var
        return var

    def _close_scope(self, scope, closure):
        """Wrap a block so that names it shadowed show their outer value again afterwards."""
        if not scope.restore:
            return closure
        restore = tuple(scope.restore)

        def scoped(f):
            r = closure(f)
            for cur, slot in restore:
                f[cur] = slot
            return r

        return scoped

    # ================================================================ run-time helpers

    def event(self, etype, line, **fields):
        e = {"type": etype, "line": line}
        e.update(fields)
        if self.rec:
            self.ev.append(e)
            e = dict(e)
        e["test"] = self.test
        self.events.append(e)

    def record(self, line):
        """Close the current step: the statement on `line` has finished."""
        steps = self.steps
        if len(steps) < self.max_steps:
            f = self.frame
            shown = {}
            for name, slot, cur in self.disp:
                if cur is not None:
                    live = f[cur]
                    v = None if live is NOTYET else f[live]
                else:
                    v = f[slot]
                if v is UNINIT or v is NOTYET or (v.__class__ is float and (v != v or v - v != 0)):
                    v = None
                shown[name] = v
            steps.append({"i": len(steps), "line": line, "vars": shown, "events": self.ev, "effects": self.eff})
        else:
            self.truncated = True
        self.ev = []
        self.eff = []

    def cap(self, line):
        self.event("step_cap_hit", line)
        raise Halt("timeout", line)

    def overflow(self, value, line):
        self.event("overflow", line)
        return wrap32(value)

    def uninit(self, name, line, is_float):
        self.event("uninit_read", line, var=name)
        return FGARBAGE if is_float else GARBAGE

    def to_int(self, value, line):
        """float -> int as C does it (toward zero); out of range wraps and is reported."""
        try:
            i = int(value)
        except (OverflowError, ValueError):
            self.event("overflow", line)
            return INT_MIN
        if INT_MIN <= i <= INT_MAX:
            return i
        self.event("overflow", line)
        return wrap32(i)

    def aread(self, arr, i, name, line):
        data = arr.data
        if 0 <= i < len(data):
            v = data[i]
            if arr.world:
                self.cnt["read_cell"] += 1
                if self.rec:
                    self.eff.append("read_cell:%d" % i)
            if v is UNINIT:
                self.event("uninit_read", line, var=name, idx=i)
                return FGARBAGE if arr.elem == "float" else GARBAGE
            return v
        self.event("oob_read", line, arr=name, idx=i, size=len(data))
        if arr.world:
            self.cnt["read_void"] += 1
            if self.rec:
                self.eff.append("read_void:%d" % i)
        return FGARBAGE if arr.elem == "float" else GARBAGE

    def awrite(self, arr, i, v, name, line):
        data = arr.data
        if 0 <= i < len(data):
            data[i] = v
            if arr.world:
                self.cnt["write_cell"] += 1
                if self.rec:
                    self.eff.append("write_cell:%d:%s" % (i, fmt_num(v)))
        else:
            self.event("oob_write", line, arr=name, idx=i, size=len(data))

    def invoke(self, fn, args, line):
        d = self.depth + 1
        if d > DEPTH_CAP:
            shown = [a.name if a.__class__ is Arr else json_num(a) for a in args]
            self.event("depth_cap_hit", line, fn=fn.name, last_args=shown)
            raise Halt("timeout", line)
        nf = fn.blank[:]
        nf[:len(args)] = args
        self.depth = d
        if d > self.maxdepth:
            self.maxdepth = d
        self.cnt["call"] += 1
        if self.rec and self.eff:
            # the calling statement has made effects already (its own `call`, a cell read):
            # record them now, so effects stay in the order they happened
            self.record(line)
        frame, disp = self.frame, self.disp
        self.frame, self.disp = nf, fn.disp
        if self.rec:
            eff, ev = self.eff, self.ev
            shown = ",".join(p.name if _is_arr(p.type) else fmt_num(a) for p, a in zip(fn.params, args))
            self.eff = ["call:%s:%d:%s" % (fn.name, d, shown)]
            self.ev = []
            fn.run(nf)
            self.eff, self.ev = eff, ev
        else:
            fn.run(nf)
        self.frame, self.disp = frame, disp
        self.depth = d - 1
        return self.retval

    # ================================================================ expressions
    # Each _x_ method returns (closure, ctype). `ctx` is true when the value is about to be
    # converted to float: an int division compiled under it reports into_float.

    def typeof(self, node):
        kind = node.__class__
        if kind is c_ast.Constant:
            if node.type == "string":
                return "arr:char"
            return "float" if ("float" in node.type or "double" in node.type) else "int"
        if kind is c_ast.ID:
            return self._lookup(node.name, _line(node)).type
        if kind is c_ast.ArrayRef:
            base = self.typeof(node.name)
            if not _is_arr(base):
                raise SourceError("parse_error", _line(node), "only arrays can be indexed")
            return "float" if base == "arr:float" else "int"
        if kind is c_ast.BinaryOp:
            if node.op in _CMP or node.op in ("&&", "||"):
                return "int"
            left, right = self.typeof(node.left), self.typeof(node.right)
            return "float" if "float" in (left, right) else "int"
        if kind is c_ast.UnaryOp:
            if node.op in ("!", "sizeof", "~"):
                return "int"
            return self.typeof(node.expr)
        if kind is c_ast.Cast:
            return self._scalar_type(self._type_names(node.to_type.type, "cast"), _line(node))
        if kind is c_ast.TernaryOp:
            a, b = self.typeof(node.iftrue), self.typeof(node.iffalse)
            return "float" if "float" in (a, b) else a
        if kind is c_ast.Assignment:
            return self.typeof(node.lvalue)
        if kind is c_ast.FuncCall:
            name = node.name.name if isinstance(node.name, c_ast.ID) else None
            if name in self.fns:
                return self.fns[name].rtype
            return "int"
        if kind is c_ast.ExprList:
            return self.typeof(node.exprs[-1])
        raise SourceError("unsupported", _line(node), "this expression is not supported",
                          node.__class__.__name__.lower())

    def expr(self, node, ctx=False):
        method = getattr(self, "_x_" + node.__class__.__name__, None)
        if method is None:
            raise SourceError("unsupported", _line(node), "this expression is not supported",
                              node.__class__.__name__.lower())
        return method(node, ctx)

    def scalar(self, node, ctx=False):
        """Compile an expression that must be an int or a float; return (closure, ctype)."""
        fn, ctype = self.expr(node, ctx)
        if ctype == "void":
            raise SourceError("parse_error", _line(node), "a `void` function gives no value")
        if _is_arr(ctype):
            raise SourceError("unsupported", _line(node), "an array cannot be used as a number", "pointer")
        return fn, ctype

    def conv(self, fn, src, dst, line):
        """Closure that gives `fn`'s value as ctype `dst`."""
        if src == dst:
            return fn
        if src == "void" or _is_arr(src) or _is_arr(dst):
            raise SourceError("parse_error", line, "the types here do not fit")
        if dst == "float":
            if fn in self.kval:
                return self._const(float(self.kval[fn]))
            return lambda f: float(fn(f))
        to_int = self.to_int
        return lambda f: to_int(fn(f), line)

    def _convv(self, src, dst, line):
        """Value-level version of conv, or None when nothing is to be done."""
        if src == dst:
            return None
        if dst == "float":
            return float
        to_int = self.to_int
        return lambda v: to_int(v, line)

    def _const(self, value):
        def k(f):
            return value
        self.kval[k] = value
        return k

    def _x_Constant(self, node, ctx):
        kind = node.type
        if kind == "string":
            codes = parse_string(node.value) + [0]
            label = node.value
            return (lambda f: Arr(codes[:], label, "char")), "arr:char"
        if kind == "char":
            return self._const(parse_char(node.value)), "int"
        if "float" in kind or "double" in kind:
            return self._const(parse_float(node.value)), "float"
        try:
            return self._const(parse_int(node.value)), "int"
        except ValueError:
            raise SourceError("parse_error", _line(node), f"bad number `{node.value}`") from None

    def _x_ID(self, node, ctx):
        line = _line(node)
        var = self._lookup(node.name, line)
        slot = var.slot
        if var.kind == "global":
            G = self.G
            return (lambda f: G[slot]), var.type
        if var.kind == "param" or _is_arr(var.type):
            return itemgetter(slot), var.type
        name, is_float, uninit = var.name, var.type == "float", self.uninit

        def load(f):
            v = f[slot]
            if v is UNINIT:
                return uninit(name, line, is_float)
            return v

        return load, var.type

    def _array_parts(self, node):
        """(array closure, index closure, name for events, element ctype) of an ArrayRef."""
        line = _line(node)
        base, btype = self.expr(node.name)
        if not _is_arr(btype):
            raise SourceError("parse_error", line, "only arrays can be indexed")
        index, itype = self.scalar(node.subscript)
        if itype != "int":
            raise SourceError("parse_error", line, "an array index must be an integer")
        name = node.name.name if isinstance(node.name, c_ast.ID) else "<literal>"
        return base, index, name, ("float" if btype == "arr:float" else "int")

    def _x_ArrayRef(self, node, ctx):
        line = _line(node)
        base, index, name, etype = self._array_parts(node)
        aread = self.aread
        return (lambda f: aread(base(f), index(f), name, line)), etype

    def _x_Cast(self, node, ctx):
        line = _line(node)
        dst = self._scalar_type(self._type_names(node.to_type.type, "cast"), line, allow_void=True)
        if dst == "void":
            fn, _ = self.expr(node.expr)
            return fn, "void"
        fn, src = self.scalar(node.expr, dst == "float")
        return self.conv(fn, src, dst, line), dst

    def _x_TernaryOp(self, node, ctx):
        line = _line(node)
        cond = self._cond(node.cond)
        rtype = self.typeof(node)
        sub = ctx or rtype == "float"
        a, at = self.scalar(node.iftrue, sub)
        b, bt = self.scalar(node.iffalse, sub)
        a, b = self.conv(a, at, rtype, line), self.conv(b, bt, rtype, line)
        return (lambda f: a(f) if cond(f) else b(f)), rtype

    def _x_ExprList(self, node, ctx):
        parts = [self.expr(e, ctx if e is node.exprs[-1] else False) for e in node.exprs]
        fns = [p[0] for p in parts[:-1]]
        last = parts[-1][0]

        def run(f):
            for fn in fns:
                fn(f)
            return last(f)

        return run, parts[-1][1]

    def _x_UnaryOp(self, node, ctx):
        op, line = node.op, _line(node)
        if op in ("p++", "p--", "++", "--"):
            return self._incdec(node)
        if op == "sizeof":
            return self._const(self._sizeof(node.expr)), "int"
        fn, ctype = self.scalar(node.expr, ctx if op in ("-", "+") else False)
        if op == "!":
            return (lambda f: 0 if fn(f) else 1), "int"
        if op == "+":
            return fn, ctype
        if op == "-":
            if fn in self.kval:
                return self._const(-self.kval[fn] if ctype == "float" else wrap32(-self.kval[fn])), ctype
            if ctype == "float":
                return (lambda f: -fn(f)), "float"
            overflow = self.overflow

            def neg(f):
                v = -fn(f)
                return v if v <= INT_MAX else overflow(v, line)

            return neg, "int"
        if op == "~" and ctype == "int":
            return (lambda f: ~fn(f)), "int"
        raise SourceError("unsupported", line, f"operator `{op}` is not supported", op)

    def _sizeof(self, node):
        line = _line(node)
        if isinstance(node, c_ast.Typename):
            names = self._type_names(node.type, "sizeof")
            return 8 if "double" in names else 1 if "char" in names else 4
        if isinstance(node, c_ast.ArrayRef) and isinstance(node.name, c_ast.ID):
            return 1 if self._lookup(node.name.name, line).type == "arr:char" else 4
        if isinstance(node, c_ast.ID):
            var = self._lookup(node.name, line)
            if _is_arr(var.type):
                if var.kind == "param":
                    return 8                    # an array parameter is a pointer in C
                if var.size is None:
                    raise SourceError("unsupported", line, "sizeof of this array is not supported", "sizeof")
                return var.size * (1 if var.type == "arr:char" else 4)
        self.typeof(node)
        return 4

    def _incdec(self, node):
        op, line = node.op, _line(node)
        post = op[0] == "p"
        target = node.expr
        ttype = self.typeof(target)
        if _is_arr(ttype) or ttype == "void":
            raise SourceError("parse_error", line, "++ and -- need a number variable")
        is_float = ttype == "float"
        delta = (1.0 if is_float else 1) * (1 if op.endswith("++") else -1)
        overflow, uninit = self.overflow, self.uninit

        if isinstance(target, c_ast.ArrayRef):
            base, index, name, _ = self._array_parts(target)
            aread, awrite = self.aread, self.awrite

            def cell(f):
                arr = base(f)
                i = index(f)
                v = aread(arr, i, name, line)
                n = v + delta
                if not is_float and not INT_MIN <= n <= INT_MAX:
                    n = overflow(n, line)
                awrite(arr, i, n, name, line)
                return v if post else n

            return cell, ttype
        if not isinstance(target, c_ast.ID):
            raise SourceError("parse_error", line, "++ and -- need a variable")
        var = self._lookup(target.name, line)
        slot, name = var.slot, var.name
        if var.kind == "global":
            G = self.G

            def gstep(f):
                v = G[slot]
                n = v + delta
                if not is_float and not INT_MIN <= n <= INT_MAX:
                    n = overflow(n, line)
                G[slot] = n
                return v if post else n

            return gstep, ttype
        if is_float:
            def fstep(f):
                v = f[slot]
                if v is UNINIT:
                    v = uninit(name, line, True)
                n = f[slot] = v + delta
                return v if post else n

            return fstep, ttype
        if post:
            def post_step(f):
                v = f[slot]
                if v is UNINIT:
                    v = uninit(name, line, False)
                n = v + delta
                if n > INT_MAX or n < INT_MIN:
                    n = overflow(n, line)
                f[slot] = n
                return v

            return post_step, ttype

        def pre_step(f):
            v = f[slot]
            if v is UNINIT:
                v = uninit(name, line, False)
            n = v + delta
            if n > INT_MAX or n < INT_MIN:
                n = overflow(n, line)
            f[slot] = n
            return n

        return pre_step, ttype

    # ---------------------------------------------------------------- binary operators

    def _opv(self, op, ltype, rtype, line, ctx):
        """(function of two values, result ctype) for a binary operator on scalars."""
        M = self
        overflow = self.overflow
        if op in _CMP:
            test = _CMP[op]
            return (lambda a, b: 1 if test(a, b) else 0), "int"
        is_float = "float" in (ltype, rtype)
        if is_float:
            if op == "+":
                return operator.add, "float"
            if op == "-":
                return operator.sub, "float"
            if op == "*":
                return operator.mul, "float"
            if op == "/":
                def fdiv(a, b):
                    if b == 0:
                        M.event("div_zero", line)
                        raise Halt("runtime_error", line)
                    return a / b
                return fdiv, "float"
            raise SourceError("parse_error", line, f"`{op}` needs whole numbers")
        if op == "+":
            def add(a, b):
                r = a + b
                return r if INT_MIN <= r <= INT_MAX else overflow(r, line)
            return add, "int"
        if op == "-":
            def sub(a, b):
                r = a - b
                return r if INT_MIN <= r <= INT_MAX else overflow(r, line)
            return sub, "int"
        if op == "*":
            def mul(a, b):
                r = a * b
                return r if INT_MIN <= r <= INT_MAX else overflow(r, line)
            return mul, "int"
        if op == "/":
            def idiv(a, b):
                if b == 0:
                    M.event("div_zero", line)
                    raise Halt("runtime_error", line)
                q = abs(a) // abs(b)
                if (a < 0) != (b < 0):
                    q = -q
                M.event("intdiv", line, remainder_nonzero=a % b != 0, into_float=ctx)
                return q if q <= INT_MAX else overflow(q, line)
            return idiv, "int"
        if op == "%":
            def imod(a, b):
                if b == 0:
                    M.event("div_zero", line)
                    raise Halt("runtime_error", line)
                r = abs(a) % abs(b)
                return -r if a < 0 else r
            return imod, "int"
        if op == "&":
            return operator.and_, "int"
        if op == "|":
            return operator.or_, "int"
        if op == "^":
            return operator.xor, "int"
        if op == "<<":
            return (lambda a, b: wrap32(a << (b & 31))), "int"
        if op == ">>":
            return (lambda a, b: a >> (b & 31)), "int"
        raise SourceError("unsupported", line, f"operator `{op}` is not supported", op)

    def _x_BinaryOp(self, node, ctx):
        op, line = node.op, _line(node)
        if op == "&&":
            a, b = self._truth(node.left), self._truth(node.right)
            return (lambda f: 1 if a(f) and b(f) else 0), "int"
        if op == "||":
            a, b = self._truth(node.left), self._truth(node.right)
            return (lambda f: 1 if a(f) or b(f) else 0), "int"
        ltype, rtype = self.typeof(node.left), self.typeof(node.right)
        if _is_arr(ltype) or _is_arr(rtype):
            return self._address_compare(node, ltype, rtype)
        if "void" in (ltype, rtype):
            raise SourceError("parse_error", line, "a `void` function gives no value")
        if op not in _CMP and op not in _ARITH and op not in _BITS:
            raise SourceError("unsupported", line, f"operator `{op}` is not supported", op)
        is_float = "float" in (ltype, rtype)
        if op in _CMP:
            if isinstance(node.left, c_ast.ArrayRef) and isinstance(node.right, c_ast.ArrayRef):
                return self._cell_compare(node)
            lf = self.expr(node.left, rtype == "float")[0]
            rf = self.expr(node.right, ltype == "float")[0]
            return self._compare(op, lf, rf), "int"
        sub = is_float or ctx
        lf = self.expr(node.left, sub)[0]
        rf = self.expr(node.right, sub)[0]
        if not is_float and op in ("+", "-", "*"):
            return self._int_arith(op, lf, rf, line), "int"
        opv, rtype_out = self._opv(op, ltype, rtype, line, ctx)
        return (lambda f: opv(lf(f), rf(f))), rtype_out

    def _truth(self, node):
        return self.scalar(node)[0]

    def _compare(self, op, lf, rf):
        k = self.kval.get(rf)
        if k is not None:
            if op == "<":
                return lambda f: 1 if lf(f) < k else 0
            if op == "<=":
                return lambda f: 1 if lf(f) <= k else 0
            if op == ">":
                return lambda f: 1 if lf(f) > k else 0
            if op == ">=":
                return lambda f: 1 if lf(f) >= k else 0
            if op == "==":
                return lambda f: 1 if lf(f) == k else 0
            return lambda f: 1 if lf(f) != k else 0
        if op == "<":
            return lambda f: 1 if lf(f) < rf(f) else 0
        if op == "<=":
            return lambda f: 1 if lf(f) <= rf(f) else 0
        if op == ">":
            return lambda f: 1 if lf(f) > rf(f) else 0
        if op == ">=":
            return lambda f: 1 if lf(f) >= rf(f) else 0
        if op == "==":
            return lambda f: 1 if lf(f) == rf(f) else 0
        return lambda f: 1 if lf(f) != rf(f) else 0

    def _int_arith(self, op, lf, rf, line):
        overflow = self.overflow
        k = self.kval.get(rf)
        if k is not None and lf in self.kval:
            a = self.kval[lf]
            folded = a + k if op == "+" else a - k if op == "-" else a * k
            if INT_MIN <= folded <= INT_MAX:        # an overflowing constant is left for run time
                return self._const(folded)
        if op == "+":
            if k is not None:
                def add_k(f):
                    r = lf(f) + k
                    return r if INT_MIN <= r <= INT_MAX else overflow(r, line)
                return add_k

            def add(f):
                r = lf(f) + rf(f)
                return r if INT_MIN <= r <= INT_MAX else overflow(r, line)
            return add
        if op == "-":
            if k is not None:
                def sub_k(f):
                    r = lf(f) - k
                    return r if INT_MIN <= r <= INT_MAX else overflow(r, line)
                return sub_k

            def sub(f):
                r = lf(f) - rf(f)
                return r if INT_MIN <= r <= INT_MAX else overflow(r, line)
            return sub

        def mul(f):
            r = lf(f) * rf(f)
            return r if INT_MIN <= r <= INT_MAX else overflow(r, line)
        return mul

    def _cell_compare(self, node):
        """a[i] < a[j]: a relational compare of two cells; a `compare` effect on a test array."""
        line = _line(node)
        base1, index1, name1, _ = self._array_parts(node.left)
        base2, index2, name2, _ = self._array_parts(node.right)
        test = _CMP[node.op]
        M, cnt, aread = self, self.cnt, self.aread

        def compare(f):
            a1 = base1(f)
            i = index1(f)
            v1 = aread(a1, i, name1, line)
            a2 = base2(f)
            j = index2(f)
            v2 = aread(a2, j, name2, line)
            if a1 is a2 and a1.world:
                cnt["compare"] += 1
                if M.rec:
                    M.eff.append("compare:%d:%d" % (i, j))
            return 1 if test(v1, v2) else 0

        return compare, "int"

    def _address_compare(self, node, ltype, rtype):
        """== / != where a side is a string literal or an array name: C compares addresses."""
        op, line = node.op, _line(node)
        if op not in ("==", "!="):
            raise SourceError("unsupported", line, "arrays cannot be used in arithmetic", "pointer")
        etype = "str_literal_compare" if _is_string(node.left) or _is_string(node.right) else "array_compare"
        left = None if _is_string(node.left) else self.expr(node.left)[0]
        right = None if _is_string(node.right) else self.expr(node.right)[0]
        equal = op == "=="
        M = self

        def compare(f):
            a = left(f) if left else None
            b = right(f) if right else None
            M.event(etype, line)
            same = a is b and a.__class__ is Arr
            return 1 if same == equal else 0

        return compare, "int"

    # ---------------------------------------------------------------- assignment

    def _x_Assignment(self, node, ctx):
        op, line = node.op, _line(node)
        target = node.lvalue
        if op != "=" and op[:-1] not in _ARITH and op[:-1] not in _BITS:
            raise SourceError("unsupported", line, f"operator `{op}` is not supported", op)
        if isinstance(target, c_ast.ArrayRef):
            return self._assign_cell(node)
        if not isinstance(target, c_ast.ID):
            raise SourceError("parse_error", line, "the left side of `=` must be a variable")
        var = self._lookup(target.name, line)
        ttype = var.type
        if _is_arr(ttype):
            raise SourceError("parse_error", line, f"array `{var.name}` cannot be assigned as a whole")
        slot, name = var.slot, var.name
        is_float = ttype == "float"
        uninit, overflow = self.uninit, self.overflow
        if op == "=":
            rf, rtype = self.scalar(node.rvalue, is_float)
            rf = self.conv(rf, rtype, ttype, line)
            if var.kind == "global":
                G = self.G

                def gset(f):
                    v = G[slot] = rf(f)
                    return v
                return gset, ttype

            def lset(f):
                v = f[slot] = rf(f)
                return v
            return lset, ttype

        bop = op[:-1]
        rf, rtype = self.scalar(node.rvalue, is_float)
        if var.kind != "global" and ttype == "int" and rtype == "int" and bop in ("+", "-"):
            if bop == "+":
                def add_to(f):
                    v = f[slot]
                    if v is UNINIT:
                        v = uninit(name, line, False)
                    v += rf(f)
                    if v > INT_MAX or v < INT_MIN:
                        v = overflow(v, line)
                    f[slot] = v
                    return v
                return add_to, ttype

            def sub_from(f):
                v = f[slot]
                if v is UNINIT:
                    v = uninit(name, line, False)
                v -= rf(f)
                if v > INT_MAX or v < INT_MIN:
                    v = overflow(v, line)
                f[slot] = v
                return v
            return sub_from, ttype

        opv, otype = self._opv(bop, ttype, rtype, line, is_float or ctx)
        back = self._convv(otype, ttype, line)
        if var.kind == "global":
            G = self.G

            def gupdate(f):
                v = opv(G[slot], rf(f))
                if back:
                    v = back(v)
                G[slot] = v
                return v
            return gupdate, ttype

        def update(f):
            v = f[slot]
            if v is UNINIT:
                v = uninit(name, line, is_float)
            v = opv(v, rf(f))
            if back:
                v = back(v)
            f[slot] = v
            return v
        return update, ttype

    def _assign_cell(self, node):
        op, line = node.op, _line(node)
        base, index, name, etype = self._array_parts(node.lvalue)
        is_float = etype == "float"
        rf, rtype = self.scalar(node.rvalue, is_float)
        aread, awrite = self.aread, self.awrite
        if op == "=":
            rf = self.conv(rf, rtype, etype, line)

            def store(f):
                arr = base(f)
                i = index(f)
                v = rf(f)
                awrite(arr, i, v, name, line)
                return v
            return store, etype
        opv, otype = self._opv(op[:-1], etype, rtype, line, is_float)
        back = self._convv(otype, etype, line)

        def update(f):
            arr = base(f)
            i = index(f)
            v = opv(aread(arr, i, name, line), rf(f))
            if back:
                v = back(v)
            awrite(arr, i, v, name, line)
            return v
        return update, etype

    # ---------------------------------------------------------------- calls

    def _x_FuncCall(self, node, ctx):
        line = _line(node)
        if not isinstance(node.name, c_ast.ID):
            raise SourceError("unsupported", line, "only plain function calls are supported", "pointer")
        name = node.name.name
        args = list(node.args.exprs) if node.args else []
        if name in self.fns:
            return self._call_user(self.fns[name], args, line)
        if name in WORLD_CALLS:
            return self._call_world(name, args, line)
        if name == "printf":
            return self._call_printf(args, line)
        if name == "strlen":
            return self._call_strlen(args, line)
        if any(name in scope.vars for scope in self.scopes):
            raise SourceError("parse_error", line, f"`{name}` is a variable, not a function")
        raise SourceError("parse_error", line, f"function `{name}` is not defined", name + "()")

    def _call_user(self, fn, args, line):
        if len(args) != len(fn.params):
            raise SourceError("parse_error", line,
                              f"`{fn.name}` takes {len(fn.params)} argument(s), not {len(args)}")
        arg_fns = []
        for param, arg in zip(fn.params, args):
            af, atype = self.expr(arg, param.type == "float")
            if _is_arr(param.type) or _is_arr(atype):
                if param.type != atype:
                    raise SourceError("parse_error", _line(arg),
                                      f"argument for `{param.name}` of `{fn.name}` has the wrong type")
            else:
                if atype == "void":
                    raise SourceError("parse_error", _line(arg), "a `void` function gives no value")
                af = self.conv(af, atype, param.type, line)
            arg_fns.append(af)
        invoke = self.invoke
        if len(arg_fns) == 1:
            only = arg_fns[0]
            return (lambda f: invoke(fn, [only(f)], line)), fn.rtype
        if len(arg_fns) == 2:
            first, second = arg_fns
            return (lambda f: invoke(fn, [first(f), second(f)], line)), fn.rtype
        return (lambda f: invoke(fn, [a(f) for a in arg_fns], line)), fn.rtype

    def _call_world(self, name, args, line):
        effect, arity = WORLD_CALLS[name]
        if len(args) != arity:
            raise SourceError("parse_error", line, f"`{name}` takes {arity} argument(s), not {len(args)}")
        M, cnt = self, self.cnt
        if arity == 0:
            def act(f):
                cnt[effect] += 1
                if M.rec:
                    M.eff.append(effect)
                return 0
            return act, "int"
        af, _ = self.scalar(args[0])

        def act_with(f):
            v = af(f)
            cnt[effect] += 1
            if M.rec:
                M.eff.append("%s:%s" % (effect, fmt_num(v)))
            return 0
        return act_with, "int"

    def _call_printf(self, args, line):
        if not args or not _is_string(args[0]):
            raise SourceError("unsupported", line, "printf needs a string literal as its format", "printf")
        try:
            parts = parse_format(parse_string(args[0].value))
        except FormatError as exc:
            raise SourceError("unsupported", line, f"printf format `{exc}` is not supported", "printf") from None
        arg_fns = []
        for arg in args[1:]:
            af, atype = self.expr(arg)
            if atype == "void":
                raise SourceError("parse_error", _line(arg), "a `void` function gives no value")
            arg_fns.append(af)
        M = self
        if all(part.__class__ is str for part in parts) and not arg_fns:
            text = "".join(parts)

            def put(f):
                M.out.append(text)
                return len(text)
            return put, "int"
        count = len(arg_fns)

        def printf(f):
            values = [af(f) for af in arg_fns]
            out = []
            k = 0
            for part in parts:
                if part.__class__ is str:
                    out.append(part)
                    continue
                if k < count:
                    value = values[k]
                    if part[3] == "s" and value.__class__ is Arr and 0 not in value.data:
                        M.event("oob_read", line, arr=value.name, idx=len(value.data), size=len(value.data))
                    out.append(format_arg(part, value, True))
                else:
                    out.append(format_arg(part, None, False))
                k += 1
            text = "".join(out)
            M.out.append(text)
            return len(text)
        return printf, "int"

    def _call_strlen(self, args, line):
        if "strlen" in self.forbid:
            raise SourceError("unsupported", line, "this problem does not allow `strlen`", "strlen")
        if len(args) != 1:
            raise SourceError("parse_error", line, f"`strlen` takes 1 argument, not {len(args)}")
        af, atype = self.expr(args[0])
        if atype != "arr:char":
            raise SourceError("parse_error", _line(args[0]), "`strlen` needs a char array")
        M = self

        def strlen(f):
            arr = af(f)
            n = 0
            for v in arr.data:
                if v == 0:
                    return n
                n += 1
            M.event("oob_read", line, arr=arr.name, idx=n, size=n)
            return n
        return strlen, "int"

    # ================================================================ statements
    # Each closure takes the frame and returns a signal. A statement counts as executed,
    # and is recorded as a step, when it finishes.

    def stmt(self, node):
        method = getattr(self, "_s_" + node.__class__.__name__, None)
        if method is not None:
            return method(node)
        return self._expr_stmt(node)

    def _cond(self, node):
        """A condition. If it is a plain assignment, every evaluation reports assign_in_cond."""
        if isinstance(node, c_ast.Assignment) and node.op == "=":
            fn = self.scalar(node)[0]
            line = _line(node)
            target = node.lvalue
            name = target.name if isinstance(target, c_ast.ID) else getattr(target.name, "name", "?")
            M = self

            def cond(f):
                v = fn(f)
                M.event("assign_in_cond", line, var=name, value=json_num(v))
                return v
            return cond
        return self.scalar(node)[0]

    def _block_items(self, items):
        """Statements of one block, in the scope that is already open."""
        closures = []
        k = 0
        while k < len(items):
            item = items[k]
            if isinstance(item, c_ast.Decl):
                # `int a = 1, b;` is one statement: pycparser gives one Decl per name, all
                # pointing at the same type specifier
                group = [item]
                key = self._spec_coord(item)
                while (k + 1 < len(items) and isinstance(items[k + 1], c_ast.Decl)
                       and key is not None and self._spec_coord(items[k + 1]) == key):
                    k += 1
                    group.append(items[k])
                closure = self._decl_stmt(group)
            else:
                closure = self.stmt(item)
            if closure is not None:
                closures.append(closure)
            k += 1
        if not closures:
            return lambda f: None
        if len(closures) == 1:
            return closures[0]
        if len(closures) == 2:
            first, second = closures
            return lambda f: first(f) or second(f)
        closures = tuple(closures)

        def block(f):
            for run in closures:
                r = run(f)
                if r:
                    return r
        return block

    @staticmethod
    def _spec_coord(decl):
        node = decl.type
        while node is not None and not isinstance(node, c_ast.IdentifierType):
            node = getattr(node, "type", None)
        if node is None or node.coord is None:
            return None
        return (node.coord.line, node.coord.column)

    def _s_Compound(self, node):
        scope = _Scope()
        self.scopes.append(scope)
        closure = self._block_items(node.block_items or [])
        self.scopes.pop()
        return self._close_scope(scope, closure)

    def _s_EmptyStatement(self, node):
        return None

    def _s_Pragma(self, node):
        return None

    def _s_Decl(self, node):
        return self._decl_stmt([node])

    def _s_DeclList(self, node):
        return self._decl_stmt(list(node.decls))

    def _decl_stmt(self, decls):
        actions = [a for a in (self._decl_action(d) for d in decls) if a is not None]
        if not actions:
            return None
        line = _line(decls[0])
        M = self
        if len(actions) == 1:
            action = actions[0]

            def declare(f):
                if M.n >= STEP_CAP:
                    M.cap(line)
                action(f)
                M.n += 1
                if M.rec:
                    M.record(line)
            return declare

        def declare_all(f):
            if M.n >= STEP_CAP:
                M.cap(line)
            for action in actions:
                action(f)
            M.n += 1
            if M.rec:
                M.record(line)
        return declare_all

    def _decl_action(self, decl, is_global=False):
        """Closure that gives a declared variable its starting value. No step is counted here."""
        line = _line(decl)
        dtype = decl.type
        if isinstance(dtype, c_ast.FuncDecl):
            return None                                 # a prototype
        if decl.storage and any(word in decl.storage for word in ("static", "extern", "register")):
            word = decl.storage[0]
            raise SourceError("unsupported", line, f"`{word}` variables are not supported", word)
        if isinstance(dtype, c_ast.ArrayDecl):
            return self._array_decl(decl, is_global)
        ctype = self._scalar_type(self._type_names(dtype, "declaration"), line)
        init = decl.init
        if isinstance(init, c_ast.InitList):
            if len(init.exprs) != 1:
                raise SourceError("parse_error", line, f"`{decl.name}` is not an array")
            init = init.exprs[0]
        var = self._declare(decl.name, ctype, line, is_global)
        slot = var.slot
        cur = None if is_global else self.fn.dups.get(decl.name)
        if init is None:
            start = (0.0 if ctype == "float" else 0) if is_global else UNINIT
            if cur is None:
                def blank(f):
                    f[slot] = start
                return blank

            def blank_live(f):
                f[slot] = start
                f[cur] = slot
            return blank_live
        fn, itype = self.scalar(init, ctype == "float")
        fn = self.conv(fn, itype, ctype, line)
        if not is_global and _mentions(init, decl.name):
            inner = fn                          # `int x = x + 1;` reads the new, unset x

            def fn(f):
                f[slot] = UNINIT
                return inner(f)
        if cur is None:
            def assign(f):
                f[slot] = fn(f)
            return assign

        def assign_live(f):
            f[cur] = slot
            f[slot] = fn(f)
        return assign_live

    def _array_decl(self, decl, is_global):
        line = _line(decl)
        dtype = decl.type
        names = self._type_names(dtype.type, "array")
        elem = "char" if "char" in names else self._scalar_type(names, line)
        etype = "float" if elem == "float" else "int"
        zero = 0.0 if elem == "float" else 0
        name = decl.name
        size_fn = None
        if dtype.dim is not None:
            size_fn, stype = self.scalar(dtype.dim)
            if stype != "int":
                raise SourceError("parse_error", line, "an array size must be an integer")
        init = decl.init
        var = self._declare(name, "arr:" + elem, line, is_global)
        slot = var.slot
        if size_fn in self.kval:
            var.size = self.kval[size_fn]
        elif size_fn is None and _is_string(init):
            var.size = len(parse_string(init.value)) + 1
        elif size_fn is None and isinstance(init, c_ast.InitList):
            var.size = len(init.exprs)

        def size_of(f, given):
            if size_fn is None:
                return given
            n = size_fn(f)
            if n < 0 or n > 100000:
                raise Halt("runtime_error", line)
            return n

        if init is None:
            if size_fn is None:
                raise SourceError("parse_error", line, f"array `{name}` needs a size")
            fill = zero if is_global else UNINIT

            def empty(f):
                f[slot] = Arr([fill] * size_of(f, 0), name, elem)
            return empty
        if _is_string(init):
            if elem != "char":
                raise SourceError("parse_error", line, "only a char array can hold a string")
            codes = parse_string(init.value) + [0]

            def from_text(f):
                n = size_of(f, len(codes))
                f[slot] = Arr((codes + [0] * (n - len(codes)))[:n], name, elem)
            return from_text
        if isinstance(init, c_ast.InitList):
            cells = []
            for item in init.exprs:
                if isinstance(item, c_ast.InitList):
                    raise SourceError("unsupported", line, "multi-dimensional arrays are not supported",
                                      "multi_dim_array")
                fn, itype = self.scalar(item, elem == "float")
                cells.append(self.conv(fn, itype, etype, line))

            def from_list(f):
                values = [cell(f) for cell in cells]
                n = size_of(f, len(values))
                f[slot] = Arr((values + [zero] * (n - len(values)))[:n], name, elem)
            return from_list
        raise SourceError("parse_error", line, f"array `{name}` needs a `{{...}}` initialiser")

    def _expr_stmt(self, node):
        line = _line(node)
        M = self
        fn, _ = self.expr(node)
        if (isinstance(node, c_ast.FuncCall) and isinstance(node.name, c_ast.ID)
                and node.name.name in self.fns and self.fns[node.name.name].rtype != "void"):
            called = node.name.name

            def discard(f):
                if M.n >= STEP_CAP:
                    M.cap(line)
                M.event("discarded_call_value", line, fn=called)
                fn(f)
                M.n += 1
                if M.rec:
                    M.record(line)
            return discard

        def run(f):
            if M.n >= STEP_CAP:
                M.cap(line)
            fn(f)
            M.n += 1
            if M.rec:
                M.record(line)
        return run

    def _body(self, node):
        """Loop or branch body. Returns (closure or None, is_empty_statement)."""
        if node is None:
            return None, False
        if isinstance(node, c_ast.EmptyStatement):
            return None, True
        if isinstance(node, c_ast.Decl):
            raise SourceError("parse_error", _line(node), "a declaration cannot be a loop or branch body")
        return self.stmt(node), False

    def _s_If(self, node):
        line = _line(node)
        cond = self._cond(node.cond)
        then, then_empty = self._body(node.iftrue)
        other, other_empty = self._body(node.iffalse)
        empty = then_empty or other_empty
        k = len(self.branch_lines)
        self.branch_lines.append(line)
        self.BT.append(0)
        self.BF.append(0)
        M, BT, BF = self, self.BT, self.BF

        def branch(f):
            if M.n >= STEP_CAP:
                M.cap(line)
            if empty:
                M.event("empty_body", line, kind="if")
            c = cond(f)
            M.n += 1
            if M.rec:
                M.record(line)
            if c:
                BT[k] += 1
                if then:
                    return then(f)
            else:
                BF[k] += 1
                if other:
                    return other(f)
        return branch

    def _loop(self, line):
        k = len(self.loop_lines)
        self.loop_lines.append(line)
        self.L.append(0)
        return k

    def _s_While(self, node):
        line = _line(node)
        cond = self._cond(node.cond)
        self.loop_depth += 1
        body, empty = self._body(node.stmt)
        self.loop_depth -= 1
        k = self._loop(line)
        M, L = self, self.L

        def loop(f):
            if empty:
                M.event("empty_body", line, kind="while")
            while True:
                if M.n >= STEP_CAP:
                    M.cap(line)
                c = cond(f)
                M.n += 1
                if M.rec:
                    M.record(line)
                if not c:
                    return
                L[k] += 1
                if body:
                    r = body(f)
                    if r:
                        if r == BRK:
                            return
                        if r == RET:
                            return RET
        return loop

    def _s_DoWhile(self, node):
        line = _line(node)
        cond_line = _line(node.cond) or line
        self.loop_depth += 1
        body, empty = self._body(node.stmt)
        self.loop_depth -= 1
        cond = self._cond(node.cond)
        k = self._loop(line)
        M, L = self, self.L

        def loop(f):
            if empty:
                M.event("empty_body", line, kind="while")
            while True:
                L[k] += 1
                if body:
                    r = body(f)
                    if r:
                        if r == BRK:
                            return
                        if r == RET:
                            return RET
                if M.n >= STEP_CAP:
                    M.cap(cond_line)
                c = cond(f)
                M.n += 1
                if M.rec:
                    M.record(cond_line)
                if not c:
                    return
        return loop

    def _s_For(self, node):
        line = _line(node)
        scope = _Scope()
        self.scopes.append(scope)
        init = None
        if node.init is not None:
            if isinstance(node.init, c_ast.DeclList):
                actions = [a for a in (self._decl_action(d) for d in node.init.decls) if a is not None]

                def init(f):
                    for action in actions:
                        action(f)
            else:
                init = self.expr(node.init)[0]
        cond = self._cond(node.cond) if node.cond is not None else None
        update = self.expr(node.next)[0] if node.next is not None else None
        self.loop_depth += 1
        body, empty = self._body(node.stmt)
        self.loop_depth -= 1
        self.scopes.pop()
        k = self._loop(line)
        M, L = self, self.L

        def loop(f):
            if empty:
                M.event("empty_body", line, kind="for")
            if init:
                if M.n >= STEP_CAP:
                    M.cap(line)
                init(f)
                M.n += 1
                if M.rec:
                    M.record(line)
            while True:
                if M.n >= STEP_CAP:
                    M.cap(line)
                c = cond(f) if cond else 1
                M.n += 1
                if M.rec:
                    M.record(line)
                if not c:
                    return
                L[k] += 1
                if body:
                    r = body(f)
                    if r:
                        if r == BRK:
                            return
                        if r == RET:
                            return RET
                if update:
                    if M.n >= STEP_CAP:
                        M.cap(line)
                    update(f)
                    M.n += 1
                    if M.rec:
                        M.record(line)
        return self._close_scope(scope, loop)

    def _s_Break(self, node):
        return self._jump(node, BRK, "break")

    def _s_Continue(self, node):
        return self._jump(node, CNT, "continue")

    def _jump(self, node, signal, word):
        line = _line(node)
        if not self.loop_depth:
            raise SourceError("parse_error", line, f"`{word}` outside a loop")
        M = self

        def jump(f):
            if M.n >= STEP_CAP:
                M.cap(line)
            M.n += 1
            if M.rec:
                M.record(line)
            return signal
        return jump

    def _s_Return(self, node):
        line = _line(node)
        fn = self.fn
        name, rtype = fn.name, fn.rtype
        M, cnt = self, self.cnt
        garbage = FGARBAGE if rtype == "float" else GARBAGE
        value = None
        missing = False
        if node.expr is None:
            missing = rtype != "void"
        else:
            value, vtype = self.expr(node.expr, rtype == "float")
            if rtype != "void":
                if vtype == "void" or _is_arr(vtype):
                    raise SourceError("parse_error", line, "this cannot be returned from the function")
                value = self.conv(value, vtype, rtype, line)
        keep = rtype != "void"

        def ret(f):
            if M.n >= STEP_CAP:
                M.cap(line)
            v = value(f) if value else None
            if missing:
                M.event("missing_return", line)
                v = garbage
            elif not keep:
                v = None
            M.retval = v
            cnt["ret"] += 1
            M.n += 1
            if M.rec:
                M.eff.append("ret:%s:%d:%s" % (name, M.depth, fmt_num(v)))
                M.record(line)
            return RET
        return ret

    # ================================================================ running

    def bind(self, entry, py_args):
        """Turn a test's JSON args into values for `entry`'s parameters (03 §2.3)."""
        fn = self.fns.get(entry)
        if fn is None:
            raise SourceError("parse_error", 1, f"function `{entry}` is missing", entry)
        if len(py_args) != len(fn.params):
            raise SourceError("parse_error", fn.line,
                              f"`{entry}` must take {len(py_args)} argument(s), not {len(fn.params)}", entry)
        values = []
        for param, arg in zip(fn.params, py_args):
            ptype = param.type
            bad = SourceError("parse_error", fn.line, f"parameter `{param.name}` of `{entry}` has the wrong type",
                              entry)
            if _is_arr(ptype):
                elem = ptype[4:]
                if isinstance(arg, str):
                    if elem != "char":
                        raise bad
                    data = [ord(ch) & 0xFF for ch in arg] + [0]
                elif isinstance(arg, (list, tuple)):
                    if elem == "float":
                        data = [float(v) for v in arg]
                    else:
                        data = [ord(v) if isinstance(v, str) else wrap32(int(v)) for v in arg]
                else:
                    raise bad
                values.append(Arr(data, param.name, elem, world=True))
            elif isinstance(arg, (list, tuple)):
                raise bad
            elif isinstance(arg, str):
                if len(arg) != 1 or ptype != "int":
                    raise bad
                values.append(ord(arg))
            elif ptype == "float":
                values.append(float(arg))
            else:
                values.append(wrap32(int(arg)))
        return fn, values

    def run_test(self, entry, py_args, test_index=0, record=False, max_steps=None):
        """Call `entry` once. Returns the test's own status, value, output, counters, events
        and (when `record`) steps. Raises SourceError if the entry function does not fit."""
        fn, values = self.bind(entry, py_args)
        with self.lock:
            self.n = 0
            self.rec = bool(record)
            self.steps, self.eff, self.ev, self.events, self.out = [], [], [], [], []
            self.test = test_index
            self.depth = self.maxdepth = 0
            self.retval = None
            self.frame, self.disp = None, ()
            self.truncated = False
            self.max_steps = MAX_RECORDED_STEPS if max_steps is None else max_steps
            cnt, L, BT, BF = self.cnt, self.L, self.BT, self.BF
            for key in cnt:
                cnt[key] = 0
            L[:] = [0] * len(L)
            BT[:] = [0] * len(BT)
            BF[:] = [0] * len(BF)
            self.G[:] = self.global_blank
            if sys.getrecursionlimit() < _PY_FRAMES:
                sys.setrecursionlimit(_PY_FRAMES)
            status, returned, error = "ok", None, None
            try:
                for action in self.global_inits:
                    action(self.G)
                returned = self.invoke(fn, values, fn.line)
            except Halt as halt:
                status = halt.status
                if self.rec and self.frame is not None:
                    self.record(halt.line)
            except Exception as exc:        # a gap in the interpreter must not take the caller down
                if STRICT:
                    raise
                status, error = "runtime_error", f"{type(exc).__name__}: {exc}"

            loops, branch = {}, {}
            for k, line in enumerate(self.loop_lines):
                if L[k]:
                    key = "L%d" % line
                    loops[key] = loops.get(key, 0) + L[k]
            for k, line in enumerate(self.branch_lines):
                if BT[k] or BF[k]:
                    counts = branch.setdefault("B%d" % line, {"true": 0, "false": 0})
                    counts["true"] += BT[k]
                    counts["false"] += BF[k]
            arrays = [[None if v is UNINIT else v for v in a.data] if a.__class__ is Arr else None
                      for a in values]
            return {
                "status": status,
                "returned": json_num(returned),
                "printed": "".join(self.out),
                "loop_iters": loops,
                "branch": branch,
                "effects_count": {key: n for key, n in cnt.items() if n},
                "max_depth": self.maxdepth,
                "events": self.events,
                "steps": self.steps,
                "truncated": self.truncated,
                "arrays": arrays,
                "internal_error": error,        # not None only if the interpreter itself failed
            }
