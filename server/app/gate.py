"""Serving-time gate (package G1, 03 §3.7.1): runs before the interpreter and the model.

    check(problem, code) -> {"code": "G0" | "G1" | ... | "G7", "message": str}     # schemas.Gate

`problem` is a problem dict (schemas.Problem); `code` is what the learner typed.
G0 means "go on to the interpreter". Messages come from ml/contracts/subset.GATE_MESSAGES.
G8 (smart quotes and other non-ASCII punctuation) is fixed silently: the gate works on
normalise(code) and reports G0. Callers that pass the code on should pass normalise(code).

Order of the checks:
    G1 empty  ->  G7 too large  ->  G2 same as starter  ->  G3c C++  ->  G4 bad `#` line
    ->  parse (G3b Python / G3a parse error)  ->  G4b forbidden builtin  ->  G4 unsupported
    construct  ->  G5 entry signature  ->  G5b no recursion  ->  G6 constant answer  ->  G0

Only pycparser is used; nothing is executed.
"""
import logging
import re
import time

from pycparser import c_ast, c_parser

from ml.contracts.subset import GATE_MESSAGES, LIB_BUILTINS, MAX_BYTES, MAX_LINES, REJECTED, WORLD_BUILTINS

log = logging.getLogger("relearn.gate")

# ------------------------------------------------------------------ G8: normalising

_PUNCT = {
    "“": '"', "”": '"', "„": '"', "‟": '"', "«": '"', "»": '"', "″": '"',
    "‘": "'", "’": "'", "‚": "'", "‛": "'", "′": "'", "´": "'",
    "–": "-", "—": "-", "−": "-", "‐": "-", "‑": "-",
    " ": " ", " ": " ", " ": " ", " ": " ", " ": " ", " ": " ", "　": " ",
    "​": "", "‌": "", "‍": "", "﻿": "", "⁠": "",
    "…": "...", "×": "*", "÷": "/", "≤": "<=", "≥": ">=", "≠": "!=",
}
_PUNCT_RE = re.compile("|".join(re.escape(ch) for ch in _PUNCT))


def normalise(code):
    """G8: smart quotes, long dashes, odd spaces and full-width ASCII become plain ASCII.
    Line ends become "\\n". The number of lines never changes."""
    code = code.replace("\r\n", "\n").replace("\r", "\n")
    code = _PUNCT_RE.sub(lambda m: _PUNCT[m.group(0)], code)
    return "".join(chr(ord(ch) - 0xFEE0) if "！" <= ch <= "～" else ch for ch in code)


# ------------------------------------------------------------------ comments and `#` lines

_TOKEN_RE = re.compile(r'//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\\n])*"|\'(?:\\.|[^\'\\\n])*\'', re.S)


def strip_comments(code):
    """Comments become spaces; newlines stay, so line numbers do not move (03 §2.5)."""
    def blank(match):
        text = match.group(0)
        return text if text[0] in "\"'" else re.sub(r"[^\n]", " ", text)
    return _TOKEN_RE.sub(blank, code)


_LITERAL_RE = re.compile(r'"(?:\\.|[^"\\\n])*"|\'(?:\\.|[^\'\\\n])*\'')


def _outside_literals(code, fn):
    """Apply fn to the parts of `code` that are not string or char literals."""
    out, last = [], 0
    for match in _LITERAL_RE.finditer(code):
        out += [fn(code[last:match.start()]), match.group(0)]
        last = match.end()
    return "".join(out + [fn(code[last:])])


def _without_literals(code):
    """String and char literals become "" so that their contents never look like code."""
    return _LITERAL_RE.sub('""', code)


_DEFINE_RE = re.compile(r"^\s*#\s*define\s+([A-Za-z_]\w*)(\(?)\s*(.*?)\s*$")
_DIRECTIVE_RE = re.compile(r"^\s*#\s*(\w*)")


def preprocess(code):
    """`code` must already be normalised and comment-free.
    Returns (text for the parser, None) or (None, name of the rejected directive)."""
    lines, defines = code.split("\n"), []
    for index, line in enumerate(lines):
        if not line.lstrip().startswith("#"):
            continue
        word = _DIRECTIVE_RE.match(line).group(1)
        if word == "include":
            lines[index] = ""
            continue
        define = _DEFINE_RE.match(line)
        if define and not define.group(2) and define.group(3):
            defines.append((index, define.group(1), define.group(3)))
            lines[index] = ""
            continue
        if define and define.group(2):
            return None, "#define with arguments"
        return None, "#" + word
    for index, name, value in defines:                      # text substitution, below the #define only
        pattern = re.compile(rf"\b{re.escape(name)}\b")
        for k in range(index + 1, len(lines)):
            if name in lines[k]:
                lines[k] = _outside_literals(lines[k], lambda text: pattern.sub(lambda _m: value, text))
    return "\n".join(lines), None


# ------------------------------------------------------------------ G3: what kind of "not C" is it

_CPP_RE = re.compile(r"\bcout\b|\bcin\b|\bendl\b|\bstd\s*::|\busing\s+namespace\b|\bclass\s+\w+\s*[{:]|"
                     r"\btemplate\s*<|\bnew\s+\w+\s*[\[(;]|::")
_CPP_INCLUDE_RE = re.compile(r"^\s*#\s*include\s*[<\"]\s*(iostream|bits/stdc\+\+\.h|vector|string|algorithm|"
                             r"cmath|cstdio|cstring|map|set|iomanip)\s*[>\"]", re.M)
_PY_STRONG = [
    re.compile(r"^\s*def\s+\w+\s*\(.*\)\s*(->[^:]+)?:\s*$", re.M),
    re.compile(r"\belif\b"),
    re.compile(r"^\s*(if|else|for|while)\b[^;{}]*:\s*\n[ \t]+\S", re.M),
    re.compile(r"\bfor\s+\w+\s+in\b"),
    re.compile(r"^\s*(import\s+\w+|from\s+\w+\s+import\b)", re.M),
]
_PY_PRINT = re.compile(r"\bprint\s*\(")


def _looks_like_cpp(code):
    return bool(_CPP_INCLUDE_RE.search(code) or _CPP_RE.search(_without_literals(code)))


def _looks_like_python(code):
    bare = _without_literals(code)
    if any(pattern.search(bare) for pattern in _PY_STRONG):
        return True
    return bool(_PY_PRINT.search(bare)) and ";" not in bare and "{" not in bare


# ------------------------------------------------------------------ G3a: line and short reason

_KEYWORDS = set("auto break case char const continue default do double else enum extern float for goto if "
                "inline int long register restrict return short signed sizeof static struct switch typedef "
                "union unsigned void volatile while".split())
_COORD_RE = re.compile(r"^[^:]*:(\d+):(\d+):\s*(.*)$", re.S)
_SCAN_BUDGET_S = 0.15


def _try_parse(text):
    """Returns (ast, None) or (None, error message). pycparser can raise more than ParseError."""
    try:
        return c_parser.CParser().parse(text), None
    except Exception as exc:                                # noqa: BLE001 - any failure is "does not parse"
        return None, str(exc) or type(exc).__name__


def _brace_problem(text):
    """(line, reason) for an unmatched `}` or a `{` never closed, else None."""
    depth, bare = 0, _without_literals(text)
    for number, line in enumerate(bare.split("\n"), start=1):
        for ch in line:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth < 0:
                    return number, "unmatched `}`"
    if depth > 0:
        last = max((n for n, line in enumerate(bare.split("\n"), start=1) if line.strip()), default=1)
        return last, "missing `}`"
    return None


def _line_by_removal(text):
    """For errors pycparser reports without a position: the first line whose removal lets the file parse."""
    lines, started = text.split("\n"), time.perf_counter()
    # Lines with an operator left hanging are tried first; they are the usual cause.
    hanging = [i for i, line in enumerate(lines) if re.search(r"[-+*/%=<>!&|,(]\s*[;)]", _without_literals(line))]
    for index in hanging + [i for i in range(len(lines)) if i not in hanging]:
        if not lines[index].strip() or lines[index].strip() in ("{", "}"):
            continue
        if time.perf_counter() - started > _SCAN_BUDGET_S:
            break
        if _try_parse("\n".join(lines[:index] + [""] + lines[index + 1:]))[0] is not None:
            return index + 1
    if hanging:
        return hanging[0] + 1
    return next((i + 1 for i, line in enumerate(lines) if line.strip()), 1)


def _parse_error(text, message):
    """Turn pycparser's message into (line, short reason)."""
    lines = text.split("\n")
    for number, row in enumerate(_without_literals(text).split("\n"), start=1):
        if "/*" in row:                                     # strip_comments left it: it never closes
            return number, "unterminated comment"
    match = _COORD_RE.match(message)
    if not match:
        if "end of input" in message.lower():
            return _brace_problem(text) or (len(lines), "the code stops too early")
        problem = _brace_problem(text)
        if problem:
            return problem
        reason = "incomplete expression" if "expression" in message.lower() else "could not be read as C"
        return _line_by_removal(text), reason

    line, col, detail = int(match.group(1)), int(match.group(2)), match.group(3).strip()
    if detail.startswith("Illegal character"):
        char = detail[len("Illegal character"):].strip()
        if len(char) >= 2 and char[0] == char[-1] and char[0] in "'\"":
            char = char[1:-1]
        if char == '"':
            return line, "unterminated string"
        if char == "'":
            return line, "bad character literal"
        if len(char) == 1 and char.isascii() and char.isprintable():
            return line, f"illegal character `{char}`"
        return line, "illegal character"
    if detail.startswith("before:"):
        token = detail[len("before:"):].strip()
        row = lines[line - 1] if 0 < line <= len(lines) else ""
        head = row[:col - 1]
        previous = re.search(r"([A-Za-z_]\w*)\s*$", head)
        if previous and re.match(r"[A-Za-z_]\w*$", token) and previous.group(1) not in _KEYWORDS \
                and not head[:previous.start()].strip():
            return line, f"unknown type `{previous.group(1)}`"
        before = "\n".join(lines[:line - 1] + [head]).rstrip()
        if before and before[-1] not in ";{}:" and (not head.strip() or token == "}"):
            return before.count("\n") + 1, "missing `;`"
        bare = _without_literals(row)
        if bare.count("(") > bare.count(")"):
            return line, "missing `)`"
        return line, f"unexpected `{token}`"
    if "function definition" in detail.lower():
        return line, "this is not a C function"
    if not detail or "http" in detail or "\n" in detail:
        return line, "could not be read as C"
    return line, (detail[:1].lower() + detail[1:60]).rstrip(". ")


# ------------------------------------------------------------------ G4: unsupported constructs

_STRING_H = {"strcmp", "strncmp", "strcpy", "strncpy", "strcat", "strncat", "strchr", "strrchr", "strstr",
             "strtok", "strrev", "strlwr", "strupr", "strdup", "memcpy", "memset", "memmove", "memcmp"}
_MALLOC = {"malloc", "calloc", "realloc", "free"}
_SCANF = {"scanf", "sscanf", "fscanf", "gets", "fgets", "getchar", "getc"}
# Which rejected construct to name when a file has several: the most telling one first.
_PRIORITY = ["scanf", "malloc", "string_h", "struct", "switch", "goto", "multi_dim_array", "pointer", "address_of"]


class _Scan(c_ast.NodeVisitor):
    """One walk over the file: rejected constructs, calls, and who calls whom."""

    def __init__(self):
        self.rejected = {}          # REJECTED key -> first line
        self.other = []             # (line, word) for unsupported things that have no REJECTED key
        self.functions = {}         # name -> FuncDef
        self.calls = {}             # function name -> set of called names
        self.called = []            # (name, line) for every call
        self._current = None

    def _reject(self, key, node):
        self.rejected.setdefault(key, node.coord.line if node.coord else 0)

    def visit_FuncDef(self, node):
        self.functions[node.decl.name] = node
        self._current = node.decl.name
        self.calls.setdefault(self._current, set())
        self.generic_visit(node)
        self._current = None

    def visit_PtrDecl(self, node):
        self._reject("pointer", node)
        self.generic_visit(node)

    def visit_UnaryOp(self, node):
        if node.op == "&":
            self._reject("address_of", node)
        elif node.op == "*":
            self._reject("pointer", node)
        elif node.op == "sizeof":
            self.other.append((node.coord.line if node.coord else 0, "sizeof"))
        self.generic_visit(node)

    def visit_Struct(self, node):
        self._reject("struct", node)
        self.generic_visit(node)

    visit_Union = visit_StructRef = visit_Struct

    def visit_Switch(self, node):
        self._reject("switch", node)
        self.generic_visit(node)

    def visit_Goto(self, node):
        self._reject("goto", node)
        self.generic_visit(node)

    visit_Label = visit_Goto

    def visit_ArrayDecl(self, node):
        if isinstance(node.type, c_ast.ArrayDecl):
            self._reject("multi_dim_array", node)
        self.generic_visit(node)

    def visit_ArrayRef(self, node):
        if isinstance(node.name, c_ast.ArrayRef):
            self._reject("multi_dim_array", node)
        self.generic_visit(node)

    def visit_Typedef(self, node):
        self.other.append((node.coord.line if node.coord else 0, "typedef"))
        self.generic_visit(node)

    def visit_Enum(self, node):
        self.other.append((node.coord.line if node.coord else 0, "enum"))
        self.generic_visit(node)

    def visit_FuncCall(self, node):
        if isinstance(node.name, c_ast.ID):
            name = node.name.name
            self.called.append((name, node.coord.line if node.coord else 0))
            if self._current:
                self.calls[self._current].add(name)
            if name in _SCANF:
                self._reject("scanf", node)
            elif name in _MALLOC:
                self._reject("malloc", node)
            elif name in _STRING_H:
                self._reject("string_h", node)
        else:
            self._reject("pointer", node)                   # a call through an expression
        self.generic_visit(node)


def _unsupported(scan):
    """The word for the G4 message, or None."""
    for key in _PRIORITY:
        if key in scan.rejected:
            return REJECTED[key]
    if scan.other:
        return min(scan.other)[1]
    known = set(scan.functions) | set(WORLD_BUILTINS) | set(LIB_BUILTINS)
    for name, _line in scan.called:
        if name not in known:
            return f"{name}()"
    return None


# ------------------------------------------------------------------ G5, G5b, G6

_SIG_RE = re.compile(r"^\s*[A-Za-z_][\w\s]*?\s+([A-Za-z_]\w*)\s*\((.*)\)\s*;?\s*$", re.S)


def _entry(signature):
    """('total_energy', 2) from 'int total_energy(int cells[], int n)'."""
    match = _SIG_RE.match(signature or "")
    if not match:
        return None, 0
    inner = match.group(2).strip()
    return match.group(1), 0 if inner in ("", "void") else len(inner.split(","))


def _params(funcdef):
    args = funcdef.decl.type.args
    params = list(args.params) if args else []
    if len(params) == 1 and isinstance(params[0], c_ast.Typename):      # f(void)
        return []
    return [getattr(p, "name", None) for p in params]


def _recursive(scan, entry):
    """True if a call chain starting in `entry` reaches a function that calls itself, directly or in a cycle."""
    reachable, stack = set(), [entry]
    while stack:
        name = stack.pop()
        if name in reachable or name not in scan.functions:
            continue
        reachable.add(name)
        stack.extend(scan.calls.get(name, ()))

    def reaches(start, target, seen):
        for callee in scan.calls.get(start, ()):
            if callee == target:
                return True
            if callee in scan.functions and callee not in seen:
                seen.add(callee)
                if reaches(callee, target, seen):
                    return True
        return False

    return any(reaches(name, name, set()) for name in reachable)


class _Reads(c_ast.NodeVisitor):
    """Names read in a function body, and whether it returns a value or prints."""

    def __init__(self):
        self.read, self.outputs = set(), False

    def visit_Compound(self, node):
        for item in node.block_items or []:
            if isinstance(item, c_ast.Assignment) and item.op == "=" and isinstance(item.lvalue, c_ast.ID):
                self.visit(item.rvalue)                     # the statement `n = 5;` writes n, it does not read it
            else:
                self.visit(item)                            # `if (code = 42)` uses the value: that is a read (M06)

    def visit_ID(self, node):
        self.read.add(node.name)

    def visit_FuncCall(self, node):
        if isinstance(node.name, c_ast.ID) and node.name.name == "printf":
            self.outputs = True
        if node.args:
            self.visit(node.args)
        if not isinstance(node.name, c_ast.ID):
            self.visit(node.name)

    def visit_Return(self, node):
        if node.expr is not None:
            self.outputs = True
            self.visit(node.expr)


def _constant_answer(funcdef):
    """G6: the function never reads a parameter and yet returns a value or prints."""
    params = [p for p in _params(funcdef) if p]
    if not params:
        return False
    reads = _Reads()
    reads.visit(funcdef.body)
    return reads.outputs and not (set(params) & reads.read)


# ------------------------------------------------------------------ the gate

def _gate(code, **fields):
    return {"code": code, "message": GATE_MESSAGES[code].format(**fields)}


def _tokens(code):
    return re.findall(r"[A-Za-z_]\w*|\d+\.?\d*|\S", strip_comments(normalise(code)))


MAX_NESTING = 100


def _too_deep(text):
    """Line where brackets nest deeper than MAX_NESTING, else None.

    Checked before parsing so the answer does not depend on Python's recursion limit, which the
    interpreter raises when it is imported.
    """
    depth, line = 0, 1
    for char in text:
        if char == "\n":
            line += 1
        elif char in "([{":
            depth += 1
            if depth > MAX_NESTING:
                return line
        elif char in ")]}":
            depth -= 1
    return None


def check(problem, code):
    """Run the gate. Returns schemas.Gate: {"code": ..., "message": ...}."""
    problem = problem or {}
    raw = code or ""
    code = normalise(raw)                                               # G8
    bare = strip_comments(code)

    if not bare.strip():                                                # G1
        return _gate("G1")
    if len(raw.splitlines()) > MAX_LINES or len(raw.encode("utf-8")) > MAX_BYTES:    # G7
        return _gate("G7")
    if problem.get("starter") is not None and _tokens(code) == _tokens(problem["starter"]):     # G2
        return _gate("G2")
    if _looks_like_cpp(bare):                                           # G3c
        return _gate("G3c")

    text, directive = preprocess(bare)
    if directive:                                                       # G4: a `#` line outside the subset
        return _gate("G4", construct=directive)
    deep_line = _too_deep(text)
    if deep_line:                                                       # G3a, the same answer whatever the recursion limit is
        return _gate("G3a", n=deep_line, reason="expression nested too deeply")
    ast, error = _try_parse(text)
    if ast is None:
        if _looks_like_python(bare):                                    # G3b
            return _gate("G3b")
        line, reason = _parse_error(text, error)                        # G3a
        return _gate("G3a", n=line, reason=reason)

    scan = _Scan()
    try:
        scan.visit(ast)
    except RecursionError:                                              # absurdly deep nesting
        return _gate("G3a", n=1, reason="expression nested too deeply")
    construct = _unsupported(scan)
    forbidden = [name for name, _line in scan.called if name in (problem.get("forbid") or [])]
    if forbidden:                                                       # G4b (before G4: strlen is a builtin)
        return {"code": "G4b", "message": GATE_MESSAGES["G4b"].replace("strlen", forbidden[0])}
    if construct:                                                       # G4
        return _gate("G4", construct=construct)

    signature = problem.get("signature")
    name, arity = _entry(signature)
    if name:
        entry = scan.functions.get(name)
        if entry is None or len(_params(entry)) != arity:               # G5
            return _gate("G5", signature=signature.strip())
        if problem.get("sector") == "recursion" and not _recursive(scan, name):     # G5b
            return _gate("G5b")
        if _constant_answer(entry):                                     # G6
            log.info("gate G6 (constant answer) on %s", problem.get("problem_id"))
            return _gate("G6")
    return _gate("G0")


gate = check
