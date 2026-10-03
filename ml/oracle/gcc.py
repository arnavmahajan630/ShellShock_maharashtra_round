"""gcc stand-in for the interpreter (package A2).

    run_tests(problem, code, sample_only=False) -> dict      # schemas.RunResult, backend "gcc"
    observe(problem, code, arg_lists=None)      -> dict      # everything measured, per test

How it works
  1. The learner's file is compiled as one translation unit together with a generated
     `main` that calls the entry function with the arguments of the test whose index is
     given on the command line. No system header is visible to the learner's code:
     `#include` lines are blanked (as the interpreter does), `printf` goes to a bounded
     buffer, `strlen` and the world builtins are declared by hand.
  2. `rl_support.c` (compiled once per Python process) holds the world builtins as call
     counters, the printf buffer, the call-depth hooks and the report writer.
  3. Each test runs in its own process, all tests at the same time, each with a wall-clock
     limit. A crash or an endless loop in one test cannot lose the results of the others.

What gcc can and cannot report (see notes/A2.md)
  - returned, printed, array0, world-effect counts (fire, launch, door_open, door_closed,
    scan), `call` count and max recursion depth (`max_depth_le`) are measured.
  - read_cell / read_void / write_cell / compare / ret are not measurable here: a test that
    expects them has those keys skipped, and listed under got["unchecked"].
  - There is no trace, no events and no subset check: gcc accepts pointers, structs, etc.
"""
import atexit
import copy
import hashlib
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ml.contracts.subset import DEPTH_CAP, FLOAT_TOLERANCE, GARBAGE

BACKEND = "gcc"
TEST_TIMEOUT_S = 2.0        # wall clock per test process; tests run in parallel
COMPILE_TIMEOUT_S = 60.0
CACHE_SIZE = 256            # finished runs kept per process, keyed by the generated C source
STD = "gnu99"

WORLD_EFFECTS = ["fire", "launch", "door_open", "door_closed", "scan"]
EFFECT_ALIASES = {"open_door": "door_open", "close_door": "door_closed"}
UNMEASURABLE = ["read_cell", "read_void", "write_cell", "compare", "ret"]

_SUPPORT_SRC = Path(__file__).with_name("rl_support.c")
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

_lock = threading.Lock()
_state = {"dir": None, "support": None}
_cache = OrderedDict()


class GccUnavailable(RuntimeError):
    pass


# ------------------------------------------------------------------ toolchain

def gcc_path():
    path = os.environ.get("RELEARN_GCC") or shutil.which("gcc")
    if not path:
        raise GccUnavailable("gcc was not found on PATH (set RELEARN_GCC to its full path)")
    return path


def _rmtree(path):
    """Remove a directory. On Windows a just-killed .exe can stay locked for a moment."""
    for _ in range(20):
        shutil.rmtree(path, ignore_errors=True)
        if not os.path.exists(path):
            return True
        time.sleep(0.05)
    return False


def _cleanup_process_dir():
    if _state["dir"]:
        _rmtree(_state["dir"])
        _state["dir"] = _state["support"] = None


def _process_dir():
    """One temp directory per Python process; removed at exit. Each run gets a subfolder."""
    if _state["dir"] is None or not os.path.isdir(_state["dir"]):
        _state["dir"] = tempfile.mkdtemp(prefix="relearn_gcc_")
        _state["support"] = None
        atexit.register(_cleanup_process_dir)
    return _state["dir"]


def _support_object():
    with _lock:
        root = _process_dir()
        if _state["support"] and os.path.exists(_state["support"]):
            return _state["support"]
        obj = os.path.join(root, "rl_support.o")
        cmd = [gcc_path(), f"-std={STD}", "-O1", "-w", f"-DRL_GARBAGE=({GARBAGE})",
               f"-DRL_DEPTH_CAP={DEPTH_CAP}", "-c", str(_SUPPORT_SRC), "-o", obj]
        done = subprocess.run(cmd, capture_output=True, text=True, timeout=COMPILE_TIMEOUT_S,
                              stdin=subprocess.DEVNULL, creationflags=_NO_WINDOW)
        if done.returncode != 0:
            raise GccUnavailable(f"could not compile rl_support.c: {done.stderr.strip()[:500]}")
        _state["support"] = obj
        return obj


def clear_cache():
    _cache.clear()


# ------------------------------------------------------------------ source generation

_SIG_RE = re.compile(r"^\s*([A-Za-z_][\w\s]*?)\s+([A-Za-z_]\w*)\s*\((.*)\)\s*;?\s*$", re.S)
_PARAM_RE = re.compile(r"^\s*([A-Za-z_][\w\s]*?)\s+([A-Za-z_]\w*)\s*(\[\s*\d*\s*\])?\s*$")
_INCLUDE_RE = re.compile(r"^[ \t]*#[ \t]*include\b[^\n]*", re.M)

_ARRAY_KIND = {"int": ("int", "__rl_iarr", 1), "float": ("float", "__rl_farr", 2),
               "double": ("double", "__rl_darr", 3), "char": ("char", "__rl_carr", 4)}


def parse_signature(signature):
    """'int total_energy(int cells[], int n)' -> ('int', 'total_energy', [('int', True, 'cells'), ('int', False, 'n')])."""
    match = _SIG_RE.match(signature)
    if not match:
        raise ValueError(f"cannot read signature: {signature!r}")
    ret, name, inner = match.group(1).strip(), match.group(2), match.group(3).strip()
    params = []
    if inner and inner != "void":
        for part in inner.split(","):
            pm = _PARAM_RE.match(part)
            if not pm:
                raise ValueError(f"cannot read parameter {part!r} in signature {signature!r}")
            params.append((" ".join(pm.group(1).split()), bool(pm.group(3)), pm.group(2)))
    return " ".join(ret.split()), name, params


def _base(ctype):
    """The last type word decides: 'unsigned int' -> int, 'const char' -> char."""
    word = ctype.split()[-1]
    return word if word in ("int", "float", "double", "char", "long", "short", "void") else "int"


def _num(value, base):
    if isinstance(value, bool):
        value = int(value)
    if isinstance(value, str):
        if len(value) != 1:
            raise ValueError(f"string {value!r} given for a scalar parameter")
        value = ord(value)
    if base in ("float", "double"):
        return repr(float(value)) + ("f" if base == "float" else "")
    if isinstance(value, float):
        return repr(value)
    if value == -2147483648:
        return "(-2147483647 - 1)"
    return str(int(value))


def _c_string(text):
    return '"' + "".join("\\%03o" % b for b in text.encode("utf-8")) + '"'


def _case(index, ret, name, params, args):
    if len(args) != len(params):
        raise ValueError(f"test {index}: {len(args)} args for {len(params)} parameters of {name}")
    lines, call_args, arr0 = [f"    case {index}: {{"], [], None
    for k, ((ctype, is_array, _pname), value) in enumerate(zip(params, args)):
        base = _base(ctype)
        if is_array or isinstance(value, list):
            elem, maker, kind = _ARRAY_KIND.get(base, _ARRAY_KIND["int"])
            if isinstance(value, str):
                raw = value.encode("utf-8")
                n = len(raw) + 1                                   # with the '\0' terminator
                lines.append(f"        static const char __rl_i{k}[] = {_c_string(value)};")
                shown = len(raw)
            else:
                items = list(value)
                n = shown = len(items)
                body = ", ".join(_num(v, base) for v in items) if items else "0"
                lines.append(f"        static const {elem} __rl_i{k}[] = {{{body}}};")
            lines.append(f"        {elem} *__rl_a{k} = {maker}(__rl_i{k}, {n});")
            call_args.append(f"__rl_a{k}")
            if k == 0:
                arr0 = (kind, f"__rl_a{k}", shown)
        else:
            call_args.append(_num(value, base))
    if arr0:
        lines.append(f"        __rl_arr0({arr0[0]}, {arr0[1]}, {arr0[2]});")
    call = f"{name}({', '.join(call_args)})"
    rbase = _base(ret)
    lines.append("        __rl_start();")
    if rbase == "void":
        lines += [f"        {call};", "        __rl_stop();", "        __rl_ret_void();"]
    elif rbase in ("float", "double"):
        lines += [f"        double __rl_r = (double)({call});", "        __rl_stop();",
                  "        __rl_ret_float(__rl_r);"]
    else:
        lines += [f"        long long __rl_r = (long long)({call});", "        __rl_stop();",
                  "        __rl_ret_int(__rl_r);"]
    lines.append("        break; }")
    return "\n".join(lines)


_PRELUDE = """\
int __rl_printf(const char *, ...);
__SIZE_TYPE__ strlen(const char *);
void fire(void); void launch(void); void open_door(void); void close_door(void); int scan(int);
int __rl_begin(int, char **); void __rl_start(void); void __rl_stop(void); int __rl_finish(void);
void __rl_ret_int(long long); void __rl_ret_float(double); void __rl_ret_void(void);
int *__rl_iarr(const int *, int); float *__rl_farr(const float *, int);
double *__rl_darr(const double *, int); char *__rl_carr(const char *, int);
void __rl_arr0(int, const void *, int);
#define printf __rl_printf
#define main __rl_learner_main
#line 1 "core.c"
"""


def build_source(signature, code, arg_lists):
    """The C file handed to gcc: declarations, the learner's code, a generated main."""
    ret, name, params = parse_signature(signature)
    learner = _INCLUDE_RE.sub("", code.replace("\r\n", "\n").replace("\r", "\n"))
    cases = "\n".join(_case(i, ret, name, params, args) for i, args in enumerate(arg_lists))
    return (_PRELUDE + learner + "\n#line 1 \"rl_harness.c\"\n#undef main\n#undef printf\n"
            "__attribute__((no_instrument_function)) int main(int __rl_argc, char **__rl_argv) {\n"
            "    switch (__rl_begin(__rl_argc, __rl_argv)) {\n" + cases + "\n"
            "    default: return 2;\n    }\n    return __rl_finish();\n}\n")


# ------------------------------------------------------------------ compile and run

_ERR_RE = re.compile(r"core\.c:(\d+):(?:\d+:)? (?:fatal )?error: (.*)")


def _compile_error(stderr):
    for line in stderr.splitlines():
        match = _ERR_RE.search(line)
        if match:
            return f"line {match.group(1)}: {match.group(2).strip()}"
    for line in stderr.splitlines():
        if "undefined reference" in line or "multiple definition" in line or "error" in line:
            return re.sub(r"^.*?(undefined reference|multiple definition|error)", r"\1", line).strip()[:200]
    return (stderr.strip().splitlines() or ["compile failed"])[0][:200]


def _blank(status, error):
    return {"status": status, "error": error, "returned": None, "printed": "", "array0": None,
            "effects": {name: 0 for name in WORLD_EFFECTS}, "max_depth": 0, "calls": 0}


_STATUS = {"ok": ("ok", None), "div_zero": ("runtime_error", "div_zero"), "segv": ("runtime_error", "segfault"),
           "depth_cap": ("timeout", "depth_cap_hit"), "output_limit": ("timeout", "output_limit")}


def _parse_report(stdout):
    start = stdout.rfind("@@RL1@@")
    end = stdout.rfind("@@END@@")
    if start < 0 or end < start:
        return None
    fields = {}
    for line in stdout[start:end].splitlines()[1:]:
        key, _, rest = line.partition(" ")
        fields[key] = rest.strip()
    status, error = _STATUS.get(fields.get("status", ""), ("runtime_error", fields.get("status") or "no_status"))
    obs = _blank(status, error)
    kind, _, value = fields.get("ret", "none").partition(" ")
    if kind == "int":
        obs["returned"] = int(value)
    elif kind == "float":
        obs["returned"] = float(value)
    obs["printed"] = bytes.fromhex(fields.get("printed", "")).decode("utf-8", errors="replace")
    cells = fields.get("array0", "none").split()
    if cells != ["none"]:                       # an empty array argument is [], no array argument is None
        obs["array0"] = [float(c) if any(ch in c for ch in ".einf") else int(c) for c in cells]
    obs["max_depth"] = int(fields.get("depth", 0))
    obs["calls"] = int(fields.get("calls", 0))
    for pair in fields.get("fx", "").split():
        name, _, count = pair.partition("=")
        obs["effects"][name] = int(count)
    return obs


def _run_one(exe, index, timeout_s):
    try:
        done = subprocess.run([exe, str(index)], capture_output=True, timeout=timeout_s,
                              stdin=subprocess.DEVNULL, creationflags=_NO_WINDOW)
    except subprocess.TimeoutExpired:
        return _blank("timeout", "time_limit")
    obs = _parse_report(done.stdout.decode("latin-1"))
    if obs is not None:
        return obs
    code = done.returncode & 0xFFFFFFFF
    reason = {0xC0000094: "div_zero", 0xC0000005: "segfault", 0xC00000FD: "stack_overflow"}.get(code, f"crash_0x{code:08X}")
    return _blank("runtime_error", reason if done.returncode else "no_report")


def _first_bad(statuses):
    return next((s for s in statuses if s != "ok"), "ok")


def observe(problem, code, arg_lists=None, timeout_s=None):
    """Compile `code` and call the problem's entry function once per argument list.

    arg_lists defaults to the args of every test in problem["tests"]. `problem` only needs
    "signature" (and "tests" when arg_lists is None), so ad-hoc checks can pass
    {"signature": "int f(int x)"}.

    Returns {"status", "error", "runs": [...], "compile_ms", "run_ms"} where each run is
    {"status", "error", "returned", "printed", "array0", "effects", "max_depth", "calls"}.
    status is the first run that did not end "ok" (in test order), or "parse_error" when
    gcc rejects the file; then every run carries the compiler's first error.
    """
    if arg_lists is None:
        arg_lists = [t["args"] for t in problem["tests"]]
    timeout_s = TEST_TIMEOUT_S if timeout_s is None else timeout_s
    source = build_source(problem["signature"], code, arg_lists)
    key = hashlib.sha1(source.encode("utf-8")).hexdigest()
    if key in _cache:
        _cache.move_to_end(key)
        hit = copy.deepcopy(_cache[key])
        hit["cached"] = True
        return hit

    support = _support_object()
    workdir = tempfile.mkdtemp(prefix="run_", dir=_process_dir())
    try:
        src, exe = os.path.join(workdir, "prog.c"), os.path.join(workdir, "prog.exe")
        Path(src).write_text(source, encoding="utf-8", newline="\n")
        started = time.perf_counter()
        cmd = [gcc_path(), f"-std={STD}", "-O0", "-w", "-fno-builtin", "-finstrument-functions",
               src, support, "-o", exe]
        try:
            built = subprocess.run(cmd, capture_output=True, text=True, errors="replace",
                                   timeout=COMPILE_TIMEOUT_S, stdin=subprocess.DEVNULL,
                                   creationflags=_NO_WINDOW)
            failed, stderr = built.returncode != 0, built.stderr
        except subprocess.TimeoutExpired:
            failed, stderr = True, "error: compiler timed out"
        compile_ms = (time.perf_counter() - started) * 1000
        if failed or not os.path.exists(exe):
            error = _compile_error(stderr)
            result = {"status": "parse_error", "error": error, "compile_ms": compile_ms, "run_ms": 0.0,
                      "runs": [_blank("parse_error", error) for _ in arg_lists], "cached": False}
        else:
            started = time.perf_counter()
            if arg_lists:
                with ThreadPoolExecutor(max_workers=min(8, len(arg_lists))) as pool:
                    runs = list(pool.map(lambda i: _run_one(exe, i, timeout_s), range(len(arg_lists))))
            else:
                runs = []
            result = {"status": _first_bad(r["status"] for r in runs), "error": None,
                      "compile_ms": compile_ms, "run_ms": (time.perf_counter() - started) * 1000,
                      "runs": runs, "cached": False}
            result["error"] = next((r["error"] for r in runs if r["status"] != "ok"), None)
    finally:
        _rmtree(workdir)

    # A wall-clock time-out can be the machine being busy, so it is never remembered.
    if not any(r["error"] == "time_limit" for r in result["runs"]):
        _cache[key] = copy.deepcopy(result)
        while len(_cache) > CACHE_SIZE:
            _cache.popitem(last=False)
    return result


# ------------------------------------------------------------------ expectations

def _same(expected, got):
    if isinstance(expected, bool) or isinstance(got, bool):
        return expected == got
    if isinstance(expected, (int, float)) and isinstance(got, (int, float)):
        if isinstance(expected, float) or isinstance(got, float):
            return abs(float(expected) - float(got)) <= FLOAT_TOLERANCE
        return expected == got
    if isinstance(expected, list) and isinstance(got, list):
        return len(expected) == len(got) and all(_same(e, g) for e, g in zip(expected, got))
    return expected == got


def _judge(test, obs):
    """Compare one observation with one test's `expect`. Returns (got, passed)."""
    got, ok, unchecked = {}, obs["status"] == "ok", []
    for key, expected in test["expect"].items():
        effect = EFFECT_ALIASES.get(key, key)
        if key == "returned":
            got[key] = obs["returned"]
            ok = ok and obs["returned"] is not None and _same(expected, obs["returned"])
        elif key == "printed":
            got[key] = obs["printed"]
            ok = ok and expected == obs["printed"]
        elif key == "array0":
            value = obs["array0"]
            if value is not None and isinstance(expected, str):
                value = bytes(v & 0xFF for v in value).decode("utf-8", errors="replace")
            got[key] = value
            ok = ok and value is not None and _same(expected, value)
        elif key == "max_depth_le":
            got["max_depth"] = obs["max_depth"]
            ok = ok and obs["max_depth"] <= expected
        elif effect in obs["effects"]:
            got[key] = obs["effects"][effect]
            ok = ok and expected == obs["effects"][effect]
        elif key == "call":
            got[key] = obs["calls"]
            ok = ok and expected == obs["calls"]
        else:
            unchecked.append(key)           # read_cell, write_cell, compare, ...: gcc cannot count them
    if unchecked:
        got["unchecked"] = unchecked
    if obs["status"] != "ok":
        got["error"] = obs["error"] or obs["status"]
    return got, bool(ok)


def _forbidden_used(problem, code):
    stripped = re.sub(r"//[^\n]*|/\*.*?\*/", " ", code, flags=re.S)
    stripped = re.sub(r'"(?:\\.|[^"\\\n])*"', '""', stripped)
    return [name for name in problem.get("forbid") or [] if re.search(rf"\b{re.escape(name)}\s*\(", stripped)]


def run_tests(problem, code, sample_only=False):
    """Run the problem's tests (only the `sample` ones when sample_only) on `code` with gcc.

    Returns schemas.RunResult: {"status", "backend": "gcc", "tests": {"passed", "total", "results"}}.
    status: "ok" when every test ran to its end (passing or not); otherwise how the first
    test that did not ended: "timeout", "runtime_error"; or "parse_error" when gcc rejects
    the file; or "unsupported" when the code calls a builtin the problem forbids.
    """
    tests = [t for t in problem["tests"] if t.get("sample")] if sample_only else list(problem["tests"])
    forbidden = _forbidden_used(problem, code)
    if forbidden:
        status = "unsupported"
        runs = [_blank("unsupported", f"forbidden builtin: {forbidden[0]}") for _ in tests]
    else:
        seen = observe(problem, code, [t["args"] for t in tests])
        status, runs = seen["status"], seen["runs"]
    results, passed = [], 0
    for test, obs in zip(tests, runs):
        got, ok = _judge(test, obs)
        passed += ok
        results.append({"args": test["args"], "expected": dict(test["expect"]), "got": got, "pass": ok})
    return {"status": status, "backend": BACKEND,
            "tests": {"passed": passed, "total": len(tests), "results": results}}
