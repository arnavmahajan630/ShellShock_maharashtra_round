"""Values of the C subset and the helpers that work on them (ml_plan/03 §2.1, §2.2, §2.5).

A C `int` (and `char`, `long`, `short`, `unsigned`) is a Python `int` kept inside 32 bits.
A C `float` or `double` is a Python `float`. The Python type is the type tag.
An array is an `Arr`: a Python list shared by reference when it is passed to a function.
"""
import math
import re
import struct

from ml.contracts.subset import GARBAGE

INT_MIN = -2147483648
INT_MAX = 2147483647
FGARBAGE = float(GARBAGE)


class _Unset:
    """A slot with no value. UNINIT: declared without an initialiser. NOTYET: not declared yet."""
    __slots__ = ("label",)

    def __init__(self, label):
        self.label = label

    def __repr__(self):
        return f"<{self.label}>"


UNINIT = _Unset("uninit")
NOTYET = _Unset("notyet")


class Arr:
    """A 1D array. `world` is true for the arrays the harness passed in as test arguments:
    only those produce read_cell / read_void / write_cell / compare effects."""
    __slots__ = ("data", "name", "elem", "world")

    def __init__(self, data, name, elem, world=False):
        self.data = data
        self.name = name
        self.elem = elem            # "int" | "float" | "char"
        self.world = world

    def text(self):
        """The C string held in a char array: characters up to the first terminator."""
        out = []
        for v in self.data:
            if v == 0 or v is UNINIT:
                break
            out.append(chr(int(v) & 0xFF))
        return "".join(out)


def wrap32(v):
    """Wrap a Python int to a signed 32-bit value."""
    return ((v + 0x80000000) & 0xFFFFFFFF) - 0x80000000


def fmt_num(v):
    """A value as written inside an effect string."""
    if v is None:
        return ""
    if isinstance(v, float):
        if v != v or v in (math.inf, -math.inf):
            return "nan" if v != v else ("inf" if v > 0 else "-inf")
        return format(v, ".6g")
    return str(v)


def json_num(v):
    """A value that json.dumps accepts under strict rules (no NaN or Infinity)."""
    if isinstance(v, float) and (v != v or v in (math.inf, -math.inf)):
        return None
    return v


# ---------------------------------------------------------------- literals

_SIMPLE_ESCAPES = {"n": 10, "t": 9, "r": 13, "0": 0, "\\": 92, "'": 39, '"': 34, "a": 7, "b": 8,
                   "f": 12, "v": 11, "?": 63}


def _unescape(body):
    """C escape sequences in the inside of a char or string literal -> list of char codes."""
    out, i, n = [], 0, len(body)
    while i < n:
        ch = body[i]
        if ch != "\\":
            out.append(ord(ch))
            i += 1
            continue
        i += 1
        if i >= n:
            out.append(92)
            break
        ch = body[i]
        if ch == "x":
            m = re.match(r"[0-9a-fA-F]+", body[i + 1:])
            if m:
                out.append(int(m.group(0), 16) & 0xFF)
                i += 1 + len(m.group(0))
                continue
            out.append(ord("x"))
            i += 1
        elif ch in "01234567":
            m = re.match(r"[0-7]{1,3}", body[i:])
            out.append(int(m.group(0), 8) & 0xFF)
            i += len(m.group(0))
        else:
            out.append(_SIMPLE_ESCAPES.get(ch, ord(ch)))
            i += 1
    return out


def parse_char(token):
    """'a' / '\\n' / '\\0' -> its code."""
    codes = _unescape(token[token.index("'") + 1: token.rindex("'")])
    return codes[0] if codes else 0


_STRING_PIECE = re.compile(r'"((?:[^"\\]|\\.)*)"', re.S)


def parse_string(token):
    """A string literal token (adjacent pieces are joined) -> list of char codes, no terminator."""
    codes = []
    for piece in _STRING_PIECE.findall(token):
        codes += _unescape(piece)
    return codes


def parse_int(token):
    text = token.rstrip("uUlL")
    if text[:2] in ("0x", "0X"):
        value = int(text, 16)
    elif text[:2] in ("0b", "0B"):
        value = int(text, 2)
    elif len(text) > 1 and text[0] == "0":
        value = int(text, 8)
    else:
        value = int(text)
    return wrap32(value)


def parse_float(token):
    return float(token.rstrip("fFlL"))


# ---------------------------------------------------------------- printf

_SPEC = re.compile(r"%([-+ 0#]*)(\d+)?(?:\.(\d+))?(hh|h|ll|l|L|z)?([diufFeEgGcsxXo%])")


class FormatError(ValueError):
    pass


def parse_format(codes):
    """A printf format (char codes) -> list of literal strings and (flags, width, prec, conv) tuples."""
    text = "".join(chr(c) for c in codes)
    parts, pos = [], 0
    while True:
        at = text.find("%", pos)
        if at < 0:
            break
        m = _SPEC.match(text, at)
        if not m:
            raise FormatError(text[at:at + 3])
        if at > pos:
            parts.append(text[pos:at])
        flags, width, prec, _length, conv = m.groups()
        if conv == "%":
            parts.append("%")
        else:
            parts.append((flags or "", width or "", prec, conv))
        pos = m.end()
    if pos < len(text):
        parts.append(text[pos:])
    merged = []
    for part in parts:
        if isinstance(part, str) and merged and isinstance(merged[-1], str):
            merged[-1] += part
        else:
            merged.append(part)
    return merged


def format_arg(spec, value, has_value):
    """One conversion. A mismatch between the conversion and the value is undefined in C;
    here it is deterministic: what an x86-64 Windows build prints."""
    flags, width, prec, conv = spec
    head = "%" + flags + width
    if conv in "diuxXo":
        if not has_value or isinstance(value, Arr):
            value = GARBAGE
        elif isinstance(value, float):
            value = struct.unpack("<i", struct.pack("<d", value)[:4])[0]
        if conv in "uxXo" and value < 0:
            value &= 0xFFFFFFFF
        conv = "d" if conv in "iu" else conv
        return (head + ("." + prec if prec is not None else "") + conv) % value
    if conv in "fFeEgG":
        if not has_value or isinstance(value, Arr):
            value = 0.0
        elif not isinstance(value, float):
            value = 0.0
        return (head + ("." + prec if prec is not None else "") + conv) % value
    if conv == "c":
        if not has_value or not isinstance(value, int):
            value = 63                                  # '?'
        return (head + "c") % chr(value & 0xFF)
    if conv == "s":
        text = value.text() if isinstance(value, Arr) else "(null)"
        return (head + ("." + prec if prec is not None else "") + "s") % text
    raise FormatError(conv)
