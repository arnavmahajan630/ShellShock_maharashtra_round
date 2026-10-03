"""Text clean-up before pycparser sees the learner's file (plans/03 §2.5).

  1. normalise smart quotes and other non-ASCII punctuation (gate G8),
  2. replace `//` and `/* */` comments with spaces, keeping every newline,
  3. blank out `#include` lines,
  4. apply `#define NAME literal` as text substitution and blank the line,
  5. reject every other `#` line.

Lines are never added or removed, so `coord.line` of every node is the learner's own line.
"""
import re


class SourceError(Exception):
    """The file cannot be run. `status` is "parse_error" or "unsupported" (trace statuses)."""

    def __init__(self, status, line, reason, construct=None):
        super().__init__(f"line {line}: {reason}")
        self.status = status
        self.line = line
        self.reason = reason
        self.construct = construct      # key of subset.REJECTED, or the offending word


_PUNCT = str.maketrans({
    "“": '"', "”": '"', "„": '"', "‟": '"', "«": '"', "»": '"',
    "″": '"', "＂": '"',
    "‘": "'", "’": "'", "‚": "'", "‛": "'", "′": "'", "´": "'",
    "＇": "'",
    "–": "-", "—": "-", "−": "-", "‐": "-", "‑": "-",
    " ": " ", " ": " ", " ": " ", " ": " ", " ": " ", "　": " ",
    "​": " ", "﻿": " ",
    "×": "*", "∗": "*", "÷": "/",
    "；": ";", "，": ",", "（": "(", "）": ")", "｛": "{", "｝": "}",
    "＝": "=", "＜": "<", "＞": ">",
    "≤": "<=", "≥": ">=", "≠": "!=",
})


def normalise(code):
    """Line endings and punctuation. One character in, one character out (except <=, >=, !=)."""
    if code.startswith("﻿"):
        code = code[1:]
    return code.replace("\r\n", "\n").replace("\r", "\n").translate(_PUNCT)


def strip_comments(code):
    """Replace comments with spaces. Newlines inside comments stay; literals are left alone."""
    out = []
    i, n = 0, len(code)
    while i < n:
        ch = code[i]
        nxt = code[i + 1] if i + 1 < n else ""
        if ch == "/" and nxt == "/":
            while i < n and code[i] != "\n":
                out.append(" ")
                i += 1
        elif ch == "/" and nxt == "*":
            end = code.find("*/", i + 2)
            end = n if end < 0 else end + 2
            out.append("".join("\n" if c == "\n" else " " for c in code[i:end]))
            i = end
        elif ch in "\"'":
            j = i + 1
            while j < n and code[j] != ch and code[j] != "\n":
                j += 2 if code[j] == "\\" else 1
            j = min(j + 1, n)
            out.append(code[i:j])
            i = j
        else:
            out.append(ch)
            i += 1
    return "".join(out)


_DEFINE = re.compile(r"\s*#\s*define\s+([A-Za-z_]\w*)(.*)$", re.S)
_INCLUDE = re.compile(r"\s*#\s*include\b")
_DIRECTIVE = re.compile(r"\s*#\s*(\w*)")
_CODE_OR_LITERAL = re.compile(r'("(?:[^"\\\n]|\\.)*"?|\'(?:[^\'\\\n]|\\.)*\'?)')


def _substitute(line, macros):
    """Replace macro names outside string and char literals."""
    if not macros:
        return line
    pieces = _CODE_OR_LITERAL.split(line)
    for k in range(0, len(pieces), 2):              # even pieces are code, odd ones literals
        piece = pieces[k]
        for _ in range(8):                          # a macro body may name another macro
            before = piece
            for name, (pattern, body) in macros.items():
                if name in piece:
                    piece = pattern.sub(lambda _m, b=body: b, piece)
            if piece == before:
                break
        pieces[k] = piece
    return "".join(pieces)


def preprocess(code):
    """Return text that pycparser can parse, with the same line numbers as `code`."""
    lines = strip_comments(normalise(code)).split("\n")
    macros = {}
    for index, line in enumerate(lines):
        if not line.lstrip().startswith("#"):
            lines[index] = _substitute(line, macros)
            continue
        number = index + 1
        if _INCLUDE.match(line):
            lines[index] = ""
            continue
        m = _DEFINE.match(line)
        if m:
            name, body = m.group(1), m.group(2)
            if body.startswith("("):
                raise SourceError("unsupported", number, "macros with arguments are not supported",
                                  "#define " + name + "()")
            body = body.strip()
            if not body:
                raise SourceError("unsupported", number, "#define needs a value", "#define " + name)
            macros[name] = (re.compile(r"\b" + re.escape(name) + r"\b"), _substitute(body, macros))
            lines[index] = ""
            continue
        word = "#" + _DIRECTIVE.match(line).group(1)
        raise SourceError("unsupported", number, f"`{word}` is not supported", word)
    return "\n".join(lines)


def brace_blocks(text):
    """Top-level `{ ... }` blocks of preprocessed text as (open_line, close_line), in order.
    Used to find the line of each function's closing brace."""
    blocks, depth, line, start = [], 0, 1, 0
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch == "\n":
            line += 1
        elif ch in "\"'":
            j = i + 1
            while j < n and text[j] != ch and text[j] != "\n":
                j += 2 if text[j] == "\\" else 1
            line += text.count("\n", i, j + 1)
            i = j
        elif ch == "{":
            if depth == 0:
                start = line
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                blocks.append((start, line))
            depth = max(depth, 0)
        i += 1
    return blocks
