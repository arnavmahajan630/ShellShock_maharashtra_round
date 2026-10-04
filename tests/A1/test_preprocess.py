"""preprocess.py: ml_plan/03 §2.5 (comments become spaces, newlines stay) and gate G8."""
import pytest
from pycparser import c_parser

from ml.c_interp.preprocess import SourceError, brace_blocks, normalise, preprocess, strip_comments


def test_line_comment_becomes_spaces():
    src = "int x = 1; // set x\nint y = 2;"
    out = strip_comments(src)
    assert out == "int x = 1;         \nint y = 2;"
    assert len(out) == len(src)


def test_block_comment_keeps_every_newline():
    src = "int a; /* one\n two\n three */ int b;\nint c;"
    out = strip_comments(src)
    assert out.count("\n") == src.count("\n") and len(out) == len(src)
    assert out.split("\n")[2].strip() == "int b;" and "two" not in out


def test_comment_markers_inside_literals_are_kept():
    src = 'printf("// not a comment /* nor this */"); char c = \'/\'; // real'
    out = strip_comments(src)
    assert '"// not a comment /* nor this */"' in out and "real" not in out and "'/'" in out


def test_unterminated_block_comment_swallows_the_rest():
    assert strip_comments("int a; /* open\nint b;").strip() == "int a;"


def test_division_is_not_a_comment():
    assert strip_comments("x = a / b / c;") == "x = a / b / c;"


def test_commented_code_parses_with_original_line_numbers():
    src = ("// header\n/* block\n   comment */\nint f(int n) { // open\n    /* a */ int t = 0; /* b */\n"
           "    return t; // done\n}\n")
    with pytest.raises(c_parser.ParseError):
        c_parser.CParser().parse(src)                  # pycparser 3.0 refuses comments
    ast = c_parser.CParser().parse(preprocess(src))
    func = ast.ext[0]
    assert func.decl.coord.line == 4
    assert [item.coord.line for item in func.body.block_items] == [5, 6]


def test_include_lines_are_blanked():
    out = preprocess("#include <stdio.h>\n  #  include \"x.h\"\nint a;")
    assert out == "\n\nint a;"


def test_define_is_text_substitution():
    out = preprocess("#define N 5\n#define LIMIT (N + 1)\nint a[N];\nint NN = LIMIT;\nchar s[] = \"N\";")
    assert out.split("\n") == ["", "", "int a[5];", "int NN = (5 + 1);", 'char s[] = "N";']


def test_define_applies_only_after_its_line():
    out = preprocess("int a = N;\n#define N 5\nint b = N;")
    assert out.split("\n") == ["int a = N;", "", "int b = 5;"]


@pytest.mark.parametrize("src,line", [
    ("#pragma once\nint a;", 1),
    ("int a;\n#ifdef X\n#endif", 2),
    ("int a;\n\n#undef N", 3),
    ("#define MAX(a, b) ((a) > (b) ? (a) : (b))", 1),
    ("#define EMPTY", 1),
])
def test_other_directives_are_rejected(src, line):
    with pytest.raises(SourceError) as info:
        preprocess(src)
    assert info.value.status == "unsupported" and info.value.line == line


def test_hash_inside_a_comment_or_string_is_fine():
    assert preprocess("/* #pragma */ int a;\n// #error\nchar s[] = \"#x\";").count("#") == 1


def test_smart_quotes_and_dashes_are_normalised():
    assert normalise("printf(“hi”); char c = ‘x’; a – b; a + b") == \
        "printf(\"hi\"); char c = 'x'; a - b; a + b"
    assert normalise("a\r\nb\rc") == "a\nb\nc"
    assert normalise("﻿int a;") == "int a;"
    assert normalise("a ≤ b") == "a <= b"


def test_brace_blocks_find_function_ends():
    text = "int g[2] = {1, 2};\nint f() {\n    if (1) {\n        char c = '}';\n    }\n    return \"}\"[0];\n}\nvoid h() { }"
    assert brace_blocks(text) == [(1, 1), (2, 7), (8, 8)]
