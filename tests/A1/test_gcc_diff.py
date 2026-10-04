"""Differential check against gcc (ml_plan/03 §2.5).

Each program is defined behaviour in C: its trace has none of uninit_read, oob_read,
overflow, str_literal_compare. gcc compiles and runs it; the interpreter must print the
same text and return the same exit value.
"""
import shutil
import subprocess

import pytest

from ml.c_interp import harness

from .util import problem

GCC = shutil.which("gcc")
UNDEFINED = {"uninit_read", "oob_read", "oob_write", "overflow", "str_literal_compare", "array_compare",
             "missing_return", "div_zero", "step_cap_hit", "depth_cap_hit"}

PROGRAMS = {
    "arithmetic": r"""
#include <stdio.h>
int main() {
    int a = 17, b = -5;
    printf("%d %d %d %d\n", a / b, a % b, -a / 5, -a % 5);
    printf("%d %d %d\n", a + b * 2, (a + b) * 2, a - b - 3);
    int c = 3;
    c += 4; c -= 1; c *= 3; c /= 4; c %= 3;
    printf("%d\n", c);
    int i = 5;
    int j = i++ + 2;
    int k = ++i * 2;
    int m = i-- - 1;
    int q = --i;
    printf("%d %d %d %d %d\n", i, j, k, m, q);
    printf("%d %d %d %d\n", 7 & 3, 7 | 8, 7 ^ 2, 1 << 4);
    printf("%d %d\n", -7 / 2, -7 % 2);
    printf("%d %d %d\n", 0x1F, 010, 'a' - 'A');
    return 3;
}
""",
    "floats": r"""
#include <stdio.h>
double mean(int t[], int n) {
    int s = 0;
    for (int i = 0; i < n; i++) s += t[i];
    return (double)s / n;
}
int main() {
    int t[4] = {1, 2, 4, 8};
    double m = mean(t, 4);
    float h = 7 / 2;
    float g = 7 / 2.0f;
    double p = (double)(7 / 2);
    int back = m * 2;
    int neg = -2.75;
    printf("%.2f %.1f %.1f %.3f %d %d\n", m, h, g, p, back, neg);
    printf("%f %lf %.0f|%8.3f|%-8.2f|\n", m, 0.5, 3.75 * 4, m, m);
    double x = 10;
    x /= 4; x += 1; x *= 3; x -= 0.5;
    printf("%.4f %d %d\n", x, x > 10, 3 < 2.5);
    printf("%.3f %.3f\n", 1 / 3.0, (float)1 / 3);
    return 0;
}
""",
    "control_flow": r"""
#include <stdio.h>
int main() {
    int t = 0;
    for (int i = 0; i < 10; i++) {
        if (i == 7) break;
        if (i % 2 == 0) continue;
        t += i;
    }
    printf("%d\n", t);
    int n = 5;
    while (n > 0) { printf("%d ", n); n -= 2; }
    printf("\n");
    int k = 0;
    do { k += 3; } while (k < 10);
    printf("%d\n", k);
    for (int i = 0, j = 5; i < j; i++, j--) printf("%d%d ", i, j);
    printf("\n");
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            if (j > i) break;
            printf("%d", i * j);
        }
        printf(";");
    }
    printf("\n");
    int x = 15;
    if (x < 10) printf("low\n"); else if (x < 20) printf("mid\n"); else printf("high\n");
    printf("%d %d %d\n", x > 10 ? 1 : 2, x > 10 && x < 12, x < 10 || x == 15);
    printf("%d %d\n", 0 < x < 10, x == 42 || 7);
    return t;
}
""",
    "arrays_and_functions": r"""
#include <stdio.h>
#define N 6
void bubble(int a[], int n) {
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - 1 - i; j++) {
            if (a[j] > a[j + 1]) {
                int t = a[j];
                a[j] = a[j + 1];
                a[j + 1] = t;
            }
        }
    }
}
void bump(int x) { x += 100; }
int total(int a[], int n) {
    int s = 0;
    int i = 0;
    while (i < n) { s += a[i]; i++; }
    return s;
}
int main() {
    int v[N] = {5, 3, 9, 1, 7};      /* the last cell is zero */
    int w[] = {4, 4, 4};
    int z = 1;
    bump(z);
    bubble(v, N);
    for (int i = 0; i < N; i++) printf("%d ", v[i]);
    printf("\n%d %d %d\n", total(v, N), total(w, 3), z);
    printf("%d %d\n", (int)(sizeof(v) / sizeof(v[0])), (int)sizeof(w));
    v[2] += 10; v[3]++; --v[4]; v[5] *= 2;
    printf("%d %d %d %d\n", v[2], v[3], v[4], v[5]);
    return 0;
}
""",
    "recursion": r"""
#include <stdio.h>
int fact(int n) { if (n <= 1) return 1; return n * fact(n - 1); }
int fib(int n) { return n < 2 ? n : fib(n - 1) + fib(n - 2); }
int gcd(int a, int b) { if (b == 0) return a; return gcd(b, a % b); }
int sum(int a[], int n) { if (n == 0) return 0; return a[n - 1] + sum(a, n - 1); }
int digits(int n) { if (n < 10) return n; return n % 10 + digits(n / 10); }
int is_even(int n);
int is_odd(int n) { if (n == 0) return 0; return is_even(n - 1); }
int is_even(int n) { if (n == 0) return 1; return is_odd(n - 1); }
void countdown(int n) { if (n == 0) { printf("go\n"); return; } printf("%d ", n); countdown(n - 1); }
int main() {
    int a[5] = {1, 2, 3, 4, 5};
    printf("%d %d %d %d %d %d\n", fact(10), fib(12), gcd(84, 36), sum(a, 5), digits(98765), is_even(9));
    countdown(4);
    return fact(4);
}
""",
    "strings_and_chars": r"""
#include <stdio.h>
#include <string.h>
int count_vowels(char s[]) {
    int c = 0;
    for (int i = 0; s[i] != '\0'; i++) {
        if (s[i] == 'a' || s[i] == 'e' || s[i] == 'i' || s[i] == 'o' || s[i] == 'u') c++;
    }
    return c;
}
int is_palindrome(char s[]) {
    int n = 0;
    while (s[n] != '\0') n++;
    for (int i = 0; i < n / 2; i++) {
        if (s[i] != s[n - 1 - i]) return 0;
    }
    return 1;
}
int main() {
    char s[] = "level";
    char t[10] = "abc";
    char u[] = {'h', 'i', '\0'};
    printf("%s %s %s %d %d\n", s, t, u, (int)strlen(s), (int)strlen("space ship"));
    printf("%d %d %d\n", count_vowels("education"), is_palindrome(s), is_palindrome(t));
    t[0] = 'x'; t[3] = 'd'; t[4] = '\0';
    printf("%s %c %d %c\n", t, t[1], t[1], s[0] + 1);
    char c = 'a';
    c = c + 2;
    printf("%c%c%c %d\n", c, 'A' + 25, '0' + 7, c - 'a');
    printf("tab\there \"quoted\" back\\slash 100%%\n");
    printf("%5d|%-5d|%05d|%x|%s|%c\n", 42, 42, 42, 255, "ok", 'z');
    return 0;
}
""",
    "scopes_and_globals": r"""
#include <stdio.h>
int counter = 0;
int table[4] = {1, 2};
int next() { counter++; return counter; }
int main() {
    int x = 1;
    {
        int x = 2;
        x += 5;
        printf("%d ", x);
    }
    for (int x = 10; x < 12; x++) printf("%d ", x);
    printf("%d\n", x);
    int a = next();
    int b = next();
    int c = (next(), next());
    table[3] = a + b + c;
    printf("%d %d %d %d %d %d\n", a, b, c, counter, table[1] + table[2], table[3]);
    int k = 0;
    int hits = (k > 0 && next() > 0) + (k == 0 || next() > 0);
    printf("%d %d\n", hits, counter);
    return 0;
}
""",
    "problem_style": r"""
#include <stdio.h>
int total_energy(int cells[], int n) {
    int total = 0;
    for (int i = 0; i < n; i++) {
        total += cells[i];
    }
    return total;
}
int binary_search(int a[], int n, int x) {
    int low = 0, high = n - 1;
    while (low <= high) {
        int mid = (low + high) / 2;
        if (a[mid] == x) return mid;
        if (a[mid] < x) low = mid + 1; else high = mid - 1;
    }
    return -1;
}
void reverse(int a[], int n) {
    for (int i = 0; i < n / 2; i++) {
        int t = a[i];
        a[i] = a[n - 1 - i];
        a[n - 1 - i] = t;
    }
}
int max_of_three(int a, int b, int c) {
    int m = a;
    if (b > m) m = b;
    if (c > m) m = c;
    return m;
}
float fuel_percent(int fuel, int cap) { return (float)fuel / cap * 100; }
int main() {
    int a[7] = {1, 3, 5, 7, 9, 11, 13};
    printf("%d %d %d %d\n", total_energy(a, 7), binary_search(a, 7, 11), binary_search(a, 7, 4), binary_search(a, 7, 1));
    reverse(a, 7);
    for (int i = 0; i < 7; i++) printf("%d,", a[i]);
    printf("\n%d %.2f\n", max_of_three(3, 9, 4), fuel_percent(1, 4));
    return 0;
}
""",
    "conversions_and_formats": r"""
#include <stdio.h>
int main() {
    int x = 7;
    x *= 1.5;
    int y = -9;
    y /= 2;
    int z = 10;
    z -= 2.7;
    double d = 5;
    d /= 2;
    int w = d * 3 + 0.9;
    printf("%d %d %d %.2f %d\n", x, y, z, d, w);
    int a[5] = {0, 0, 0, 0, 0};
    int i = 0;
    a[i++] = 10;
    a[i++] = 20;
    a[++i] = 30;
    a[i--] += 5;
    printf("%d %d %d %d %d %d\n", a[0], a[1], a[2], a[3], a[4], i);
    int n = 3;
    int r = n > 2 ? n > 5 ? 1 : 2 : 3;
    printf("%d %d %d\n", r, (1, 2, 3), !5 + !0);
    printf("%d %d %d\n", 5 == 5.0, 3 / 2 * 2.0 == 2.0, 3 / 2.0 * 2 == 3);
    char c = 'x';
    int ci = c + 1;
    printf("%d %c %d\n", ci, ci, 'a' < 'b');
    int big = 1000000;
    printf("%d %d\n", big * 2, big / 3 * 3 + big % 3);
    float f = 0.1f + 0.2f;
    printf("%.1f %.2f %d\n", f, f * 10, (int)(f * 10));
    printf("%d\n", (int)3.99 + (int)-3.99 + (int)0.5);
    printf("%i|%3d|%-3d|%03d|%+d\n", 5, 5, 5, 5, 5);
    printf("%6.2f|%-8.3f|%08.2f|%.0f\n", 3.14159, 2.5, 2.5, 7.0);
    printf("%x %X %o %u\n", 255, 255, 8, 42);
    printf("%c|%5s|%-5s|%.2s\n", 'q', "ab", "ab", "abcdef");
    return 0;
}
""",
    "more_loops": r"""
#include <stdio.h>
int collatz(int n) { int s = 0; while (n != 1) { if (n % 2 == 0) n /= 2; else n = 3 * n + 1; s++; } return s; }
int power(int b, int k) { int r = 1; for (int i = 0; i < k; i++) r *= b; return r; }
int main() {
    printf("%d %d %d\n", collatz(27), power(2, 10), power(-3, 3));
    int i = 0, t = 0;
    do { i++; if (i == 3) continue; if (i > 6) break; t += i; } while (i < 100);
    printf("%d %d\n", i, t);
    int count = 0;
    for (int a = 0; a < 5; a++) { for (int b = a; b < 5; b++) { if ((a + b) % 3 == 0) continue; count++; } }
    printf("%d\n", count);
    int k = 10;
    while (k--) { if (k == 4) break; }
    printf("%d\n", k);
    for (k = 0; k < 3; ) { k++; }
    printf("%d\n", k);
    int m = 0;
    for (;;) { m += 2; if (m > 7) break; }
    printf("%d\n", m);
    int arr[6] = {3, -1, 4, -1, 5, -9};
    int best = arr[0], pos = 0;
    for (int j = 1; j < 6; j++) if (arr[j] > best) { best = arr[j]; pos = j; }
    printf("%d %d\n", best, pos);
    return 0;
}
""",
    "dsa_style": r"""
#include <stdio.h>
void selection_sort(int a[], int n) {
    for (int i = 0; i < n - 1; i++) {
        int min = i;
        for (int j = i + 1; j < n; j++) if (a[j] < a[min]) min = j;
        int t = a[i]; a[i] = a[min]; a[min] = t;
    }
}
void rotate_left(int a[], int n) { int first = a[0]; for (int i = 0; i < n - 1; i++) a[i] = a[i + 1]; a[n - 1] = first; }
int second_largest(int a[], int n) {
    int m1 = a[0], m2 = -1000000;
    for (int i = 1; i < n; i++) { if (a[i] > m1) { m2 = m1; m1 = a[i]; } else if (a[i] > m2) m2 = a[i]; }
    return m2;
}
int pair_sum_exists(int a[], int n, int target) {
    int i = 0, j = n - 1;
    while (i < j) { int s = a[i] + a[j]; if (s == target) return 1; if (s < target) i++; else j--; }
    return 0;
}
int count_guesses(int secret, int lo, int hi) {
    int g = 0;
    while (lo <= hi) { int mid = (lo + hi) / 2; g++; if (mid == secret) return g; if (mid < secret) lo = mid + 1; else hi = mid - 1; }
    return g;
}
int sum_digits(int n) { if (n < 10) return n; return n % 10 + sum_digits(n / 10); }
int main() {
    int a[8] = {8, 3, 5, 1, 9, 2, 7, 4};
    selection_sort(a, 8);
    for (int i = 0; i < 8; i++) printf("%d ", a[i]);
    rotate_left(a, 8);
    printf("| %d %d | %d %d %d %d %d\n", a[0], a[7], second_largest(a, 8), pair_sum_exists(a, 7, 11),
           pair_sum_exists(a, 7, 100), count_guesses(37, 1, 100), sum_digits(90817));
    return 0;
}
""",
    "more_strings": r"""
#include <stdio.h>
#include <string.h>
int str_length(char s[]) { int n = 0; while (s[n] != '\0') n++; return n; }
void upper(char s[]) { for (int i = 0; s[i]; i++) if (s[i] >= 'a' && s[i] <= 'z') s[i] = s[i] - 'a' + 'A'; }
int digits_value(char s[]) { int v = 0; for (int i = 0; s[i] != '\0'; i++) v = v * 10 + (s[i] - '0'); return v; }
int main() {
    char s[] = "deep space";
    char e[] = "";
    char n[8] = "4096";
    upper(s);
    printf("%s|%s|%d %d %d|%d\n", s, e, str_length(s), str_length(e), (int)strlen(n), digits_value(n));
    printf("%d %d\n", (int)sizeof(s), (int)sizeof(n));
    char line[4];
    line[0] = 'o'; line[1] = 'k'; line[2] = '\0';
    printf("[%s] %c%c\n", line, line[0], "xyz"[1]);
    return str_length("hello" " world");
}
""",
}


@pytest.mark.skipif(GCC is None, reason="gcc is not on PATH")
@pytest.mark.parametrize("name", sorted(PROGRAMS))
def test_same_output_as_gcc(name, tmp_path):
    source = PROGRAMS[name]
    c_file = tmp_path / f"{name}.c"
    exe = tmp_path / f"{name}.exe"
    c_file.write_text(source, encoding="utf-8")
    build = subprocess.run([GCC, "-std=c99", "-O0", "-w", str(c_file), "-o", str(exe)],
                           capture_output=True, text=True, timeout=120)
    assert build.returncode == 0, build.stderr
    real = subprocess.run([str(exe)], capture_output=True, text=True, timeout=30)

    out = harness.trace(problem("int main()", [[]]), source)
    assert out["status"] == "ok"
    assert not [e for e in out["events"] if e["type"] in UNDEFINED], "program is not defined behaviour"
    assert out["printed"] == real.stdout
    assert out["returned"] == real.returncode
