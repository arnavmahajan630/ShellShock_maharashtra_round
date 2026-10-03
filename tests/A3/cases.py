"""Programs whose result the C standard defines, with the result each test expects.

Every case is used twice:
  - test_expectations_gcc.py / test_strings.py run it through the gcc backend (package A2),
    so the expected values are what a real C compiler produces, not only a reading of 03;
  - test_subset_defined.py / test_strings.py run it through the interpreter.

Nothing here reads uninitialised memory, goes out of bounds, overflows an int or compares a
string literal: 03 §2.5 excludes those from the gcc comparison.

`expect` uses the keys of 03 §2.3: returned, printed, array0, max_depth_le and world-effect
counts (fire, launch, door_open, door_closed, scan).
"""
from collections import namedtuple

from tests.A3.helpers import src

Case = namedtuple("Case", "id spec signature code tests")

CASES = [
    # ------------------------------------------------------------ 03 §2.1 arithmetic and types
    Case("arith_precedence", "03 §2.1: + - * / %", "int f(int a, int b, int c)", src("""
        int f(int a, int b, int c) {
            return a + b * c - a / b % c;
        }"""), [([7, 2, 3], {"returned": 13}), ([10, 3, 4], {"returned": 19}), ([-9, 2, 5], {"returned": 5})]),

    Case("intdiv_truncates_toward_zero", "03 §2.2 row 4: int / int", "int f(int a, int b)", src("""
        int f(int a, int b) {
            int q = a / b;
            return q;
        }"""), [([7, 2], {"returned": 3}), ([-7, 2], {"returned": -3}), ([7, -2], {"returned": -3}),
                ([6, 3], {"returned": 2}), ([1, 2], {"returned": 0}), ([-1, 2], {"returned": 0})]),

    Case("modulo_sign_follows_dividend", "03 §2.1: %", "int f(int a, int b)", src("""
        int f(int a, int b) {
            return a % b;
        }"""), [([7, 3], {"returned": 1}), ([-7, 3], {"returned": -1}), ([7, -3], {"returned": 1}),
                ([6, 3], {"returned": 0})]),

    Case("intdiv_into_float_variable", "03 §2.2 row 4: int / int assigned to float", "float f(int a, int b)", src("""
        float f(int a, int b) {
            float r = a / b;
            return r;
        }"""), [([7, 2], {"returned": 3.0}), ([1, 2], {"returned": 0.0}), ([9, 3], {"returned": 3.0})]),

    Case("intdiv_returned_as_float", "03 §2.2 row 4: int / int returned as float (M04 shape)",
         "float avg_fuel(int tanks[], int n)", src("""
        float avg_fuel(int tanks[], int n) {
            int total = 0;
            for (int i = 0; i < n; i++) {
                total += tanks[i];
            }
            return total / n;
        }"""), [([[1, 2], 2], {"returned": 1.0}), ([[4, 4, 5], 3], {"returned": 4.0}), ([[3], 1], {"returned": 3.0})]),

    Case("cast_makes_float_division", "03 §2.1: casts (float)", "float f(int a, int b)", src("""
        float f(int a, int b) {
            return (float) a / b;
        }"""), [([7, 2], {"returned": 3.5}), ([1, 4], {"returned": 0.25}), ([-7, 2], {"returned": -3.5})]),

    Case("float_literal_makes_float_division", "03 §2.1: float arithmetic", "float f(int a)", src("""
        float f(int a) {
            return a / 2.0;
        }"""), [([7], {"returned": 3.5}), ([-3], {"returned": -1.5})]),

    Case("cast_to_int_truncates", "03 §2.1: casts (int) (double)", "int f(float x)", src("""
        int f(float x) {
            double d = (double) x;
            return (int) d + (int) x;
        }"""), [([3.7], {"returned": 6}), ([-3.7], {"returned": -6}), ([0.5], {"returned": 0})]),

    Case("float_assigned_to_int_truncates", "03 §2.1: int and float variables", "int f(float x)", src("""
        int f(float x) {
            int k = x * 2;
            return k;
        }"""), [([1.8], {"returned": 3}), ([-1.8], {"returned": -3}), ([2.0], {"returned": 4})]),

    Case("float_returned_from_int_function", "03 §2.1: int and float", "int f(int a)", src("""
        int f(int a) {
            return a * 0.5;
        }"""), [([5], {"returned": 2}), ([-5], {"returned": -2}), ([8], {"returned": 4})]),

    Case("float_arithmetic", "03 §2.1: float, double (as float)", "float f(float x, int n)", src("""
        float f(float x, int n) {
            double t = x * n;
            t = t + 0.5;
            return t / 2;
        }"""), [([1.5, 3], {"returned": 2.5}), ([0.25, 2], {"returned": 0.5})]),

    Case("char_is_a_small_int", "03 §2.1: char (as int, char literals)", "int f(int n)", src("""
        int f(int n) {
            char c = 'a';
            c = c + n;
            return c * 1000 + ('z' - 'a') + '0' + '\\n' + '\\0';
        }"""), [([0], {"returned": 97083}), ([2], {"returned": 99083})]),

    Case("compound_assignment", "03 §2.1: += -= *= /= %=", "int f(int x)", src("""
        int f(int x) {
            x += 3;
            x -= 1;
            x *= 4;
            x /= 3;
            x %= 5;
            return x;
        }"""), [([2], {"returned": 0}), ([4], {"returned": 3}), ([-10], {"returned": 0})]),

    Case("pre_and_post_increment", "03 §2.1: ++/-- (pre and post)", "int f(int i)", src("""
        int f(int i) {
            int a = i++;
            int b = ++i;
            int c = i--;
            int d = --i;
            return a * 1000 + b * 100 + c * 10 + d;
        }"""), [([1], {"returned": 1331}), ([5], {"returned": 5775})]),

    Case("relational_and_logical", "03 §2.1: relational, && || !", "int f(int a, int b)", src("""
        int f(int a, int b) {
            return (a < b) + (a <= b) * 2 + (a > b) * 4 + (a >= b) * 8 + (a == b) * 16
                 + (a != b) * 32 + (!a) * 64 + (a && b) * 128 + (a || b) * 256;
        }"""), [([1, 2], {"returned": 419}), ([2, 2], {"returned": 410}), ([0, 0], {"returned": 90}),
                ([3, 0], {"returned": 300})]),

    Case("and_short_circuits", "03 §2.1: && short-circuit (right operand not evaluated)", "int f(int n)", src("""
        int f(int n) {
            if (n != 0 && 10 / n > 1) {
                return 1;
            }
            return 0;
        }"""), [([0], {"returned": 0}), ([2], {"returned": 1}), ([20], {"returned": 0})]),

    Case("short_circuit_skips_side_effects", "03 §2.1: && || short-circuit", "int f(int n)", src("""
        int bump(int v) {
            fire();
            return v;
        }
        int f(int n) {
            int r = 0;
            if (n > 0 || bump(0)) {
                r += 1;
            }
            if (n > 0 && bump(1)) {
                r += 2;
            }
            return r;
        }"""), [([5], {"returned": 3, "fire": 1}), ([0], {"returned": 0, "fire": 1})]),

    Case("ternary", "03 §2.1: ternary", "int f(int a, int b)", src("""
        int f(int a, int b) {
            int big = a > b ? a : b;
            return big == a ? big * 2 : big;
        }"""), [([3, 9], {"returned": 9}), ([9, 3], {"returned": 18}), ([4, 4], {"returned": 8})]),

    Case("unary_minus_and_not", "03 §2.1: unary operators", "int f(int a)", src("""
        int f(int a) {
            return -a + !a * 10 + !!a;
        }"""), [([4], {"returned": -3}), ([0], {"returned": 10}), ([-2], {"returned": 3})]),

    Case("chained_assignment", "03 §2.1: assignment is an expression", "int f(int n)", src("""
        int f(int n) {
            int a;
            int b;
            a = b = n + 1;
            return a + b;
        }"""), [([1], {"returned": 4}), ([-1], {"returned": 0})]),

    # ------------------------------------------------------------ 03 §2.1 control flow
    Case("if_else_chain", "03 §2.1: if/else (P12 shield_mode)", "int shield_mode(int energy)", src("""
        int shield_mode(int energy) {
            if (energy < 30) {
                return 0;
            } else if (energy < 70) {
                return 1;
            } else {
                return 2;
            }
        }"""), [([10], {"returned": 0}), ([30], {"returned": 1}), ([69], {"returned": 1}), ([70], {"returned": 2})]),

    Case("nested_if", "03 §2.1: if/else (P17 max_of_three)", "int max_of_three(int a, int b, int c)", src("""
        int max_of_three(int a, int b, int c) {
            int best = a;
            if (b > best) best = b;
            if (c > best) best = c;
            return best;
        }"""), [([1, 2, 3], {"returned": 3}), ([3, 2, 1], {"returned": 3}), ([2, 9, 4], {"returned": 9}),
                ([-5, -2, -9], {"returned": -2})]),

    Case("while_loop", "03 §2.1: while (P05 charge_steps)", "int charge_steps(int level, int target)", src("""
        int charge_steps(int level, int target) {
            int steps = 0;
            while (level < target) {
                level += 7;
                steps++;
            }
            return steps;
        }"""), [([0, 21], {"returned": 3}), ([5, 5], {"returned": 0}), ([0, 22], {"returned": 4})]),

    Case("do_while_runs_once", "03 §2.1: do-while", "int f(int n)", src("""
        int f(int n) {
            int c = 0;
            do {
                c++;
                n--;
            } while (n > 0);
            return c;
        }"""), [([0], {"returned": 1}), ([3], {"returned": 3}), ([-4], {"returned": 1})]),

    Case("break_and_continue", "03 §2.1: break, continue", "int f(int n)", src("""
        int f(int n) {
            int s = 0;
            for (int i = 0; i < n; i++) {
                if (i == 2) continue;
                if (i == 5) break;
                s += i;
            }
            return s;
        }"""), [([10], {"returned": 8}), ([2], {"returned": 1}), ([4], {"returned": 4})]),

    Case("break_leaves_only_the_inner_loop", "03 §2.1: break in nested loops", "int f(int n)", src("""
        int f(int n) {
            int c = 0;
            int i = 0;
            while (i < n) {
                for (int j = 0; j < 10; j++) {
                    if (j == 2) break;
                    c++;
                }
                i++;
            }
            return c;
        }"""), [([3], {"returned": 6}), ([0], {"returned": 0})]),

    Case("nested_for_loops", "03 §2.1: for (C99 declaration in init)", "int f(int n)", src("""
        int f(int n) {
            int c = 0;
            for (int i = 0; i < n; i++)
                for (int j = i + 1; j < n; j++)
                    c++;
            return c;
        }"""), [([4], {"returned": 6}), ([1], {"returned": 0}), ([5], {"returned": 10})]),

    Case("two_loops_reuse_the_same_counter_name", "03 §2.1: for (C99 declaration in init)", "int f(int n)", src("""
        int f(int n) {
            int s = 0;
            for (int i = 0; i < n; i++) s += i;
            for (int i = n; i > 0; i--) s += 10;
            return s;
        }"""), [([3], {"returned": 33}), ([0], {"returned": 0})]),

    Case("block_scope_shadowing", "03 §2.1: blocks", "int f(int n)", src("""
        int f(int n) {
            int x = 1;
            {
                int x = 5;
                n += x;
            }
            return n + x;
        }"""), [([1], {"returned": 7})]),

    Case("early_return_from_loop", "03 §2.1: return (Q01 linear_search)", "int linear_search(int a[], int n, int x)", src("""
        int linear_search(int a[], int n, int x) {
            for (int i = 0; i < n; i++) {
                if (a[i] == x) {
                    return i;
                }
            }
            return -1;
        }"""), [([[4, 8, 15], 3, 15], {"returned": 2}), ([[4, 8, 15], 3, 4], {"returned": 0}),
                ([[4, 8, 15], 3, 9], {"returned": -1}), ([[], 0, 9], {"returned": -1})]),

    Case("binary_search", "03 §2.1: while + int division (Q03)", "int binary_search(int a[], int n, int x)", src("""
        int binary_search(int a[], int n, int x) {
            int low = 0;
            int high = n - 1;
            while (low <= high) {
                int mid = (low + high) / 2;
                if (a[mid] == x) return mid;
                if (a[mid] < x) low = mid + 1;
                else high = mid - 1;
            }
            return -1;
        }"""), [([[1, 3, 5, 7, 9], 5, 7], {"returned": 3}), ([[1, 3, 5, 7, 9], 5, 1], {"returned": 0}),
                ([[1, 3, 5, 7, 9], 5, 4], {"returned": -1}), ([[2], 1, 2], {"returned": 0})]),

    # ------------------------------------------------------------ 03 §2.1 arrays and functions
    Case("local_arrays_and_initialisers", "03 §2.1: 1D arrays, array initialisers", "int f(int k)", src("""
        int f(int k) {
            int a[5] = {5, 4, 3, 2, 1};
            int b[] = {10, 20, 30};
            int c[3];
            c[0] = 7;
            c[1] = a[k];
            c[2] = b[k];
            return c[0] + c[1] + c[2];
        }"""), [([1], {"returned": 31}), ([2], {"returned": 40})]),

    Case("short_initialiser_zero_fills", "03 §2.1: array initialisers (C99 6.7.8: the rest is zero)", "int f(int k)", src("""
        int f(int k) {
            int a[5] = {1, 2};
            return a[k] + a[4] * 100;
        }"""), [([1], {"returned": 2}), ([3], {"returned": 0})]),

    Case("float_array", "03 §2.1: 1D arrays of float", "float f(int n)", src("""
        float f(int n) {
            float w[3] = {0.5, 1.5, 2.0};
            float s = 0;
            for (int i = 0; i < n; i++) s += w[i];
            return s;
        }"""), [([3], {"returned": 4.0}), ([2], {"returned": 2.0})]),

    Case("scalars_are_passed_by_value", "03 §2.1: user functions, scalars by value", "int f(int n)", src("""
        void bump(int x) {
            x = x + 100;
        }
        int f(int n) {
            bump(n);
            return n;
        }"""), [([5], {"returned": 5})]),

    Case("arrays_are_passed_by_reference", "03 §2.1: arrays by reference / §2.2 row 14", "int f(int a[], int n)", src("""
        void zero_first(int a[]) {
            a[0] = 0;
        }
        int f(int a[], int n) {
            zero_first(a);
            return a[0] + a[1];
        }"""), [([[5, 6], 2], {"returned": 6, "array0": [0, 6]})]),

    Case("local_array_shared_with_callee", "03 §2.1: arrays by reference", "int f(int n)", src("""
        void fill(int a[], int n, int v) {
            for (int i = 0; i < n; i++) a[i] = v;
        }
        int f(int n) {
            int a[3] = {1, 2, 3};
            fill(a, 2, n);
            return a[0] + a[1] + a[2];
        }"""), [([10], {"returned": 23})]),

    Case("in_place_bubble_sort", "03 §2.5 v3 extra: in-place sort visible to the caller", "void bubble_sort(int a[], int n)", src("""
        void bubble_sort(int a[], int n) {
            for (int i = 0; i < n - 1; i++) {
                for (int j = 0; j < n - 1 - i; j++) {
                    if (a[j] > a[j + 1]) {
                        int t = a[j];
                        a[j] = a[j + 1];
                        a[j + 1] = t;
                    }
                }
            }
        }"""), [([[3, 1, 2], 3], {"array0": [1, 2, 3]}), ([[2, 1], 2], {"array0": [1, 2]}),
                ([[1, 2, 3], 3], {"array0": [1, 2, 3]}), ([[4, 3, 2, 1], 4], {"array0": [1, 2, 3, 4]}),
                ([[5], 1], {"array0": [5]})]),

    Case("swap_without_temp_loses_a_value", "03 §2.2 row 14: array param write (D03 shape)", "void bubble_sort(int a[], int n)", src("""
        void bubble_sort(int a[], int n) {
            for (int i = 0; i < n - 1; i++) {
                for (int j = 0; j < n - 1 - i; j++) {
                    if (a[j] > a[j + 1]) {
                        a[j] = a[j + 1];
                        a[j + 1] = a[j];
                    }
                }
            }
        }"""), [([[3, 1, 2], 3], {"array0": [1, 1, 2]}), ([[4, 3, 2, 1], 4], {"array0": [1, 1, 1, 1]})]),

    Case("in_place_reverse", "03 §2.2 row 14: array param write (Q10)", "void reverse(int a[], int n)", src("""
        void reverse(int a[], int n) {
            int i = 0;
            int j = n - 1;
            while (i < j) {
                int t = a[i];
                a[i] = a[j];
                a[j] = t;
                i++;
                j--;
            }
        }"""), [([[1, 2, 3, 4], 4], {"array0": [4, 3, 2, 1]}), ([[1, 2, 3], 3], {"array0": [3, 2, 1]}),
                ([[], 0], {"array0": []})]),

    Case("helper_functions", "03 §2.2 row 18: function call / return", "int f(int n)", src("""
        int sq(int x) {
            return x * x;
        }
        int f(int n) {
            return sq(n) + sq(n + 1);
        }"""), [([2], {"returned": 13, "max_depth_le": 2}), ([0], {"returned": 1, "max_depth_le": 2})]),

    Case("recursion_factorial", "03 §2.1: recursion", "int factorial(int n)", src("""
        int factorial(int n) {
            if (n <= 1) {
                return 1;
            }
            return n * factorial(n - 1);
        }"""), [([0], {"returned": 1, "max_depth_le": 1}), ([5], {"returned": 120, "max_depth_le": 5}),
                ([12], {"returned": 479001600, "max_depth_le": 12})]),

    Case("recursion_two_calls", "03 §2.1: recursion", "int fib(int n)", src("""
        int fib(int n) {
            if (n < 2) return n;
            return fib(n - 1) + fib(n - 2);
        }"""), [([0], {"returned": 0}), ([1], {"returned": 1}), ([10], {"returned": 55, "max_depth_le": 10})]),

    Case("recursion_over_an_array", "03 §2.1: recursion (Q18 array_sum_rec)", "int array_sum_rec(int a[], int n)", src("""
        int array_sum_rec(int a[], int n) {
            if (n == 0) return 0;
            return a[n - 1] + array_sum_rec(a, n - 1);
        }"""), [([[2, 4, 6], 3], {"returned": 12, "max_depth_le": 4}), ([[], 0], {"returned": 0})]),

    # ------------------------------------------------------------ 03 §2.2 rows whose value C defines
    Case("assignment_as_condition", "03 §2.2 row 6: condition expression is an assignment", "int door_open(int code)", src("""
        int door_open(int code) {
            if (code = 42) {
                return 1;
            }
            return 0;
        }"""), [([42], {"returned": 1}), ([7], {"returned": 1}), ([0], {"returned": 1})]),

    Case("assignment_of_zero_as_condition", "03 §2.2 row 6: condition expression is an assignment", "int f(int x)", src("""
        int f(int x) {
            if (x = 0) {
                fire();
            } else {
                open_door();
            }
            return x;
        }"""), [([5], {"returned": 0, "fire": 0, "door_open": 1}), ([0], {"returned": 0, "fire": 0, "door_open": 1})]),

    Case("empty_if_body", "03 §2.2 row 7: body is EmptyStatement (if)", "void f(int shield)", src("""
        void f(int shield) {
            if (shield < 0);
            fire();
        }"""), [([10], {"fire": 1}), ([-10], {"fire": 1})]),

    Case("empty_for_body", "03 §2.2 row 7: body is EmptyStatement (for)", "int f(int n)", src("""
        int f(int n) {
            int i;
            for (i = 0; i < n; i++);
            fire();
            return i;
        }"""), [([3], {"returned": 3, "fire": 1}), ([0], {"returned": 0, "fire": 1})]),

    Case("empty_while_body", "03 §2.2 row 7: body is EmptyStatement (while)", "int f(int n)", src("""
        int f(int n) {
            int i = 0;
            while (i++ < n);
            return i;
        }"""), [([3], {"returned": 4}), ([0], {"returned": 1})]),

    Case("two_distinct_arrays_never_compare_equal", "03 §2.2 row 17: == / != on two array names", "int f(int n)", src("""
        int f(int n) {
            int a[3] = {1, 2, 3};
            int b[3] = {1, 2, 3};
            int r = 0;
            if (a == b) r += 1;
            if (a != b) r += 10;
            return r + n;
        }"""), [([0], {"returned": 10})]),

    # ------------------------------------------------------------ 03 §2.1 world builtins and printf
    Case("fire_n_times", "03 §2.1: world builtin fire() (P01)", "void fire_shots(int n)", src("""
        void fire_shots(int n) {
            for (int i = 0; i < n; i++) {
                fire();
            }
        }"""), [([3], {"fire": 3}), ([0], {"fire": 0}), ([1], {"fire": 1})]),

    Case("countdown_then_launch", "03 §2.1: printf + launch() (P02)", "void countdown(int n)", src("""
        void countdown(int n) {
            while (n > 0) {
                printf("%d\\n", n);
                n--;
            }
            launch();
        }"""), [([3], {"printed": "3\n2\n1\n", "launch": 1}), ([0], {"printed": "", "launch": 1})]),

    Case("doors_and_scan", "03 §2.1: open_door() close_door() scan(int x)", "void f(int n)", src("""
        void f(int n) {
            open_door();
            for (int i = 0; i < n; i++) {
                scan(i);
            }
            close_door();
            close_door();
        }"""), [([3], {"door_open": 1, "door_closed": 2, "scan": 3}), ([0], {"door_open": 1, "door_closed": 2, "scan": 0})]),

    Case("printf_number_formats", "03 §2.1: printf %d %i %f %.Nf %c %% %lf %ld", "void show(int n, float x)", src("""
        void show(int n, float x) {
            printf("%d %i %f %.2f %c %% %lf %ld\\n", n, n, x, x, 'q', x, n);
            printf("%.1f|%.3f|%d%%", x, x, n);
        }"""), [([7, 2.5], {"printed": "7 7 2.500000 2.50 q % 2.500000 7\n2.5|2.500|7%"}),
                ([-3, 0.375], {"printed": "-3 -3 0.375000 0.38 q % 0.375000 -3\n0.4|0.375|-3%"})]),

    Case("printf_without_arguments", "03 §2.1: printf", "int f(int n)", src("""
        int f(int n) {
            printf("ready\\n");
            printf("go");
            return n;
        }"""), [([4], {"returned": 4, "printed": "ready\ngo"})]),

    # ------------------------------------------------------------ 03 §2.5 preprocessing
    Case("define_and_include", "03 §2.5: #define NAME literal, #include stripped", "int f(int n)", src("""
        #include <stdio.h>
        #define LIMIT 3
        #define STEP 2
        int f(int n) {
            int s = 0;
            for (int i = 0; i < LIMIT; i++) {
                s += STEP;
            }
            return s + n;
        }"""), [([1], {"returned": 7}), ([0], {"returned": 6})]),

    Case("comments_are_ignored", "03 §2.5: // and /* */ comments become spaces", "int f(int n)", src("""
        // returns n doubled; return 0;
        int f(int n) {
            /* a block comment
               over two lines: n = 0; */
            int twice = n * 2;   // trailing: twice = 99;
            return twice; /* inline */
        }"""), [([4], {"returned": 8}), ([-1], {"returned": -2})]),
]

STRING_CASES = [
    Case("string_literal_init_has_terminator", "03 §2.2 row 16 / §2.5 v3 extra 1", "int f(int n)", src("""
        int f(int n) {
            char s[] = "level";
            int i = 0;
            while (s[i] != '\\0') {
                i++;
            }
            return i + n;
        }"""), [([0], {"returned": 5})]),

    Case("string_literal_cells_are_char_codes", "03 §2.2 row 16 / §2.5 v3 extra 1", "int f(int k)", src("""
        int f(int k) {
            char s[] = "ab";
            return s[k];
        }"""), [([0], {"returned": 97}), ([1], {"returned": 98}), ([2], {"returned": 0})]),

    Case("sized_char_array_from_literal", "03 §2.1: char arrays and string literals", "int f(int k)", src("""
        int f(int k) {
            char s[6] = "level";
            s[0] = 'L';
            return s[0] * 1000 + s[k];
        }"""), [([4], {"returned": 76108}), ([5], {"returned": 76000})]),

    Case("string_argument_length", "03 §2.3: Python str args become char[] with a terminator (Q13)", "int str_length(char s[])", src("""
        int str_length(char s[]) {
            int i = 0;
            while (s[i] != '\\0') {
                i++;
            }
            return i;
        }"""), [(["level"], {"returned": 5}), ([""], {"returned": 0}), (["a b"], {"returned": 3})]),

    Case("count_vowels", "03 §2.1: char literals compared with char cells (Q14)", "int count_vowels(char s[])", src("""
        int count_vowels(char s[]) {
            int c = 0;
            for (int i = 0; s[i] != '\\0'; i++) {
                if (s[i] == 'a' || s[i] == 'e' || s[i] == 'i' || s[i] == 'o' || s[i] == 'u') {
                    c++;
                }
            }
            return c;
        }"""), [(["education"], {"returned": 5}), (["xyz"], {"returned": 0}), ([""], {"returned": 0})]),

    Case("is_palindrome", "03 §2.1: char arrays, two pointers (Q15)", "int is_palindrome(char s[])", src("""
        int is_palindrome(char s[]) {
            int n = 0;
            while (s[n] != '\\0') n++;
            int i = 0;
            int j = n - 1;
            while (i < j) {
                if (s[i] != s[j]) return 0;
                i++;
                j--;
            }
            return 1;
        }"""), [(["level"], {"returned": 1}), (["levex"], {"returned": 0}), (["ab"], {"returned": 0}),
                (["a"], {"returned": 1}), ([""], {"returned": 1})]),

    Case("strlen_builtin", "03 §2.1: builtin strlen", "int f(char s[])", src("""
        int f(char s[]) {
            char local[] = "four";
            int n = strlen(s);
            return n * 10 + strlen(local);
        }"""), [(["hello"], {"returned": 54}), ([""], {"returned": 4})]),

    Case("printf_percent_s", "03 §2.1: printf(\"%s\")", "void greet(char s[])", src("""
        void greet(char s[]) {
            char mark[] = "hi";
            printf("%s %s!%c\\n", mark, s, s[0]);
        }"""), [(["crew"], {"printed": "hi crew!c\n"})]),

    Case("string_argument_is_shared", "03 §2.1: char s[] params (arrays by reference)", "int f(char s[])", src("""
        void upper_first(char s[]) {
            s[0] = s[0] - 32;
        }
        int f(char s[]) {
            upper_first(s);
            return s[0];
        }"""), [(["abc"], {"returned": 65})]),
]
