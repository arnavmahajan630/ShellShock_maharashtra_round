"""C snippets for the F1 tests. Each covers one feature family or one near-miss CORRECT form (03 §3.5.3)."""

SNIPPETS = {}


def add(name, code):
    SNIPPETS[name] = code.strip("\n")


# ---------------------------------------------------------------- loops: bounds, init, update

add("sum_for_lt", """
int total_energy(int cells[], int n) {
    int total = 0;
    for (int i = 0; i < n; i++) {
        total += cells[i];
    }
    return total;
}
""")

add("sum_for_le_twin", """
int total_energy(int cells[], int n) {
    int total = 0;
    for (int i = 0; i <= n; i++) {
        total += cells[i];
    }
    return total;
}
""")

add("sum_while_no_update", """
int total_energy(int cells[], int n) {
    int total = 0;
    int i = 0;
    while (i < n) {
        total += cells[i];
    }
    return total;
}
""")

add("sum_countdown_correct", """
int total_energy(int cells[], int n) {
    int total = 0;
    int i;
    for (i = n - 1; i >= 0; i--) {
        total += cells[i];
    }
    return total;
}
""")

add("sum_one_based_correct", """
int total_energy(int cells[], int n) {
    int s = 0;
    int i;
    for (i = 1; i <= n; i++)
        s += cells[i - 1];
    return s;
}
""")

add("sum_one_based_m08", """
int total_energy(int cells[], int n) {
    int s = 0;
    for (int i = 1; i <= n; i++) {
        s += cells[i];
    }
    return s;
}
""")

add("count_le_n_minus_1_correct", """
void fire_shots(int n) {
    int i;
    for (i = 0; i <= n - 1; i = i + 1) {
        fire();
    }
}
""")

add("count_n_plus_1", """
void fire_shots(int n) {
    for (int i = 0; i < n + 1; ++i) {
        fire();
    }
}
""")

add("while_reverse_update", """
void fire_shots(int n) {
    int i = 0;
    while (i < n) {
        fire();
        i--;
    }
}
""")

add("while_wrong_var", """
int charge_steps(int level, int target) {
    int steps = 0;
    while (level < target) {
        steps++;
    }
    return steps;
}
""")

add("while_update_in_branch", """
int count_pos(int a[], int n) {
    int c = 0;
    int i = 0;
    while (i < n) {
        if (a[i] > 0) {
            c++;
            i++;
        }
    }
    return c;
}
""")

add("two_pointer_correct", """
int pair_sum_exists(int a[], int n, int target) {
    int i = 0;
    int j = n - 1;
    while (i < j) {
        int s = a[i] + a[j];
        if (s == target) {
            return 1;
        } else if (s < target) {
            i++;
        } else {
            j--;
        }
    }
    return 0;
}
""")

# ---------------------------------------------------------------- stray semicolons, assignment in condition

add("for_semicolon", """
int total_energy(int cells[], int n) {
    int total = 0;
    int i;
    for (i = 0; i < n; i++);
    {
        total += cells[i];
    }
    return total;
}
""")

add("if_semicolon", """
int door_open(int code) {
    if (code == 42); {
        return 1;
    }
    return 0;
}
""")

add("while_semicolon", """
int charge_steps(int level, int target) {
    int steps = 0;
    while (level < target);
    {
        level += 7;
        steps++;
    }
    return steps;
}
""")

add("if_assign", """
int door_open(int code) {
    if (code = 42) {
        return 1;
    }
    return 0;
}
""")

add("if_yoda_correct", """
int door_open(int code) {
    if (42 == code)
        return 1;
    return 0;
}
""")

add("while_assign", """
int charge_steps(int level, int target) {
    int steps = 0;
    while (level = target) {
        level += 7;
        steps++;
    }
    return steps;
}
""")

# ---------------------------------------------------------------- variables: uninitialised, reset in loop

add("uninit_accumulator", """
int total_energy(int cells[], int n) {
    int total;
    for (int i = 0; i < n; i++) {
        total += cells[i];
    }
    return total;
}
""")

add("decl_then_assign_correct", """
int total_energy(int cells[], int n) {
    int total;
    int i;
    total = 0;
    for (i = 0; i < n; i++) {
        total = total + cells[i];
    }
    return total;
}
""")

add("uninit_max", """
int max_shield(int s[], int n) {
    int best;
    for (int i = 0; i < n; i++) {
        if (s[i] > best) {
            best = s[i];
        }
    }
    return best;
}
""")

add("reset_decl_in_loop", """
int total_energy(int cells[], int n) {
    int total = 0;
    for (int i = 0; i < n; i++) {
        int total = 0;
        total += cells[i];
    }
    return total;
}
""")

add("reset_assign_in_loop", """
int count_overheated(int t[], int n, int limit) {
    int c = 0;
    int i;
    for (i = 0; i < n; i++) {
        c = 0;
        if (t[i] > limit)
            c = c + 1;
    }
    return c;
}
""")

# ---------------------------------------------------------------- division typing

add("int_div_returned_as_float", """
float avg_fuel(int tanks[], int n) {
    int sum = 0;
    for (int i = 0; i < n; i++) {
        sum += tanks[i];
    }
    return sum / n;
}
""")

add("cast_after_division", """
float fuel_percent(int fuel, int cap) {
    float p = (float)(fuel * 100 / cap);
    return p;
}
""")

add("cast_numerator_correct", """
float avg_fuel(int tanks[], int n) {
    int sum = 0;
    for (int i = 0; i < n; i++) {
        sum += tanks[i];
    }
    return (float)sum / n;
}
""")

add("float_literal_correct", """
float fuel_percent(int fuel, int cap) {
    return fuel * 100.0 / cap;
}
""")

add("int_literal_half", """
float half(int x) {
    float h = x / 2;
    return h;
}
""")

# ---------------------------------------------------------------- index shapes

add("last_index_n", """
int last_beacon(int ids[], int n) {
    return ids[n];
}
""")

add("last_index_n_minus_1_correct", """
int last_beacon(int ids[], int n) {
    return ids[n - 1];
}
""")

add("max_first_const1", """
int max_shield(int s[], int n) {
    int best = s[1];
    for (int i = 1; i < n; i++) {
        if (s[i] > best) best = s[i];
    }
    return best;
}
""")

add("max_first_const0_correct", """
int max_shield(int s[], int n) {
    int best = s[0];
    int i;
    for (i = 1; i < n; i++) {
        if (s[i] > best) best = s[i];
    }
    return best;
}
""")

add("index_i_plus_1", """
int total_energy(int cells[], int n) {
    int total = 0;
    for (int i = 0; i < n; i++) {
        total += cells[i + 1];
    }
    return total;
}
""")

# ---------------------------------------------------------------- printf / return

add("printf_no_return", """
int distance(int a, int b) {
    int d = a - b;
    if (d < 0) {
        d = -d;
    }
    printf("%d", d);
}
""")

add("printf_return_zero", """
int distance(int a, int b) {
    int d = a - b;
    if (d < 0) d = -d;
    printf("%d", d);
    return 0;
}
""")

add("debug_printf_correct", """
int distance(int a, int b) {
    int d = a - b;
    printf("d=%d\\n", d);
    if (d < 0) d = -d;
    return d;
}
""")

add("constant_answer", """
int door_open(int code) {
    return 1;
}
""")

# ---------------------------------------------------------------- sorting: nested loops, swaps

add("bubble_correct", """
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
}
""")

add("bubble_no_temp", """
void bubble_sort(int a[], int n) {
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - 1 - i; j++) {
            if (a[j] > a[j + 1]) {
                a[j] = a[j + 1];
                a[j + 1] = a[j];
            }
        }
    }
}
""")

add("bubble_outer_once", """
void bubble_sort(int a[], int n) {
    int i, j, t;
    for (i = 0; i < 1; i++) {
        for (j = 0; j < n - 1; j++) {
            if (a[j] > a[j + 1]) {
                t = a[j + 1];
                a[j + 1] = a[j];
                a[j] = t;
            }
        }
    }
}
""")

add("bubble_single_pass", """
void bubble_sort(int a[], int n) {
    int j;
    for (j = 0; j < n - 1; j++) {
        if (a[j] > a[j + 1]) {
            int t = a[j];
            a[j] = a[j + 1];
            a[j + 1] = t;
        }
    }
}
""")

add("bubble_pair_bound_twin", """
void bubble_sort(int a[], int n) {
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n; j++) {
            if (a[j] > a[j + 1]) {
                int t = a[j];
                a[j] = a[j + 1];
                a[j + 1] = t;
            }
        }
    }
}
""")

# ---------------------------------------------------------------- searching

add("search_else_return", """
int linear_search(int a[], int n, int x) {
    for (int i = 0; i < n; i++) {
        if (a[i] == x) {
            return i;
        } else {
            return -1;
        }
    }
    return -1;
}
""")

add("search_flag_reset", """
int contains(int a[], int n, int x) {
    int found = 0;
    for (int i = 0; i < n; i++) {
        if (a[i] == x)
            found = 1;
        else
            found = 0;
    }
    return found;
}
""")

add("search_flag_break_correct", """
int contains(int a[], int n, int x) {
    int found = 0;
    for (int i = 0; i < n; i++) {
        if (a[i] == x) {
            found = 1;
            break;
        } else {
            continue;
        }
    }
    return found;
}
""")

add("bsearch_low_mid", """
int binary_search(int a[], int n, int x) {
    int low = 0;
    int high = n - 1;
    while (low <= high) {
        int mid = (low + high) / 2;
        if (a[mid] == x) {
            return mid;
        } else if (a[mid] < x) {
            low = mid;
        } else {
            high = mid - 1;
        }
    }
    return -1;
}
""")

add("bsearch_correct_overflow_safe", """
int binary_search(int a[], int n, int x) {
    int low = 0, high = n - 1, mid;
    while (low <= high) {
        mid = low + (high - low) / 2;
        if (a[mid] < x) {
            low = mid + 1;
        } else if (a[mid] > x) {
            high = mid - 1;
        } else {
            return mid;
        }
    }
    return -1;
}
""")

# ---------------------------------------------------------------- mirrors and half bounds

add("reverse_correct", """
void reverse(int a[], int n) {
    for (int i = 0; i < n / 2; i++) {
        int t = a[i];
        a[i] = a[n - 1 - i];
        a[n - 1 - i] = t;
    }
}
""")

add("reverse_mirror_m08_full_bound", """
void reverse(int a[], int n) {
    for (int i = 0; i < n; i++) {
        int t = a[i];
        a[i] = a[n - i];
        a[n - i] = t;
    }
}
""")

# ---------------------------------------------------------------- recursion

add("fact_correct", """
int factorial(int n) {
    if (n <= 1) {
        return 1;
    }
    return n * factorial(n - 1);
}
""")

add("fact_no_base", """
int factorial(int n) {
    return n * factorial(n - 1);
}
""")

add("fact_unreachable_base", """
int factorial(int n) {
    if (n == 0) return 1;
    return n * factorial(n - 2);
}
""")

add("fact_same_arg", """
int factorial(int n) {
    if (n <= 1) return 1;
    return n * factorial(n);
}
""")

add("fact_grow_arg", """
int factorial(int n) {
    if (n <= 1) return 1;
    return n * factorial(n + 1);
}
""")

add("fact_discarded", """
int factorial(int n) {
    if (n <= 1) return 1;
    factorial(n - 1);
    return n;
}
""")

add("fact_accumulator_correct", """
int go(int n, int acc) {
    if (n <= 1) return acc;
    return go(n - 1, acc * n);
}
int factorial(int n) {
    return go(n, 1);
}
""")

add("sum_digits_correct", """
int sum_digits(int n) {
    if (n == 0)
        return 0;
    return n % 10 + sum_digits(n / 10);
}
""")

add("array_sum_rec_guarded_correct", """
int array_sum_rec(int a[], int n) {
    if (n > 0) {
        return a[n - 1] + array_sum_rec(a, n - 1);
    }
    return 0;
}
""")

# ---------------------------------------------------------------- strings

add("vowels_string_literal", """
int count_vowels(char s[]) {
    int c = 0;
    for (int i = 0; s[i] != '\\0'; i++) {
        if (s[i] == "a") {
            c++;
        }
    }
    return c;
}
""")

add("array_name_compare", """
int same_text(char s[], char t[]) {
    if (s == t) {
        return 1;
    }
    return 0;
}
""")

add("str_length_correct", """
int str_length(char s[]) {
    int len = 0;
    while (s[len]) {
        len++;
    }
    return len;
}
""")

add("palindrome_correct", """
int is_palindrome(char s[]) {
    int len = 0;
    while (s[len] != '\\0') len++;
    for (int i = 0; i < len / 2; i++) {
        if (s[i] != s[len - 1 - i]) return 0;
    }
    return 1;
}
""")

# ---------------------------------------------------------------- more near-misses and edge shapes

add("two_pointer_stuck", """
int pair_sum_exists(int a[], int n, int target) {
    int i = 0;
    int j = n - 1;
    while (i < j) {
        int s = a[i] + a[j];
        if (s == target) {
            return 1;
        } else if (s < target) {
        } else {
            j--;
        }
    }
    return 0;
}
""")

add("is_sorted_else_return", """
int is_sorted(int a[], int n) {
    for (int i = 0; i < n - 1; i++) {
        if (a[i] > a[i + 1])
            return 0;
        else
            return 1;
    }
    return 1;
}
""")

add("selection_sort_correct", """
void selection_sort(int a[], int n) {
    int i, j, min, t;
    for (i = 0; i < n - 1; i++) {
        min = i;
        for (j = i + 1; j < n; j++) {
            if (a[j] < a[min]) {
                min = j;
            }
        }
        t = a[i];
        a[i] = a[min];
        a[min] = t;
    }
}
""")

add("assign_inside_compare_correct", """
int str_length(char s[]) {
    int i = 0;
    int c;
    while ((c = s[i]) != 0) {
        i++;
    }
    return i;
}
""")

add("alias_last_index_correct", """
int last_beacon(int ids[], int n) {
    int last = n - 1;
    return ids[last];
}
""")

add("do_while_countdown", """
void countdown(int n) {
    do {
        printf("%d\\n", n);
        n -= 1;
    } while (n > 0);
    launch();
}
""")

add("rotate_left_correct", """
void rotate_left(int a[], int n) {
    int first = a[0];
    for (int i = 0; i < n - 1; i++) {
        a[i] = a[i + 1];
    }
    a[n - 1] = first;
}
""")

add("fib_two_bases_correct", """
int fib(int n) {
    if (n == 0) return 0;
    if (n == 1) return 1;
    return fib(n - 1) + fib(n - 2);
}
""")

add("void_recursion_correct", """
void count_up(int i, int n) {
    if (i < n) {
        fire();
        count_up(i + 1, n);
    }
}
""")

add("cast_denominator_correct", """
float avg_fuel(int tanks[], int n) {
    int sum = 0;
    int i = 0;
    while (i < n) {
        sum = sum + tanks[i];
        i += 1;
    }
    return sum / (float)n;
}
""")

add("nested_counter_reset_correct", """
int count_pairs(int a[], int n) {
    int total = 0;
    for (int i = 0; i < n; i++) {
        int j = 0;
        while (j < i) {
            if (a[j] == a[i]) total++;
            j++;
        }
    }
    return total;
}
""")

# ---------------------------------------------------------------- source clean-up

add("comments_include_define", """
#include <stdio.h>
#define LIMIT 3
// yaha total add karo
int total_energy(int cells[], int n) {
    int total = 0; /* running "total" */
    for (int i = 0; i <= LIMIT; i++) {   // i <= n ?
        total += cells[i];
    }
    return total;
}
""")

add("with_main_ignored", """
int door_open(int code) {
    if (code == 42) {
        return 1;
    }
    return 0;
}
int main() {
    int i;
    for (i = 0; i < 3; i++) {
        printf("%d", door_open(i));
    }
    return 0;
}
""")

add("does_not_parse", """
int door_open(int code) {
    if (code == 42 {
        return 1;
    }
""")
