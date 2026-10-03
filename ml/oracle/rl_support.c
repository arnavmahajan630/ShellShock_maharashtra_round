/* Support code linked next to the learner's file by ml/oracle/gcc.py.
 *
 * It lives in its own translation unit so that no system header is ever visible to the
 * learner's code (mingw headers define macros such as `max`, `min` and `small`).
 *
 * Provides: a bounded printf, the world builtins as call counters, the call-depth hooks
 * used by -finstrument-functions, padded argument arrays, fault handlers, and the report
 * block that gcc.py parses.
 */
#include <signal.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#include <windows.h>
#endif

#ifndef RL_GARBAGE
#define RL_GARBAGE (-858993460)
#endif
#ifndef RL_DEPTH_CAP
#define RL_DEPTH_CAP 100
#endif
#define RL_PAD 64            /* guard cells on each side of an argument array */
#define RL_OUT_CAP 65536     /* bytes of printf output kept; more than this ends the run */

static char rl_out[RL_OUT_CAP + 1];
static size_t rl_out_len = 0;

static int rl_active = 0, rl_depth = 0, rl_max_depth = 0, rl_calls = 0;
static int rl_fire = 0, rl_launch = 0, rl_door_open = 0, rl_door_closed = 0, rl_scan = 0;

static int rl_ret_kind = 0;              /* 0 none, 1 int, 2 float, 3 void */
static long long rl_ret_i = 0;
static double rl_ret_f = 0.0;

static int rl_arr_kind = 0;              /* 0 none, 1 int, 2 float, 3 double, 4 char */
static const void *rl_arr = NULL;
static int rl_arr_n = 0;

static void rl_emit(const char *status) {
    size_t i;
    int k;
    rl_active = 0;
    printf("\n@@RL1@@\nstatus %s\n", status);
    if (rl_ret_kind == 1) printf("ret int %lld\n", rl_ret_i);
    else if (rl_ret_kind == 2) printf("ret float %.9g\n", rl_ret_f);
    else if (rl_ret_kind == 3) printf("ret void\n");
    else printf("ret none\n");
    printf("printed ");
    for (i = 0; i < rl_out_len; i++) printf("%02x", (unsigned char)rl_out[i]);
    printf(rl_arr_kind ? "\narray0" : "\narray0 none");
    for (k = 0; k < rl_arr_n; k++) {
        if (rl_arr_kind == 1) printf(" %d", ((const int *)rl_arr)[k]);
        else if (rl_arr_kind == 2) printf(" %.9g", (double)((const float *)rl_arr)[k]);
        else if (rl_arr_kind == 3) printf(" %.9g", ((const double *)rl_arr)[k]);
        else if (rl_arr_kind == 4) printf(" %d", (int)((const char *)rl_arr)[k]);
    }
    printf("\ndepth %d\ncalls %d\n", rl_max_depth, rl_calls);
    printf("fx fire=%d launch=%d door_open=%d door_closed=%d scan=%d\n",
           rl_fire, rl_launch, rl_door_open, rl_door_closed, rl_scan);
    printf("@@END@@\n");
    fflush(stdout);
}

static void rl_die(const char *status) {
    rl_emit(status);
    _exit(0);
}

static void rl_on_fpe(int sig) { (void)sig; rl_die("div_zero"); }
static void rl_on_segv(int sig) { (void)sig; rl_die("segv"); }

/* ---- called by the generated main ---- */

int __rl_begin(int argc, char **argv) {
#ifdef _WIN32
    SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX | SEM_NOOPENFILEERRORBOX);
#endif
    signal(SIGFPE, rl_on_fpe);
    signal(SIGSEGV, rl_on_segv);
    return argc > 1 ? atoi(argv[1]) : 0;
}

void __rl_start(void) { rl_depth = 0; rl_active = 1; }
void __rl_stop(void) { rl_active = 0; }
void __rl_ret_int(long long v) { rl_ret_kind = 1; rl_ret_i = v; }
void __rl_ret_float(double v) { rl_ret_kind = 2; rl_ret_f = v; }
void __rl_ret_void(void) { rl_ret_kind = 3; }
int __rl_finish(void) { rl_emit("ok"); return 0; }

static void *rl_padded(const void *init, int n, size_t size, const void *fill) {
    size_t total = (size_t)n + 2 * RL_PAD, i;
    char *base = (char *)malloc(total * size);
    if (!base) rl_die("segv");
    for (i = 0; i < total; i++) memcpy(base + i * size, fill, size);
    if (n > 0) memcpy(base + RL_PAD * size, init, (size_t)n * size);
    return base + RL_PAD * size;
}

int *__rl_iarr(const int *init, int n) { int g = RL_GARBAGE; return (int *)rl_padded(init, n, sizeof g, &g); }
float *__rl_farr(const float *init, int n) { float g = (float)RL_GARBAGE; return (float *)rl_padded(init, n, sizeof g, &g); }
double *__rl_darr(const double *init, int n) { double g = (double)RL_GARBAGE; return (double *)rl_padded(init, n, sizeof g, &g); }
char *__rl_carr(const char *init, int n) { char g = (char)0xCC; return (char *)rl_padded(init, n, sizeof g, &g); }

void __rl_arr0(int kind, const void *p, int n) { rl_arr_kind = kind; rl_arr = p; rl_arr_n = n; }

/* ---- visible to the learner ---- */

int __rl_printf(const char *fmt, ...) {
    va_list ap;
    int n;
    size_t room = RL_OUT_CAP - rl_out_len;
    va_start(ap, fmt);
    n = vsnprintf(rl_out + rl_out_len, room + 1, fmt, ap);
    va_end(ap);
    if (n < 0) return n;
    if ((size_t)n > room) { rl_out_len = RL_OUT_CAP; rl_die("output_limit"); }
    rl_out_len += (size_t)n;
    return n;
}

void fire(void) { rl_fire++; }
void launch(void) { rl_launch++; }
void open_door(void) { rl_door_open++; }
void close_door(void) { rl_door_closed++; }
int scan(int x) { (void)x; rl_scan++; return 0; }

/* ---- -finstrument-functions hooks: only the learner's file is instrumented ---- */

void __cyg_profile_func_enter(void *fn, void *site) {
    (void)fn; (void)site;
    if (!rl_active) return;
    rl_calls++;
    rl_depth++;
    if (rl_depth > RL_DEPTH_CAP) rl_die("depth_cap");
    if (rl_depth > rl_max_depth) rl_max_depth = rl_depth;
}

void __cyg_profile_func_exit(void *fn, void *site) {
    (void)fn; (void)site;
    if (rl_active) rl_depth--;
}
