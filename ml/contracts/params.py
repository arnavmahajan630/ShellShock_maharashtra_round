"""Hand-set numbers shared by the Bayes layer, knowledge model, exam and simulations.

Contract: plans/03 §5.5, §6, §8 and plans/05 §4. These are design assumptions, not fitted
values. Read them from here so every package uses the same ones.
"""

# --- Bayes layer (03 §6.1, §6.2) ---
P_BELIEF = 0.6          # p_b: a learner with misconception k gives k's belief answer
Q_CORRECT = 0.5         # q: share of the rest that goes to the correct answer
SLIP = 0.1              # s: a learner without the misconception answers wrongly
P_OTHER_CORRECT = 0.4   # an OTHER learner answers correctly
PROB_FLOOR = 0.02       # every likelihood is floored here, then renormalised
PRIOR_GAMMA = 0.3       # exponent on the learner prior
PRIOR_FLOOR = 0.02

# --- Decision logic (03 §5.5, §6.4) ---
CONFIDENT_P = 0.75
CONFIDENT_MARGIN = 0.25
POSTERIOR_STOP = 0.85   # stop probing once the top class reaches this
EIG_MIN_BITS = 0.10
MAX_PROBES = 2
TWO_BUG_P2 = 0.25
NOVEL_P_OTHER = 0.5

# --- Knowledge model (03 §8.1, §8.2) ---
POPULATION_PRIOR = 0.10
EXAM_DSA_PRIOR = 0.15           # D-classes at exam start if unseen (03 §8.5.2)
DIAGNOSIS_ACTIVE_P = 0.5
LEARN_RATE = 0.35               # ℓ, applied when an intervention completes
FORGET_RATE = 0.05              # φ, per intervening level before a ghost return
HINT_GUESS_BONUS = 0.2
HINT_GUESS_CAP = 0.9
FAIL_SIGNATURE_IF_ACTIVE = 0.5
FAIL_SIGNATURE_IF_NOT = 0.03

# g = P(correct | active), s = P(wrong | not active), per item type.
# "belief_mcq" covers probes, mcq, predict_output, next_state and exam trace items:
# g is derived from the Bayes table as (1 - P_BELIEF) * Q_CORRECT.
ITEM_GUESS_SLIP = {
    "same_family_code": (0.45, 0.10),
    "transfer_code": (0.30, 0.15),
    "trap": (0.15, 0.10),
    "belief_mcq": ((1 - P_BELIEF) * Q_CORRECT, 0.10),
    "ghost": (0.25, 0.15),
    "exam_code": (0.30, 0.15),
    "debug_line": (0.30, 0.20),     # 05 §4
    "fix_bug": (0.35, 0.15),        # 05 §4
}

# State machine thresholds (03 §8.4).
STABLE_P = 0.15
MASTERED_P = 0.10
RELAPSE_TO_TREATING_P = 0.5
EXAM_MASTERY_EXPOSURE = 0.4

# --- Sentence reader (05 §4) ---
TEXT_EPSILON = 0.02
TEXT_BETA = 0.5
TEXT_MATCH_P = 0.6
TEXT_IF_ACTIVE = 0.5
TEXT_IF_NOT = 0.05

# --- Exam (03 §8.5) ---
ELO_START = 1300
ELO_PLANETS_BONUS = 100
ELO_ITEM = {1: 1200, 2: 1400, 3: 1600}
ELO_K = 40
EXAM_COST = {"coding": 1.0, "trace": 0.3}
EXAM_MIN_EXPOSURE = 0.2
EXAM_W_COVERAGE = 0.5
EXAM_W_DIFFICULTY = 0.4
EXAM_TARGET_PASS = 0.6
EXAM_W_GHOST = 0.3
EXAM_LENGTH = {"full": (5, 5), "demo": (2, 1)}   # (coding, trace)
EXAM_TIME_LIMIT_S = 1500
