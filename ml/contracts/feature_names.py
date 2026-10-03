"""The ordered feature list: the agreement between the feature packages and the model.

Contract: plans/03 §4. Order matters; `FEATURES` is the column order of every matrix.
`OFFLINE_FEATURES` (group F) are computed after the model, for the top-3 classes only,
and are not columns of the training matrix (03 §4.4, §5.5).

Main loop (03 §4.1): the loop with the most executed iterations on the display test;
ties go to the outermost. `a_main_*`, `a_bound_form_*`, `a_init_form_*` and `a_update_*`
describe that loop. When loops are nested, `a_outer_*` describes the outermost and
`a_inner_*` the loop directly inside it; both are 0 when there is only one loop.

Reference solution: `correct_variants[0]`. Its main loop is chosen by the same rule,
and every learner-vs-reference feature (group B deltas, group R) uses that variant.
"""
from ml.contracts.classes import MISCONCEPTIONS

GROUP_A = [
    "a_n_loops", "a_max_nesting",
    "a_main_cond_op_lt", "a_main_cond_op_le", "a_main_cond_op_gt", "a_main_cond_op_ge",
    "a_main_cond_op_ne", "a_main_cond_op_eq", "a_main_cond_op_other",
    "a_bound_form_n", "a_bound_form_n_minus_1", "a_bound_form_n_plus_1",
    "a_bound_form_const", "a_bound_form_other",
    "a_init_form_0", "a_init_form_1", "a_init_form_n", "a_init_form_n_minus_1", "a_init_form_other",
    "a_update_present", "a_update_dir", "a_update_var_is_cond_var", "a_update_in_branch",
    # Nested loops only (0 when the function has one loop). Outer = outermost;
    # inner = the loop nested directly inside it.
    "a_outer_cond_op_lt", "a_outer_cond_op_le", "a_outer_cond_op_gt", "a_outer_cond_op_ge",
    "a_outer_cond_op_ne", "a_outer_cond_op_eq", "a_outer_cond_op_other",
    "a_inner_cond_op_lt", "a_inner_cond_op_le", "a_inner_cond_op_gt", "a_inner_cond_op_ge",
    "a_inner_cond_op_ne", "a_inner_cond_op_eq", "a_inner_cond_op_other",
    "a_outer_bound_n", "a_outer_bound_n_minus_1", "a_outer_bound_n_plus_1",
    "a_outer_bound_const", "a_outer_bound_other",
    "a_empty_body_if", "a_empty_body_for", "a_empty_body_while",
    "a_assign_in_cond",
    "a_decl_noinit_read_first",
    "a_init_inside_loop",
    "a_int_div_into_float", "a_cast_wraps_division", "a_float_literal_in_div",
    "a_index_i", "a_index_i_plus_1", "a_index_i_minus_1", "a_index_n", "a_index_n_minus_1",
    "a_index_const0", "a_index_const1",
    "a_array_loop_start1", "a_array_loop_le_n",
    "a_printf_in_nonvoid", "a_nonvoid_missing_return", "a_return_const_only",
    "a_reads_all_params",
    # DSA (v3)
    "a_n_nested_loops", "a_outer_bound_const1", "a_single_loop_with_swap",
    "a_return_in_loop_else", "a_assign_in_loop_else",
    "a_mid_assign_no_offset",
    "a_mid_like_var",
    "a_swap_no_temp",
    "a_swap_with_temp",
    "a_index_pair_plus1", "a_index_mirror_n_minus_i", "a_index_mirror_n_minus_1_minus_i", "a_half_bound",
    "a_has_recursion", "a_n_self_calls",
    "a_base_case_present", "a_base_case_static_reachable",
    "a_rec_arg_form_minus1", "a_rec_arg_form_minus2", "a_rec_arg_form_div10",
    "a_rec_arg_form_same", "a_rec_arg_form_plus1", "a_rec_arg_form_other",
    "a_rec_call_discarded",
    "a_str_literal_compare", "a_array_name_compare",
    "a_char_array_param", "a_uses_terminator",
    # W0 addition: 03 §4.1b masks D03/D04 on "an array write" but names no feature for it.
    "a_has_array_write",
]

GROUP_B = [
    "b_pass_frac",
    "b_status_timeout", "b_status_runtime_error",
    "b_step_cap",
    "b_uninit_read",
    "b_oob_read", "b_oob_read_idx_eq_n", "b_oob_read_idx_neg", "b_oob_write",
    "b_intdiv_trunc_nonzero", "b_intdiv_into_float",
    "b_iter_delta_mean", "b_iter_delta_const_pm1",
    "b_body_once_vs_many",
    "b_empty_body_exec",
    "b_assign_in_cond_rt",
    "b_branch_always", "b_branch_never",
    "b_missing_return", "b_returned_garbage",
    "b_printed_nonempty", "b_printed_eq_ref_return",
    # DSA (v3)
    "b_depth_cap", "b_max_depth_ratio",
    "b_rec_arg_constant", "b_rec_arg_growing",
    "b_base_return_executed",
    "b_discarded_call_value",
    "b_return_first_iter",
    "b_window_frozen",
    "b_multiset_changed",
    "b_sorted_frac",
    "b_n_outer_passes_ratio",
    "b_str_literal_compare", "b_array_compare",
]

GROUP_R = [
    "r_eq_ref_drop_first", "r_eq_ref_drop_last",
    "r_eq_ref_last_only", "r_eq_ref_first_only",
    "r_eq_ref_n_minus_1", "r_eq_ref_n_plus_1",
    "r_eq_floor_ref",
    "r_is_garbage", "r_eq_zero", "r_off_by_value_1",
    # DSA (v3)
    "r_eq_one_pass",
    "r_eq_first_check_only", "r_eq_last_check_only",
    "r_eq_top_frame_only",
    "r_eq_reversed_twice",
    "r_eq_shift_without_wrap",
]

GROUP_F = [f"f_fix_{k}" for k in MISCONCEPTIONS] + [f"f_fixgain_{k}" for k in MISCONCEPTIONS]

GROUP_C = [
    "t_has_array_param", "t_returns_float", "t_is_void",
    "t_has_char_array_param", "t_mutates_array_arg",
]

# Group F is not a model input. It is computed after the model, for the top-3 classes
# only, and kept here so the E9 ablation can add it back as an offline row (03 §4.4).
OFFLINE_FEATURES = GROUP_F

GROUPS = {"A": GROUP_A, "B": GROUP_B, "R": GROUP_R, "C": GROUP_C}
FEATURES = GROUP_A + GROUP_B + GROUP_R + GROUP_C

# Set to NaN on the 15% feature-dropout rows and whenever there is no trace (03 §4.6).
TRACE_DEPENDENT_GROUPS = ["B", "R"]

# Only if Strong S4 (M09) lands; not part of FEATURES.
STRONG_FEATURES = ["a_param_written"]

# Features that are the signature of one class (03 §3.1 "Defining evidence", §4.1, §4.2).
# E6 "LOCO strict" drops class k's list before scoring k's rows, and EDA 10.5 reads it.
# Every name is in FEATURES. Group F is not here: fixers are skipped separately in that fold.
CLASS_DEFINING_FEATURES = {
    "M01": ["b_iter_delta_const_pm1", "b_iter_delta_mean", "a_half_bound", "r_eq_reversed_twice"],
    "M02": ["b_step_cap", "a_update_present", "a_update_dir", "a_update_var_is_cond_var", "a_update_in_branch"],
    "M03": ["r_eq_ref_last_only", "a_init_inside_loop"],
    "M04": ["b_intdiv_trunc_nonzero", "b_intdiv_into_float", "a_int_div_into_float",
            "a_cast_wraps_division", "a_float_literal_in_div", "r_eq_floor_ref"],
    "M05": ["b_uninit_read", "a_decl_noinit_read_first", "r_is_garbage"],
    "M06": ["a_assign_in_cond", "b_assign_in_cond_rt"],
    "M07": ["a_empty_body_if", "a_empty_body_for", "a_empty_body_while",
            "b_empty_body_exec", "b_body_once_vs_many"],
    "M08": ["b_oob_read_idx_eq_n", "a_index_n", "a_array_loop_le_n", "a_array_loop_start1",
            "a_index_i_plus_1", "a_index_const1", "r_eq_ref_drop_first"],
    "M10": ["b_printed_eq_ref_return", "a_printf_in_nonvoid", "a_nonvoid_missing_return"],
    "D01": ["a_return_in_loop_else", "a_assign_in_loop_else", "b_return_first_iter",
            "r_eq_first_check_only", "r_eq_last_check_only"],
    "D02": ["a_mid_assign_no_offset", "b_window_frozen"],
    "D03": ["a_swap_no_temp", "b_multiset_changed", "r_eq_shift_without_wrap"],
    "D04": ["a_outer_bound_const1", "a_single_loop_with_swap", "r_eq_one_pass",
            "b_n_outer_passes_ratio", "b_sorted_frac"],
    "D05": ["b_depth_cap", "a_base_case_present", "a_base_case_static_reachable"],
    "D06": ["b_depth_cap", "b_rec_arg_constant", "b_rec_arg_growing",
            "a_rec_arg_form_same", "a_rec_arg_form_plus1"],
    "D07": ["a_rec_call_discarded", "b_discarded_call_value", "r_eq_top_frame_only"],
    "D08": ["a_str_literal_compare", "a_array_name_compare", "b_str_literal_compare", "b_array_compare"],
}

# A class is impossible unless one of these code features is present (03 §4.1b).
MASK_PRECONDITIONS = {
    "D02": ["a_mid_like_var"],
    "D03": ["a_has_array_write"],
    "D04": ["a_has_array_write"],
    "D05": ["a_has_recursion"],
    "D06": ["a_has_recursion"],
    "D07": ["a_has_recursion"],
    "D08": ["a_str_literal_compare", "a_array_name_compare", "a_char_array_param"],
}
