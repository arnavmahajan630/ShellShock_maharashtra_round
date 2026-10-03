"""The ordered feature list: the agreement between the feature packages and the model.

Contract: plans/03 §4. Order matters; `FEATURES` is the column order of every matrix.
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

GROUPS = {"A": GROUP_A, "B": GROUP_B, "R": GROUP_R, "F": GROUP_F, "C": GROUP_C}
FEATURES = GROUP_A + GROUP_B + GROUP_R + GROUP_F + GROUP_C

# Set to NaN on the 15% feature-dropout rows and whenever there is no trace (03 §4.6).
TRACE_DEPENDENT_GROUPS = ["B", "R", "F"]

# Only if Strong S4 (M09) lands; not part of FEATURES.
STRONG_FEATURES = ["a_param_written"]

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
