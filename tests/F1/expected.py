"""Hand-checked group-A values for tests/F1/snippets.py.

Each entry lists every feature that is not 0 for that snippet; all other group-A
features must be exactly 0. The values were worked out from the feature definitions in
ml_plan/03 §4.1 and the rules in ml/features/ast_feats.py, snippet by snippet.
"""

EXPECTED = {
    "sum_for_lt": {
        "a_n_loops": 1, "a_max_nesting": 1, "a_main_cond_op_lt": 1, "a_bound_form_n": 1, "a_init_form_0": 1,
        "a_update_present": 1, "a_update_dir": 1, "a_update_var_is_cond_var": 1, "a_index_i": 1,
        "a_reads_all_params": 1,
    },
    "sum_for_le_twin": {
        "a_n_loops": 1, "a_max_nesting": 1, "a_main_cond_op_le": 1, "a_bound_form_n": 1, "a_init_form_0": 1,
        "a_update_present": 1, "a_update_dir": 1, "a_update_var_is_cond_var": 1, "a_index_i": 1,
        "a_array_loop_le_n": 1, "a_reads_all_params": 1,
    },
    "sum_while_no_update": {
        "a_n_loops": 1, "a_max_nesting": 1, "a_main_cond_op_lt": 1, "a_bound_form_n": 1, "a_init_form_0": 1,
        "a_index_i": 1, "a_reads_all_params": 1,
    },
    "sum_countdown_correct": {
        "a_n_loops": 1, "a_max_nesting": 1, "a_main_cond_op_ge": 1, "a_bound_form_const": 1,
        "a_init_form_n_minus_1": 1, "a_update_present": 1, "a_update_dir": -1, "a_update_var_is_cond_var": 1,
        "a_index_i": 1, "a_reads_all_params": 1,
    },
    "sum_one_based_correct": {
        "a_n_loops": 1, "a_max_nesting": 1, "a_main_cond_op_le": 1, "a_bound_form_n": 1, "a_init_form_1": 1,
        "a_update_present": 1, "a_update_dir": 1, "a_update_var_is_cond_var": 1, "a_index_i_minus_1": 1,
        "a_reads_all_params": 1,
    },
    "sum_one_based_m08": {
        "a_n_loops": 1, "a_max_nesting": 1, "a_main_cond_op_le": 1, "a_bound_form_n": 1, "a_init_form_1": 1,
        "a_update_present": 1, "a_update_dir": 1, "a_update_var_is_cond_var": 1, "a_index_i": 1,
        "a_array_loop_start1": 1, "a_array_loop_le_n": 1, "a_reads_all_params": 1,
    },
    "count_le_n_minus_1_correct": {
        "a_n_loops": 1, "a_max_nesting": 1, "a_main_cond_op_le": 1, "a_bound_form_n_minus_1": 1,
        "a_init_form_0": 1, "a_update_present": 1, "a_update_dir": 1, "a_update_var_is_cond_var": 1,
        "a_reads_all_params": 1,
    },
    "count_n_plus_1": {
        "a_n_loops": 1, "a_max_nesting": 1, "a_main_cond_op_lt": 1, "a_bound_form_n_plus_1": 1,
        "a_init_form_0": 1, "a_update_present": 1, "a_update_dir": 1, "a_update_var_is_cond_var": 1,
        "a_reads_all_params": 1,
    },
    "while_reverse_update": {
        "a_n_loops": 1, "a_max_nesting": 1, "a_main_cond_op_lt": 1, "a_bound_form_n": 1, "a_init_form_0": 1,
        "a_update_present": 1, "a_update_dir": -1, "a_update_var_is_cond_var": 1, "a_reads_all_params": 1,
    },
    "while_wrong_var": {
        "a_n_loops": 1, "a_max_nesting": 1, "a_main_cond_op_lt": 1, "a_bound_form_n": 1,
        "a_init_form_other": 1, "a_update_present": 1, "a_reads_all_params": 1,
    },
    "while_update_in_branch": {
        "a_n_loops": 1, "a_max_nesting": 1, "a_main_cond_op_lt": 1, "a_bound_form_n": 1, "a_init_form_0": 1,
        "a_update_present": 1, "a_update_dir": 1, "a_update_var_is_cond_var": 1, "a_update_in_branch": 1,
        "a_index_i": 1, "a_reads_all_params": 1,
    },
    "two_pointer_correct": {
        "a_n_loops": 1, "a_max_nesting": 1, "a_main_cond_op_lt": 1, "a_bound_form_n": 1, "a_init_form_0": 1,
        "a_update_present": 1, "a_update_dir": 1, "a_update_var_is_cond_var": 1, "a_index_i": 1,
        "a_return_const_only": 1, "a_reads_all_params": 1,
    },
    "for_semicolon": {
        "a_n_loops": 1, "a_max_nesting": 1, "a_main_cond_op_lt": 1, "a_bound_form_n": 1, "a_init_form_0": 1,
        "a_update_present": 1, "a_update_dir": 1, "a_update_var_is_cond_var": 1, "a_empty_body_for": 1,
        "a_index_i": 1, "a_reads_all_params": 1,
    },
    "if_semicolon": {
        "a_empty_body_if": 1, "a_return_const_only": 1, "a_reads_all_params": 1,
    },
    "while_semicolon": {
        "a_n_loops": 1, "a_max_nesting": 1, "a_main_cond_op_lt": 1, "a_bound_form_n": 1,
        "a_init_form_other": 1, "a_empty_body_while": 1, "a_reads_all_params": 1,
    },
    "if_assign": {
        "a_assign_in_cond": 1, "a_return_const_only": 1, "a_reads_all_params": 1,
    },
    "if_yoda_correct": {
        "a_return_const_only": 1, "a_reads_all_params": 1,
    },
    "while_assign": {
        "a_n_loops": 1, "a_max_nesting": 1, "a_main_cond_op_other": 1, "a_bound_form_other": 1,
        "a_init_form_other": 1, "a_update_present": 1, "a_update_var_is_cond_var": 1, "a_assign_in_cond": 1,
        "a_reads_all_params": 1,
    },
    "uninit_accumulator": {
        "a_n_loops": 1, "a_max_nesting": 1, "a_main_cond_op_lt": 1, "a_bound_form_n": 1, "a_init_form_0": 1,
        "a_update_present": 1, "a_update_dir": 1, "a_update_var_is_cond_var": 1, "a_decl_noinit_read_first": 1,
        "a_index_i": 1, "a_reads_all_params": 1,
    },
    "decl_then_assign_correct": {
        "a_n_loops": 1, "a_max_nesting": 1, "a_main_cond_op_lt": 1, "a_bound_form_n": 1, "a_init_form_0": 1,
        "a_update_present": 1, "a_update_dir": 1, "a_update_var_is_cond_var": 1, "a_index_i": 1,
        "a_reads_all_params": 1,
    },
    "uninit_max": {
        "a_n_loops": 1, "a_max_nesting": 1, "a_main_cond_op_lt": 1, "a_bound_form_n": 1, "a_init_form_0": 1,
        "a_update_present": 1, "a_update_dir": 1, "a_update_var_is_cond_var": 1, "a_decl_noinit_read_first": 1,
        "a_index_i": 1, "a_reads_all_params": 1,
    },
    "reset_decl_in_loop": {
        "a_n_loops": 1, "a_max_nesting": 1, "a_main_cond_op_lt": 1, "a_bound_form_n": 1, "a_init_form_0": 1,
        "a_update_present": 1, "a_update_dir": 1, "a_update_var_is_cond_var": 1, "a_init_inside_loop": 1,
        "a_index_i": 1, "a_reads_all_params": 1,
    },
    "reset_assign_in_loop": {
        "a_n_loops": 1, "a_max_nesting": 1, "a_main_cond_op_lt": 1, "a_bound_form_n": 1, "a_init_form_0": 1,
        "a_update_present": 1, "a_update_dir": 1, "a_update_var_is_cond_var": 1, "a_init_inside_loop": 1,
        "a_index_i": 1, "a_reads_all_params": 1,
    },
    "int_div_returned_as_float": {
        "a_n_loops": 1, "a_max_nesting": 1, "a_main_cond_op_lt": 1, "a_bound_form_n": 1, "a_init_form_0": 1,
        "a_update_present": 1, "a_update_dir": 1, "a_update_var_is_cond_var": 1, "a_int_div_into_float": 1,
        "a_index_i": 1, "a_reads_all_params": 1,
    },
    "cast_after_division": {
        "a_int_div_into_float": 1, "a_cast_wraps_division": 1, "a_reads_all_params": 1,
    },
    "cast_numerator_correct": {
        "a_n_loops": 1, "a_max_nesting": 1, "a_main_cond_op_lt": 1, "a_bound_form_n": 1, "a_init_form_0": 1,
        "a_update_present": 1, "a_update_dir": 1, "a_update_var_is_cond_var": 1, "a_index_i": 1,
        "a_reads_all_params": 1,
    },
    "float_literal_correct": {
        "a_float_literal_in_div": 1, "a_reads_all_params": 1,
    },
    "int_literal_half": {
        "a_int_div_into_float": 1, "a_reads_all_params": 1,
    },
    "last_index_n": {
        "a_index_n": 1, "a_reads_all_params": 1,
    },
    "last_index_n_minus_1_correct": {
        "a_index_n_minus_1": 1, "a_reads_all_params": 1,
    },
    "max_first_const1": {
        "a_n_loops": 1, "a_max_nesting": 1, "a_main_cond_op_lt": 1, "a_bound_form_n": 1, "a_init_form_1": 1,
        "a_update_present": 1, "a_update_dir": 1, "a_update_var_is_cond_var": 1, "a_index_i": 1,
        "a_index_const1": 1, "a_array_loop_start1": 1, "a_reads_all_params": 1,
    },
    "max_first_const0_correct": {
        "a_n_loops": 1, "a_max_nesting": 1, "a_main_cond_op_lt": 1, "a_bound_form_n": 1, "a_init_form_1": 1,
        "a_update_present": 1, "a_update_dir": 1, "a_update_var_is_cond_var": 1, "a_index_i": 1,
        "a_index_const0": 1, "a_array_loop_start1": 1, "a_reads_all_params": 1,
    },
    "index_i_plus_1": {
        "a_n_loops": 1, "a_max_nesting": 1, "a_main_cond_op_lt": 1, "a_bound_form_n": 1, "a_init_form_0": 1,
        "a_update_present": 1, "a_update_dir": 1, "a_update_var_is_cond_var": 1, "a_index_i_plus_1": 1,
        "a_reads_all_params": 1,
    },
    "printf_no_return": {
        "a_printf_in_nonvoid": 1, "a_nonvoid_missing_return": 1, "a_reads_all_params": 1,
    },
    "printf_return_zero": {
        "a_printf_in_nonvoid": 1, "a_return_const_only": 1, "a_reads_all_params": 1,
    },
    "debug_printf_correct": {
        "a_printf_in_nonvoid": 1, "a_reads_all_params": 1,
    },
    "constant_answer": {
        "a_return_const_only": 1,
    },
    "bubble_correct": {
        "a_n_loops": 2, "a_max_nesting": 2, "a_main_cond_op_lt": 1, "a_bound_form_n_minus_1": 1,
        "a_init_form_0": 1, "a_update_present": 1, "a_update_dir": 1, "a_update_var_is_cond_var": 1,
        "a_outer_cond_op_lt": 1, "a_inner_cond_op_lt": 1, "a_outer_bound_n_minus_1": 1, "a_index_i": 1,
        "a_index_i_plus_1": 1, "a_reads_all_params": 1, "a_n_nested_loops": 1, "a_swap_with_temp": 1,
        "a_index_pair_plus1": 1, "a_has_array_write": 1,
    },
    "bubble_no_temp": {
        "a_n_loops": 2, "a_max_nesting": 2, "a_main_cond_op_lt": 1, "a_bound_form_n_minus_1": 1,
        "a_init_form_0": 1, "a_update_present": 1, "a_update_dir": 1, "a_update_var_is_cond_var": 1,
        "a_outer_cond_op_lt": 1, "a_inner_cond_op_lt": 1, "a_outer_bound_n_minus_1": 1, "a_index_i": 1,
        "a_index_i_plus_1": 1, "a_reads_all_params": 1, "a_n_nested_loops": 1, "a_swap_no_temp": 1,
        "a_index_pair_plus1": 1, "a_has_array_write": 1,
    },
    "bubble_outer_once": {
        "a_n_loops": 2, "a_max_nesting": 2, "a_main_cond_op_lt": 1, "a_bound_form_const": 1,
        "a_init_form_0": 1, "a_update_present": 1, "a_update_dir": 1, "a_update_var_is_cond_var": 1,
        "a_outer_cond_op_lt": 1, "a_inner_cond_op_lt": 1, "a_outer_bound_const": 1, "a_index_i": 1,
        "a_index_i_plus_1": 1, "a_reads_all_params": 1, "a_n_nested_loops": 1, "a_outer_bound_const1": 1,
        "a_swap_with_temp": 1, "a_index_pair_plus1": 1, "a_has_array_write": 1,
    },
    "bubble_single_pass": {
        "a_n_loops": 1, "a_max_nesting": 1, "a_main_cond_op_lt": 1, "a_bound_form_n_minus_1": 1,
        "a_init_form_0": 1, "a_update_present": 1, "a_update_dir": 1, "a_update_var_is_cond_var": 1,
        "a_index_i": 1, "a_index_i_plus_1": 1, "a_reads_all_params": 1, "a_single_loop_with_swap": 1,
        "a_swap_with_temp": 1, "a_index_pair_plus1": 1, "a_has_array_write": 1,
    },
    "bubble_pair_bound_twin": {
        "a_n_loops": 2, "a_max_nesting": 2, "a_main_cond_op_lt": 1, "a_bound_form_n_minus_1": 1,
        "a_init_form_0": 1, "a_update_present": 1, "a_update_dir": 1, "a_update_var_is_cond_var": 1,
        "a_outer_cond_op_lt": 1, "a_inner_cond_op_lt": 1, "a_outer_bound_n_minus_1": 1, "a_index_i": 1,
        "a_index_i_plus_1": 1, "a_reads_all_params": 1, "a_n_nested_loops": 1, "a_swap_with_temp": 1,
        "a_index_pair_plus1": 1, "a_has_array_write": 1,
    },
    "search_else_return": {
        "a_n_loops": 1, "a_max_nesting": 1, "a_main_cond_op_lt": 1, "a_bound_form_n": 1, "a_init_form_0": 1,
        "a_update_present": 1, "a_update_dir": 1, "a_update_var_is_cond_var": 1, "a_index_i": 1,
        "a_reads_all_params": 1, "a_return_in_loop_else": 1,
    },
    "search_flag_reset": {
        "a_n_loops": 1, "a_max_nesting": 1, "a_main_cond_op_lt": 1, "a_bound_form_n": 1, "a_init_form_0": 1,
        "a_update_present": 1, "a_update_dir": 1, "a_update_var_is_cond_var": 1, "a_index_i": 1,
        "a_reads_all_params": 1, "a_assign_in_loop_else": 1,
    },
    "search_flag_break_correct": {
        "a_n_loops": 1, "a_max_nesting": 1, "a_main_cond_op_lt": 1, "a_bound_form_n": 1, "a_init_form_0": 1,
        "a_update_present": 1, "a_update_dir": 1, "a_update_var_is_cond_var": 1, "a_index_i": 1,
        "a_reads_all_params": 1,
    },
    "bsearch_low_mid": {
        "a_n_loops": 1, "a_max_nesting": 1, "a_main_cond_op_le": 1, "a_bound_form_n": 1, "a_init_form_0": 1,
        "a_update_present": 1, "a_update_var_is_cond_var": 1, "a_index_i": 1, "a_reads_all_params": 1,
        "a_mid_assign_no_offset": 1, "a_mid_like_var": 1,
    },
    "bsearch_correct_overflow_safe": {
        "a_n_loops": 1, "a_max_nesting": 1, "a_main_cond_op_le": 1, "a_bound_form_n": 1, "a_init_form_0": 1,
        "a_update_present": 1, "a_update_var_is_cond_var": 1, "a_index_i": 1, "a_reads_all_params": 1,
        "a_mid_like_var": 1,
    },
    "reverse_correct": {
        "a_n_loops": 1, "a_max_nesting": 1, "a_main_cond_op_lt": 1, "a_bound_form_other": 1,
        "a_init_form_0": 1, "a_update_present": 1, "a_update_dir": 1, "a_update_var_is_cond_var": 1,
        "a_index_i": 1, "a_reads_all_params": 1, "a_single_loop_with_swap": 1, "a_swap_with_temp": 1,
        "a_index_mirror_n_minus_1_minus_i": 1, "a_half_bound": 1, "a_has_array_write": 1,
    },
    "reverse_mirror_m08_full_bound": {
        "a_n_loops": 1, "a_max_nesting": 1, "a_main_cond_op_lt": 1, "a_bound_form_n": 1, "a_init_form_0": 1,
        "a_update_present": 1, "a_update_dir": 1, "a_update_var_is_cond_var": 1, "a_index_i": 1,
        "a_reads_all_params": 1, "a_single_loop_with_swap": 1, "a_swap_with_temp": 1,
        "a_index_mirror_n_minus_i": 1, "a_has_array_write": 1,
    },
    "fact_correct": {
        "a_reads_all_params": 1, "a_has_recursion": 1, "a_n_self_calls": 1, "a_base_case_present": 1,
        "a_base_case_static_reachable": 1, "a_rec_arg_form_minus1": 1,
    },
    "fact_no_base": {
        "a_reads_all_params": 1, "a_has_recursion": 1, "a_n_self_calls": 1, "a_rec_arg_form_minus1": 1,
    },
    "fact_unreachable_base": {
        "a_reads_all_params": 1, "a_has_recursion": 1, "a_n_self_calls": 1, "a_base_case_present": 1,
        "a_rec_arg_form_minus2": 1,
    },
    "fact_same_arg": {
        "a_reads_all_params": 1, "a_has_recursion": 1, "a_n_self_calls": 1, "a_base_case_present": 1,
        "a_rec_arg_form_same": 1,
    },
    "fact_grow_arg": {
        "a_reads_all_params": 1, "a_has_recursion": 1, "a_n_self_calls": 1, "a_base_case_present": 1,
        "a_rec_arg_form_plus1": 1,
    },
    "fact_discarded": {
        "a_reads_all_params": 1, "a_has_recursion": 1, "a_n_self_calls": 1, "a_base_case_present": 1,
        "a_base_case_static_reachable": 1, "a_rec_arg_form_minus1": 1, "a_rec_call_discarded": 1,
    },
    "fact_accumulator_correct": {
        "a_reads_all_params": 1, "a_has_recursion": 1, "a_n_self_calls": 1, "a_base_case_present": 1,
        "a_base_case_static_reachable": 1, "a_rec_arg_form_minus1": 1,
    },
    "sum_digits_correct": {
        "a_reads_all_params": 1, "a_has_recursion": 1, "a_n_self_calls": 1, "a_base_case_present": 1,
        "a_base_case_static_reachable": 1, "a_rec_arg_form_div10": 1,
    },
    "array_sum_rec_guarded_correct": {
        "a_index_n_minus_1": 1, "a_reads_all_params": 1, "a_has_recursion": 1, "a_n_self_calls": 1,
        "a_base_case_present": 1, "a_base_case_static_reachable": 1, "a_rec_arg_form_minus1": 1,
    },
    "vowels_string_literal": {
        "a_n_loops": 1, "a_max_nesting": 1, "a_main_cond_op_ne": 1, "a_bound_form_other": 1,
        "a_init_form_0": 1, "a_update_present": 1, "a_update_dir": 1, "a_update_var_is_cond_var": 1,
        "a_index_i": 1, "a_reads_all_params": 1, "a_str_literal_compare": 1, "a_char_array_param": 1,
        "a_uses_terminator": 1,
    },
    "array_name_compare": {
        "a_return_const_only": 1, "a_reads_all_params": 1, "a_array_name_compare": 1, "a_char_array_param": 1,
    },
    "str_length_correct": {
        "a_n_loops": 1, "a_max_nesting": 1, "a_main_cond_op_other": 1, "a_bound_form_other": 1,
        "a_init_form_0": 1, "a_update_present": 1, "a_update_dir": 1, "a_update_var_is_cond_var": 1,
        "a_index_i": 1, "a_reads_all_params": 1, "a_char_array_param": 1, "a_uses_terminator": 1,
    },
    "palindrome_correct": {
        "a_n_loops": 2, "a_max_nesting": 1, "a_main_cond_op_ne": 1, "a_bound_form_other": 1,
        "a_init_form_0": 1, "a_update_present": 1, "a_update_dir": 1, "a_update_var_is_cond_var": 1,
        "a_index_i": 1, "a_return_const_only": 1, "a_reads_all_params": 1,
        "a_index_mirror_n_minus_1_minus_i": 1, "a_half_bound": 1, "a_char_array_param": 1,
        "a_uses_terminator": 1,
    },
    "two_pointer_stuck": {
        "a_n_loops": 1, "a_max_nesting": 1, "a_main_cond_op_lt": 1, "a_bound_form_n": 1, "a_init_form_0": 1,
        "a_update_present": 1, "a_update_dir": -1, "a_update_var_is_cond_var": 1, "a_update_in_branch": 1,
        "a_index_i": 1, "a_return_const_only": 1, "a_reads_all_params": 1,
    },
    "is_sorted_else_return": {
        "a_n_loops": 1, "a_max_nesting": 1, "a_main_cond_op_lt": 1, "a_bound_form_n_minus_1": 1,
        "a_init_form_0": 1, "a_update_present": 1, "a_update_dir": 1, "a_update_var_is_cond_var": 1,
        "a_index_i": 1, "a_index_i_plus_1": 1, "a_return_const_only": 1, "a_reads_all_params": 1,
        "a_return_in_loop_else": 1, "a_index_pair_plus1": 1,
    },
    "selection_sort_correct": {
        "a_n_loops": 2, "a_max_nesting": 2, "a_main_cond_op_lt": 1, "a_bound_form_n_minus_1": 1,
        "a_init_form_0": 1, "a_update_present": 1, "a_update_dir": 1, "a_update_var_is_cond_var": 1,
        "a_outer_cond_op_lt": 1, "a_inner_cond_op_lt": 1, "a_outer_bound_n_minus_1": 1, "a_index_i": 1,
        "a_reads_all_params": 1, "a_n_nested_loops": 1, "a_swap_with_temp": 1, "a_has_array_write": 1,
    },
    "assign_inside_compare_correct": {
        "a_n_loops": 1, "a_max_nesting": 1, "a_main_cond_op_ne": 1, "a_bound_form_other": 1,
        "a_init_form_0": 1, "a_update_present": 1, "a_update_dir": 1, "a_update_var_is_cond_var": 1,
        "a_index_i": 1, "a_reads_all_params": 1, "a_char_array_param": 1,
    },
    "alias_last_index_correct": {
        "a_index_n_minus_1": 1, "a_reads_all_params": 1,
    },
    "do_while_countdown": {
        "a_n_loops": 1, "a_max_nesting": 1, "a_main_cond_op_gt": 1, "a_bound_form_const": 1,
        "a_init_form_other": 1, "a_update_present": 1, "a_update_dir": -1, "a_update_var_is_cond_var": 1,
        "a_reads_all_params": 1,
    },
    "rotate_left_correct": {
        "a_n_loops": 1, "a_max_nesting": 1, "a_main_cond_op_lt": 1, "a_bound_form_n_minus_1": 1,
        "a_init_form_0": 1, "a_update_present": 1, "a_update_dir": 1, "a_update_var_is_cond_var": 1,
        "a_index_i": 1, "a_index_i_plus_1": 1, "a_index_n_minus_1": 1, "a_index_const0": 1,
        "a_reads_all_params": 1, "a_index_pair_plus1": 1, "a_has_array_write": 1,
    },
    "fib_two_bases_correct": {
        "a_reads_all_params": 1, "a_has_recursion": 1, "a_n_self_calls": 2, "a_base_case_present": 1,
        "a_base_case_static_reachable": 1, "a_rec_arg_form_minus1": 1, "a_rec_arg_form_minus2": 1,
    },
    "void_recursion_correct": {
        "a_reads_all_params": 1, "a_has_recursion": 1, "a_n_self_calls": 1, "a_base_case_present": 1,
        "a_base_case_static_reachable": 1, "a_rec_arg_form_plus1": 1,
    },
    "cast_denominator_correct": {
        "a_n_loops": 1, "a_max_nesting": 1, "a_main_cond_op_lt": 1, "a_bound_form_n": 1, "a_init_form_0": 1,
        "a_update_present": 1, "a_update_dir": 1, "a_update_var_is_cond_var": 1, "a_index_i": 1,
        "a_reads_all_params": 1,
    },
    "nested_counter_reset_correct": {
        "a_n_loops": 2, "a_max_nesting": 2, "a_main_cond_op_lt": 1, "a_bound_form_n": 1, "a_init_form_0": 1,
        "a_update_present": 1, "a_update_dir": 1, "a_update_var_is_cond_var": 1, "a_outer_cond_op_lt": 1,
        "a_inner_cond_op_lt": 1, "a_outer_bound_n": 1, "a_index_i": 1, "a_reads_all_params": 1,
        "a_n_nested_loops": 1,
    },
    "comments_include_define": {
        "a_n_loops": 1, "a_max_nesting": 1, "a_main_cond_op_le": 1, "a_bound_form_const": 1,
        "a_init_form_0": 1, "a_update_present": 1, "a_update_dir": 1, "a_update_var_is_cond_var": 1,
        "a_index_i": 1,
    },
    "with_main_ignored": {
        "a_return_const_only": 1, "a_reads_all_params": 1,
    },
}
