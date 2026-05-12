# Test Signal Audit

Date: 2026-05-12

Scope: Python tests under `tests/test_*.py`. Vendored MATLAB and ChunkIE reference files under
`tests/references/` were not audited as project tests.

Outcome: the weak tests found in the audit were strengthened in place. The source no longer carries
non-executing audit markers; this document records which tests were improved and why.

## Criteria

A test was marked too easy when its stated behavior could pass through shape/count checks, conditional
no-op branches, string-membership checks, or comparisons against another output produced by the same
implementation path. Tests that compare against an independent dense result, explicit known values,
MATLAB/FLAM parity output, or a focused exception/failure mode were treated as adequate.

## Strengthened Tests

| Test | Former weakness addressed |
| --- | --- |
| `tests/test_core.py::CoreTests::test_id_honors_fixed_columns_and_reports_iterations` | Checks fixed-column ordering, rank, iteration count, and interpolation shape, but never checks that selected columns reconstruct redundant columns. |
| `tests/test_core.py::CoreTests::test_id_rrqr_refinement_bounds_interpolation` | Checks that RRQR refinement ran and bounded interpolation coefficients, but never checks approximation error. |
| `tests/test_core.py::CoreTests::test_id_rank_cap_and_tolerance_modes` | The rank-cap branch only checks rank and `T` shape; it does not verify capped ID reconstruction. |
| `tests/test_dense_algorithms.py::DenseAlgorithmTests::test_rskelf_partial_diag_uses_compact_factor` | Compares partial diagonal helpers against `rskelf_mv`/`rskelf_sv` from the same factor rather than an independent oracle. |
| `tests/test_geom.py::GeometryTests::test_tri3geom_multiple_faces_use_zero_based_indices` | Claims zero-based face indexing but only checks output shapes and areas, not centroids, normals, or vertex usage. |
| `tests/test_geom.py::GeometryTests::test_trisphere_subdiv_base_and_refined_sizes` | Checks sizes, index bounds, and base vertex norms, but not refined geometry or face connectivity. |
| `tests/test_matlab_parity_utils.py::MatlabParityUtilsTests::test_relerr_uses_absolute_scale_for_zero_reference` | Only covers zero-vs-zero; an implementation returning `0` for every input would pass. |
| `tests/test_matlab_parity_utils.py::MatlabParityUtilsTests::test_default_flam_reference_prefers_complete_checkout` | The non-env branch can pass by only proving the returned path has FLAM markers, not preferred candidate ordering. |
| `tests/test_matlab_parity_utils.py::MatlabParityUtilsTests::test_default_chunkie_reference_prefers_repo_submodule` | Can execute with no assertion when `CHUNKIE_REFERENCE` is set, and does not cover fallback ordering. |
| `tests/test_matlab_parity_utils.py::MatlabParityUtilsTests::test_factor_metadata_code_covers_common_factor_fields` | Only checks field names appear in generated MATLAB code, not that the code executes or serializes expected values. |
| `tests/test_sparse_core.py::SparseCoreTests::test_symmetry_helpers` | `spsymm` has an explicit value check, but the `spsymm2` branch only checks `D == C.T`; paired wrong outputs could pass. |
| `tests/test_utilities.py::UtilityTests::test_quad_sqtri3_weight_sum_is_triangle_area` | Only checks output shape and total weight, not mapped quadrature points or a non-constant integral. |
| `tests/test_utilities.py::UtilityTests::test_lsedc_enforces_constraints` | Only checks equality-constraint feasibility and residual bookkeeping; many non-optimal least-squares solutions would pass. |

## Additional Missing Behavior Covered

| Test | Missing contract now covered |
| --- | --- |
| `tests/test_sparse_core.py::SparseCoreTests::test_symmetry_helpers` | Direct dense and sparse Hermitian `spsymm` value checks, including conjugation rather than plain transpose. |
| `tests/test_sparse_core.py::SparseCoreTests::test_detperm_ismemb_and_logdet_ldl` | Mixed 1x1 and 2x2 LDL block determinant handling in `logdet_ldl`. |
| `tests/test_utilities.py::UtilityTests::test_quadrature_helpers_validate_inputs` | Validation errors for invalid Gauss-Legendre order, mismatched Golub-Welsch coefficients, malformed triangle nodes/vertices, and mismatched weights. |
| `tests/test_utilities.py::UtilityTests::test_lsedc_validates_controls` | Validation errors for negative deferred-correction tolerance and iteration limit. |

## Full Inventory

### `tests/test_chunkie_rskelf_parity.py`

| Test | Verdict |
| --- | --- |
| `ChunkIEMoreRSkelfParityTests.test_laplace_dirichlet_l2scaled_starfish` | OK |
| `ChunkIEMoreRSkelfParityTests.test_helmholtz_combined_layer_starfish` | OK |

### `tests/test_core.py`

| Test | Verdict |
| --- | --- |
| `CoreTests.test_hypoct_empty_and_singleton_trees` | OK |
| `CoreTests.test_hypoct_contains_each_point_once` | OK |
| `CoreTests.test_hypoct_repeated_points_do_not_refine_forever` | OK |
| `CoreTests.test_hypoct_high_dimension_child_codes_do_not_overflow` | OK |
| `CoreTests.test_id_reconstructs_low_rank_columns` | OK |
| `CoreTests.test_id_honors_fixed_columns_and_reports_iterations` | STRENGTHENED |
| `CoreTests.test_id_rrqr_refinement_bounds_interpolation` | STRENGTHENED |
| `CoreTests.test_id_rank_cap_and_tolerance_modes` | STRENGTHENED |
| `CoreTests.test_id_complex_empty_fixed_and_rank_deficient_inputs` | OK |
| `CoreTests.test_snorm_matches_diagonal_norm` | OK |

### `tests/test_dense_algorithms.py`

| Test | Verdict |
| --- | --- |
| `DenseAlgorithmTests.test_rskel_mv_matches_dense` | OK |
| `DenseAlgorithmTests.test_rskel_symmetric_mv_paths_match_dense` | OK |
| `DenseAlgorithmTests.test_rskel_callback_is_not_eagerly_materialized` | OK |
| `DenseAlgorithmTests.test_ifmm_mv_matches_dense` | OK |
| `DenseAlgorithmTests.test_ifmm_mv_generates_missing_interactions_from_A` | OK |
| `DenseAlgorithmTests.test_ifmm_mv_rectangular_adjoint` | OK |
| `DenseAlgorithmTests.test_ifmm_store_near_and_symmetry_modes_match_dense` | OK |
| `DenseAlgorithmTests.test_ifmm_rectangular_complex_proxy_callback` | OK |
| `DenseAlgorithmTests.test_ifmm_mv_promotes_complex_stored_blocks` | OK |
| `DenseAlgorithmTests.test_ifmm_mv_promotes_complex_callback_blocks` | OK |
| `DenseAlgorithmTests.test_rskelf_mv_sv_logdet_match_dense` | OK |
| `DenseAlgorithmTests.test_rskelf_positive_definite` | OK |
| `DenseAlgorithmTests.test_rskelf_symmetric_compact_paths_match_dense` | OK |
| `DenseAlgorithmTests.test_rskelf_generalized_cholesky_round_trips` | OK |
| `DenseAlgorithmTests.test_rskelf_callback_is_not_eagerly_materialized` | OK |
| `DenseAlgorithmTests.test_rskelf_partial_mv_sv_use_skeleton_callback` | OK |
| `DenseAlgorithmTests.test_rskelf_partial_diag_uses_compact_factor` | STRENGTHENED |
| `DenseAlgorithmTests.test_rskelf_diag_uses_selected_unfolding_for_complete_factors` | OK |
| `DenseAlgorithmTests.test_rskelf_spdiag_uses_sparse_propagation_for_complete_factors` | OK |
| `DenseAlgorithmTests.test_degenerate_points_end_to_end` | OK |

### `tests/test_geom.py`

| Test | Verdict |
| --- | --- |
| `GeometryTests.test_tri3geom_single_triangle` | OK |
| `GeometryTests.test_tri3geom_multiple_faces_use_zero_based_indices` | STRENGTHENED |
| `GeometryTests.test_trisphere_subdiv_base_and_refined_sizes` | STRENGTHENED |

### `tests/test_hifde.py`

| Test | Verdict |
| --- | --- |
| `HIFDETests.test_hifde2_operations_match_sparse_matrix` | OK |
| `HIFDETests.test_hifde2x_and_hifde3x_entry_points` | OK |
| `HIFDETests.test_hifde3_positive_definite_cholesky_helpers` | OK |

### `tests/test_hifie.py`

| Test | Verdict |
| --- | --- |
| `HIFIETests.test_hifie_compression_callbacks` | OK |
| `HIFIETests.test_hifie2_operations_match_dense` | OK |
| `HIFIETests.test_hifie2x_and_hifie3_entry_points` | OK |
| `HIFIETests.test_hifie_positive_definite_cholesky_helpers` | OK |

### `tests/test_ifmm_option_parity.py`

| Test | Verdict |
| --- | --- |
| `IFMMOptionParityTests.test_store_modes_match_matlab` | OK |
| `IFMMOptionParityTests.test_near_modes_match_matlab` | OK |
| `IFMMOptionParityTests.test_symmetry_modes_match_matlab` | OK |
| `IFMMOptionParityTests.test_proxy_callback_paths_match_matlab` | OK |
| `IFMMOptionParityTests.test_rectangular_complex_matches_matlab` | OK |
| `IFMMOptionParityTests.test_mv_transpose_modes_match_matlab` | OK |
| `IFMMOptionParityTests.test_upstream_mv_expline_proxy_case_matches_matlab` | OK |

### `tests/test_matlab_parity.py`

| Test | Verdict |
| --- | --- |
| `MatlabParityTests.test_hypoct_layout_and_permutation` | OK |
| `MatlabParityTests.test_id_fixed_columns` | OK |
| `MatlabParityTests.test_hifie_compression_callbacks` | OK |
| `MatlabParityTests.test_hifie_entry_points_match_matlab` | OK |
| `MatlabParityTests.test_hifie_covariance_proxy_matches_matlab` | OK |
| `MatlabParityTests.test_rskelf_small_apply_and_solve` | OK |
| `MatlabParityTests.test_rskelf_partial_logdet` | OK |
| `MatlabParityTests.test_rskelf_partial_apply_and_solve` | OK |
| `MatlabParityTests.test_rskelf_partial_info` | OK |
| `MatlabParityTests.test_rskel_apply_and_extended_sparse` | OK |
| `MatlabParityTests.test_ifmm_small_apply_and_adjoint` | OK |
| `MatlabParityTests.test_mf2_grid_operator` | OK |
| `MatlabParityTests.test_mf2_sparse_singular_and_near_singular_modes` | OK |
| `MatlabParityTests.test_mf3_grid_operator` | OK |
| `MatlabParityTests.test_mfx_line_operator` | OK |
| `MatlabParityTests.test_mf2_hermitian_and_positive_modes` | OK |
| `MatlabParityTests.test_mfx_complex_and_symmetric_modes` | OK |
| `MatlabParityTests.test_hifde2_grid_operator` | OK |
| `MatlabParityTests.test_hifde_entry_points_match_matlab` | OK |
| `ChunkIEStyleRSkelfParityTests.test_laplace_dirichlet_starfish_rskelf_callback` | OK |
| `ChunkIEStyleRSkelfParityTests.test_helmholtz_dirichlet_starfish_rskelf_callback` | OK |

### `tests/test_matlab_parity_utils.py`

| Test | Verdict |
| --- | --- |
| `MatlabParityUtilsTests.test_matlab_path_escapes_windows_paths_and_quotes` | OK |
| `MatlabParityUtilsTests.test_matlab_script_command_defaults_to_matlab_batch_flag` | OK |
| `MatlabParityUtilsTests.test_matlab_script_command_supports_command_launcher` | OK |
| `MatlabParityUtilsTests.test_logdet_mod_error_ignores_branch_offset` | OK |
| `MatlabParityUtilsTests.test_relerr_uses_absolute_scale_for_zero_reference` | STRENGTHENED |
| `MatlabParityUtilsTests.test_require_paths_fails_loudly` | OK |
| `MatlabParityUtilsTests.test_require_flam_reference_reports_missing_entry_points` | OK |
| `MatlabParityUtilsTests.test_default_flam_reference_prefers_complete_checkout` | STRENGTHENED |
| `MatlabParityUtilsTests.test_default_chunkie_reference_prefers_repo_submodule` | STRENGTHENED |
| `MatlabParityUtilsTests.test_reference_dependency_pins_are_loaded` | OK |
| `MatlabParityUtilsTests.test_require_pinned_reference_reports_non_git_checkout` | OK |
| `MatlabParityUtilsTests.test_factor_metadata_code_covers_common_factor_fields` | STRENGTHENED |
| `MatlabParityUtilsTests.test_load_factor_metadata_simplifies_matlab_struct` | OK |

### `tests/test_mf.py`

| Test | Verdict |
| --- | --- |
| `MultifontalTests.test_mfx_mv_sv_logdet_match_dense` | OK |
| `MultifontalTests.test_mf_diag_uses_selected_unfolding_for_hierarchical_factors` | OK |
| `MultifontalTests.test_mf_spdiag_uses_sparse_propagation_for_hierarchical_factors` | OK |
| `MultifontalTests.test_mfx_complex_sparse_transpose_solves_and_logdet` | OK |
| `MultifontalTests.test_mf2_positive_definite_cholesky_helpers` | OK |
| `MultifontalTests.test_mf3_dimension_and_solve` | OK |
| `MultifontalTests.test_mf_debug_dense_fallback_is_explicit` | OK |

### `tests/test_rskel_option_parity.py`

| Test | Verdict |
| --- | --- |
| `RSkelOptionParityTests.test_unsymmetric_callback_matrix_access_matches_matlab` | OK |
| `RSkelOptionParityTests.test_symmetric_mode_matches_matlab` | OK |
| `RSkelOptionParityTests.test_hermitian_mode_matches_matlab` | OK |
| `RSkelOptionParityTests.test_positive_definite_mode_maps_to_hermitian_and_matches_matlab` | OK |
| `RSkelOptionParityTests.test_complex_rectangular_matches_matlab` | OK |
| `RSkelOptionParityTests.test_proxy_row_and_column_paths_match_matlab` | OK |
| `RSkelOptionParityTests.test_xsp_symmetric_hermitian_positive_modes_match_matlab` | OK |
| `RSkelOptionParityTests.test_mv_transpose_modes_match_matlab` | OK |
| `RSkelOptionParityTests.test_upstream_mv_line_proxy_case_matches_matlab` | OK |

### `tests/test_rskelf_option_parity.py`

| Test | Verdict |
| --- | --- |
| `RSkelfOptionParityTests.test_proxy_unsymmetric_mode_matches_matlab` | OK |
| `RSkelfOptionParityTests.test_proxy_symmetric_mode_matches_matlab` | OK |
| `RSkelfOptionParityTests.test_proxy_hermitian_mode_matches_matlab` | OK |
| `RSkelfOptionParityTests.test_proxy_positive_definite_mode_matches_matlab` | OK |
| `RSkelfOptionParityTests.test_callable_stop_matches_matlab_partial_factorization` | OK |
| `RSkelfOptionParityTests.test_mv_transpose_modes_match_matlab` | OK |
| `RSkelfOptionParityTests.test_sv_transpose_modes_match_matlab` | OK |
| `RSkelfOptionParityTests.test_diag_and_spdiag_modes_match_matlab` | OK |
| `RSkelfOptionParityTests.test_symmetric_mode_matches_matlab` | OK |
| `RSkelfOptionParityTests.test_hermitian_mode_matches_matlab` | OK |
| `RSkelfOptionParityTests.test_positive_definite_mode_matches_matlab` | OK |
| `RSkelfOptionParityTests.test_complex_unsymmetric_mode_matches_matlab` | OK |
| `RSkelfOptionParityTests.test_complex_symmetric_mode_matches_matlab` | OK |
| `RSkelfOptionParityTests.test_complex_hermitian_mode_matches_matlab` | OK |
| `RSkelfOptionParityTests.test_complex_positive_definite_mode_matches_matlab` | OK |

### `tests/test_sparse_core.py`

| Test | Verdict |
| --- | --- |
| `SparseCoreTests.test_spget_and_column_storage_helpers` | OK |
| `SparseCoreTests.test_sparse_push_helpers_expand_capacity` | OK |
| `SparseCoreTests.test_symmetry_helpers` | STRENGTHENED |
| `SparseCoreTests.test_detperm_ismemb_and_logdet_ldl` | ADDED COVERAGE |

### `tests/test_utilities.py`

| Test | Verdict |
| --- | --- |
| `UtilityTests.test_gausspdf_matches_standard_normal_value` | OK |
| `UtilityTests.test_glegquad_integrates_polynomial` | OK |
| `UtilityTests.test_gqgw_matches_legendre_rule_on_minus_one_one` | OK |
| `UtilityTests.test_quadrature_helpers_validate_inputs` | NEW |
| `UtilityTests.test_quad_sqtri3_weight_sum_is_triangle_area` | STRENGTHENED |
| `UtilityTests.test_lsedc_validates_controls` | NEW |
| `UtilityTests.test_lsedc_enforces_constraints` | STRENGTHENED |
