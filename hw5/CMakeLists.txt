add_executable(matrix_sums_new matrix_sums.cu)

set_target_properties(matrix_sums_new PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

add_executable(max_reduction max_reduction.cu)

set_target_properties(max_reduction PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

add_executable(reductions reductions.cu)

set_target_properties(reductions PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)