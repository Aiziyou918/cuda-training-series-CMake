add_executable(matrix_sums matrix_sums.cu)

set_target_properties(matrix_sums PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)