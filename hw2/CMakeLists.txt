add_executable(matrix_mul_shared matrix_mul_shared.cu)

set_target_properties(matrix_mul_shared PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)


add_executable(stencil_1d stencil_1d.cu)

set_target_properties(stencil_1d PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)