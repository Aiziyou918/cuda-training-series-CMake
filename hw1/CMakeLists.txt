add_executable(hello hello.cu)

set_target_properties(hello PROPERTIES
         CUDA_SEPARABLE_COMPILATION ON)

add_executable(vector_add vector_add.cu)

set_target_properties(vector_add PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

add_executable(matrix_mul matrix_mul.cu)

set_target_properties(matrix_mul PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

add_executable(matrix_mul_cpu matrix_mul_cpu.cu)

set_target_properties(matrix_mul_cpu PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)