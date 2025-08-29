add_executable(vector_add_new vector_add_new.cu)

set_target_properties(vector_add_new PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)