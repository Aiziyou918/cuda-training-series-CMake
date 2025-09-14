add_executable(multi multi.cu)

set_target_properties(multi PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)


add_executable(overlap overlap.cu)

set_target_properties(overlap PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)