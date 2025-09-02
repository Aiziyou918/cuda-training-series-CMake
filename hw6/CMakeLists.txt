add_executable(array_inc array_inc.cu)

set_target_properties(array_inc PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

add_executable(linked_list linked_list.cu)

set_target_properties(linked_list PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
