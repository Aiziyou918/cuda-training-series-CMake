cmake_minimum_required(VERSION 3.31)
project(homework CUDA)

set(CMAKE_CUDA_STANDARD 20)

# add_executable(homework main.cu)

# set_target_properties(homework PROPERTIES
#         CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")

add_subdirectory(hw1)
add_subdirectory(hw2)
add_subdirectory(hw3)
add_subdirectory(hw4)
add_subdirectory(hw5)
add_subdirectory(hw6)
add_subdirectory(hw7)
