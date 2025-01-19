cmake_minimum_required(VERSION 3.24)
project(lab4)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")

add_executable(lab4 main.cpp vector_mod.cpp randomize.cpp test.cpp num_threads.cpp mod_ops.cpp performance.cpp)
