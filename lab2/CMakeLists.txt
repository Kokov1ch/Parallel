cmake_minimum_required(VERSION 3.30)
project(lab2)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx512vl -mavx2")

add_executable(lab2 main.cpp)
