cmake_minimum_required(VERSION 3.30)
project(lab3)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx512vl -mfma")

add_executable(lab3 main.cpp)
