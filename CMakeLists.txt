cmake_minimum_required(VERSION 3.15)
project(example_project LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(cpp/pybind11)  # repo complet de pybind11

pybind11_add_module(finance
    cpp/total.cpp
    cpp/eval.cpp
    cpp/tree.cpp
    cpp/monte_carlo.cpp
    
)

