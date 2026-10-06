cmake_minimum_required(VERSION 3.16)
project(sparse_linalg CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(sparse_linalg INTERFACE)
target_include_directories(sparse_linalg INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(test_sparse_linalg tests/test_sparse_linalg.cpp)
target_link_libraries(test_sparse_linalg PRIVATE sparse_linalg)
target_compile_options(test_sparse_linalg PRIVATE -O2 -Wall -Wextra)

enable_testing()
add_test(NAME sparse_linalg COMMAND test_sparse_linalg)
