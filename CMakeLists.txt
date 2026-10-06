cmake_minimum_required(VERSION 3.16)
project(local_sparse_linalg CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Header-only core library: CSR storage + CG solver (self-implemented).
add_library(sparse_linalg INTERFACE)
target_include_directories(sparse_linalg INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Eigen is used ONLY as a test reference, never in the core algorithms.
find_package(Eigen3 QUIET)
if(NOT Eigen3_FOUND)
    set(EIGEN3_HINTS
        /usr/include/eigen3
        /usr/local/include/eigen3
        /tmp/eigen_extract/usr/include/eigen3)
    find_path(EIGEN3_INCLUDE_DIR Eigen/Dense HINTS ${EIGEN3_HINTS})
    if(NOT EIGEN3_INCLUDE_DIR)
        message(FATAL_ERROR "Eigen3 not found; set EIGEN3_INCLUDE_DIR")
    endif()
    add_library(eigen_iface INTERFACE)
    target_include_directories(eigen_iface INTERFACE ${EIGEN3_INCLUDE_DIR})
else()
    add_library(eigen_iface INTERFACE)
    target_link_libraries(eigen_iface INTERFACE Eigen3::Eigen)
endif()

enable_testing()
add_executable(sparse_tests tests/test_main.cpp)
target_link_libraries(sparse_tests PRIVATE sparse_linalg eigen_iface)
target_compile_options(sparse_tests PRIVATE -Wall -Wextra -O2)
add_test(NAME sparse_tests COMMAND sparse_tests)
