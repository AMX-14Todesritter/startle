cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-Wall -Wextra -Wno-unused-parameter")
set(CMAKE_CXX_FLAGS_DEBUG "-g -pg --coverage")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

project(startle VERSION 1.0 LANGUAGES CXX)
option(STARTLE_ENABLE_CUDA "Build CUDA batch scorer" OFF)
if(STARTLE_ENABLE_CUDA)
  enable_language(CUDA)
endif()

add_subdirectory(src/)

target_include_directories(startle PUBLIC "${startle_SOURCE_DIR}/include")
target_include_directories(startle PUBLIC "${startle_SOURCE_DIR}/third-party/csv-parser/include")

add_subdirectory(third-party/spdlog)
add_subdirectory(third-party/pprint)
add_subdirectory(third-party/csv-parser)
add_subdirectory(third-party/argparse)
add_subdirectory(third-party/json)

enable_testing()

add_test(
  NAME flat_parsimony_validation
  COMMAND startle small
    "${startle_SOURCE_DIR}/examples/n100_m30_d0.2_s0_p0.2_character_matrix.csv"
    "${startle_SOURCE_DIR}/examples/n100_m30_d0.2_s0_p0.2_mutation_prior.csv"
    "${startle_SOURCE_DIR}/examples/n100_m30_d0.2_s0_p0.2_tree.newick"
    --output "${startle_BINARY_DIR}/flat_parsimony_validation"
    --validate-flat
)

add_test(
  NAME flat_parsimony_validation_unweighted
  COMMAND startle small
    "${startle_SOURCE_DIR}/examples/n100_m30_d0.2_s0_p0.2_character_matrix.csv"
    "${startle_SOURCE_DIR}/examples/n100_m30_d0.2_s0_p0.2_mutation_prior.csv"
    "${startle_SOURCE_DIR}/examples/n100_m30_d0.2_s0_p0.2_tree.newick"
    --output "${startle_BINARY_DIR}/flat_parsimony_validation_unweighted"
    --unweighted
    --validate-flat
)
