configure_file(
    "${startle_SOURCE_DIR}/include/startle.hpp.in"
    "${startle_SOURCE_DIR}/include/startle.hpp"
)

set(STARTLE_SOURCES startle.cxx treeio.cxx starhomoplasy.cxx flat_parsimony.cxx)
if(STARTLE_ENABLE_CUDA)
  list(APPEND STARTLE_SOURCES cuda_parsimony.cu)
else()
  list(APPEND STARTLE_SOURCES cuda_parsimony_stub.cxx)
endif()
add_executable(startle ${STARTLE_SOURCES})
if(STARTLE_ENABLE_CUDA)
  target_compile_definitions(startle PRIVATE STARTLE_ENABLE_CUDA=1)
  set_target_properties(startle PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
endif()

# add libraries
target_link_libraries(startle PRIVATE nlohmann_json::nlohmann_json)
target_link_libraries(startle PRIVATE pprint)
target_link_libraries(startle PRIVATE spdlog)
target_link_libraries(startle PRIVATE csv)
target_link_libraries(startle PRIVATE argparse)
