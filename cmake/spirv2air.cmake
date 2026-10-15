# Embeddable targets, kept compatible with callers that include this file directly.
get_filename_component(SPIRV2AIR_ROOT "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
# SPIRV2AIR_BUILD_TRANSLATOR=OFF builds only the AIR libraries (airbc, mtlb) and needs no SPIRV-Headers.
if(NOT DEFINED SPIRV2AIR_BUILD_TRANSLATOR)
  set(SPIRV2AIR_BUILD_TRANSLATOR ON)
endif()
if(SPIRV2AIR_BUILD_TRANSLATOR AND NOT SPIRV2AIR_SPIRV_HEADERS_DIR)
  find_path(SPIRV2AIR_SPIRV_HEADERS_DIR spirv/unified1/spirv.hpp
            PATHS /opt/homebrew/include /usr/local/include REQUIRED)
endif()
option(SPIRV2AIR_ENABLE_VERIFY "Verify IR structure and dominance after optimization" OFF)

if(NOT TARGET mtlb)
  add_library(mtlb STATIC "${SPIRV2AIR_ROOT}/src/metallib/metallib.cpp")
  target_include_directories(mtlb PUBLIC
    $<BUILD_INTERFACE:${SPIRV2AIR_ROOT}/src> $<INSTALL_INTERFACE:include>)
  target_compile_features(mtlb PUBLIC cxx_std_17)
endif()
if(NOT TARGET airbc)
  add_library(airbc STATIC "${SPIRV2AIR_ROOT}/src/airbc/airbc.cpp")
  target_include_directories(airbc PUBLIC
    $<BUILD_INTERFACE:${SPIRV2AIR_ROOT}/src/airbc> $<INSTALL_INTERFACE:include/airbc>)
  target_compile_features(airbc PUBLIC cxx_std_17)
endif()
if(SPIRV2AIR_BUILD_TRANSLATOR AND NOT TARGET spirv2air)
  add_library(spirv2air STATIC
    "${SPIRV2AIR_ROOT}/src/spirv2air/translate.cpp"
    "${SPIRV2AIR_ROOT}/src/spirv2air/mem2reg.cpp"
    "${SPIRV2AIR_ROOT}/src/spirv2air/scalarize.cpp"
    "${SPIRV2AIR_ROOT}/src/spirv2air/cse.cpp"
    "${SPIRV2AIR_ROOT}/src/spirv2air/instsimplify.cpp"
    "${SPIRV2AIR_ROOT}/src/spirv2air/simplifycfg.cpp"
    "${SPIRV2AIR_ROOT}/src/spirv2air/jumpthread.cpp"
    "${SPIRV2AIR_ROOT}/src/spirv2air/cleanup.cpp")
  target_include_directories(spirv2air PUBLIC
    $<BUILD_INTERFACE:${SPIRV2AIR_ROOT}/src/spirv2air> $<INSTALL_INTERFACE:include/spirv2air>
    PRIVATE "${SPIRV2AIR_ROOT}/src" "${SPIRV2AIR_SPIRV_HEADERS_DIR}")
  target_compile_features(spirv2air PUBLIC cxx_std_17)
  target_compile_definitions(spirv2air PRIVATE $<$<BOOL:${SPIRV2AIR_ENABLE_VERIFY}>:SPIRV2AIR_ENABLE_VERIFY>)
  target_link_libraries(spirv2air PRIVATE airbc mtlb)
endif()
foreach(name mtlb airbc spirv2air)
  if(TARGET ${name} AND NOT TARGET spirv2air::${name})
    add_library(spirv2air::${name} ALIAS ${name})
  endif()
endforeach()
