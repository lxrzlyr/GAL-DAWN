find_program(CLANG_FORMAT_EXE NAMES clang-format)
if(NOT CLANG_FORMAT_EXE)
  message(FATAL_ERROR "clang-format was not found")
endif()

if(NOT DAWN_FORMAT_SOURCE_DIR)
  message(FATAL_ERROR "DAWN_FORMAT_SOURCE_DIR was not provided")
endif()

file(GLOB_RECURSE DAWN_FORMAT_FILES
    LIST_DIRECTORIES false
    ${DAWN_FORMAT_SOURCE_DIR}/cli/*.cpp
    ${DAWN_FORMAT_SOURCE_DIR}/include/dawn/*.cuh
    ${DAWN_FORMAT_SOURCE_DIR}/include/dawn/*.hxx
    ${DAWN_FORMAT_SOURCE_DIR}/include/dawn/**/*.cuh
    ${DAWN_FORMAT_SOURCE_DIR}/include/dawn/**/*.hxx
    ${DAWN_FORMAT_SOURCE_DIR}/src/*.cpp
    ${DAWN_FORMAT_SOURCE_DIR}/src/**/*.cu
    ${DAWN_FORMAT_SOURCE_DIR}/src/**/*.cuh
    ${DAWN_FORMAT_SOURCE_DIR}/src/**/*.cpp
    ${DAWN_FORMAT_SOURCE_DIR}/src/**/*.hxx
    ${DAWN_FORMAT_SOURCE_DIR}/validation/unit/*.cpp)

execute_process(
  COMMAND ${CLANG_FORMAT_EXE} --dry-run --Werror ${DAWN_FORMAT_FILES}
  RESULT_VARIABLE FORMAT_RESULT)

if(NOT FORMAT_RESULT EQUAL 0)
  message(FATAL_ERROR "clang-format check failed")
endif()
