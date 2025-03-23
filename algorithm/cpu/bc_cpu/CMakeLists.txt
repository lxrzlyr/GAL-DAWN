# Specify source
file(GLOB SOURCES_CPP "${CMAKE_SOURCE_DIR}/src/algorithm/cpu/*.cpp" "${CMAKE_SOURCE_DIR}/src/*.cpp")

# Add the executable
# add_executable(bc_cpu bc_cpu.cpp ${SOURCES_CPP})
add_executable(bc_cpu bc_cpu.cpp ${SOURCES_CPP})

# Include directories
target_include_directories(bc_cpu PUBLIC "${CMAKE_SOURCE_DIR}/include")

# Compile options
target_compile_options(bc_cpu PUBLIC -O3)

# Find and link OpenMP
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_compile_options(bc_cpu PUBLIC -fopenmp)
    target_link_libraries(bc_cpu PUBLIC OpenMP::OpenMP_CXX)
endif()

