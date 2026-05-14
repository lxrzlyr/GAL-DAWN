cmake_minimum_required(VERSION 3.16)

project(dawn_downstream_smoke LANGUAGES CXX)

find_package(dawn CONFIG REQUIRED)

add_executable(dawn_downstream_smoke main.cpp)
target_link_libraries(dawn_downstream_smoke PRIVATE dawn::dawn)
