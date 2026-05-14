add_executable(dawn_stage1_unit
    unit/stage1_unit.cpp)
target_link_libraries(dawn_stage1_unit PRIVATE dawn_algorithms_cpu)
dawn_apply_target_options(dawn_stage1_unit)
add_test(NAME dawn_stage1_unit COMMAND dawn_stage1_unit
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}")
set_tests_properties(dawn_stage1_unit PROPERTIES LABELS "unit")

add_executable(dawn_stage2_unit
    unit/stage2_unit.cpp)
target_link_libraries(dawn_stage2_unit PRIVATE dawn_algorithms_cpu)
dawn_apply_target_options(dawn_stage2_unit)
add_test(NAME dawn_stage2_unit COMMAND dawn_stage2_unit
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}")
set_tests_properties(dawn_stage2_unit PROPERTIES LABELS "unit")

add_executable(dawn_stage3_quality
    unit/stage3_quality.cpp)
target_link_libraries(dawn_stage3_quality PRIVATE dawn_runtime)
dawn_apply_target_options(dawn_stage3_quality)
add_test(NAME dawn_stage3_quality COMMAND dawn_stage3_quality
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}")
set_tests_properties(dawn_stage3_quality PROPERTIES LABELS "unit;quality")

if(TARGET dawn_cli)
  add_test(NAME dawn_cli_inspect
      COMMAND $<TARGET_FILE:dawn_cli> inspect --input validation/fixtures/tiny_unweighted.mtx
      WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}")
  set_tests_properties(dawn_cli_inspect PROPERTIES LABELS "smoke")

  add_test(NAME dawn_cli_kernels
      COMMAND $<TARGET_FILE:dawn_cli> kernels --algorithm bfs
      WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}")
  set_tests_properties(dawn_cli_kernels PROPERTIES LABELS "smoke")

  add_test(NAME dawn_cli_run_bfs
      COMMAND $<TARGET_FILE:dawn_cli> run --algorithm bfs --input validation/fixtures/tiny_unweighted.mtx --source 0 --backend auto
      WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}")
  set_tests_properties(dawn_cli_run_bfs PROPERTIES LABELS "smoke")

  add_test(NAME dawn_cli_run_sssp
      COMMAND $<TARGET_FILE:dawn_cli> run --algorithm sssp --input validation/fixtures/tiny_weighted.mtx --source 0 --backend auto
      WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}")
  set_tests_properties(dawn_cli_run_sssp PROPERTIES LABELS "smoke")

  add_test(NAME dawn_cli_run_mssp
      COMMAND $<TARGET_FILE:dawn_cli> run --algorithm mssp --input validation/fixtures/tiny_unweighted.mtx --source-list validation/fixtures/tiny_sources.txt --backend auto
      WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}")
  set_tests_properties(dawn_cli_run_mssp PROPERTIES LABELS "smoke")

  add_test(NAME dawn_cli_run_cc
      COMMAND $<TARGET_FILE:dawn_cli> run --algorithm cc --input validation/fixtures/tiny_unweighted.mtx --source 0 --backend auto
      WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}")
  set_tests_properties(dawn_cli_run_cc PROPERTIES LABELS "smoke")

  add_test(NAME dawn_cli_run_bc
      COMMAND $<TARGET_FILE:dawn_cli> run --algorithm bc --input validation/fixtures/tiny_unweighted.mtx --backend auto
      WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}")
  set_tests_properties(dawn_cli_run_bc PROPERTIES LABELS "smoke")

  add_test(NAME dawn_cli_cuda_unavailable
      COMMAND $<TARGET_FILE:dawn_cli> run --algorithm bfs --input validation/fixtures/tiny_unweighted.mtx --source 0 --backend cuda
      WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}")
  if(NOT DAWN_ENABLE_CUDA)
    set_tests_properties(dawn_cli_cuda_unavailable PROPERTIES
        WILL_FAIL TRUE
        LABELS "cuda")
  else()
    set_tests_properties(dawn_cli_cuda_unavailable PROPERTIES LABELS "cuda")
  endif()

  find_package(Python3 COMPONENTS Interpreter)
  if(Python3_Interpreter_FOUND)
    add_test(NAME dawn_correctness_generated
        COMMAND "${Python3_EXECUTABLE}"
            "${PROJECT_SOURCE_DIR}/validation/correctness/generate_and_check.py"
            --dawn $<TARGET_FILE:dawn_cli>
            --workdir "${CMAKE_CURRENT_BINARY_DIR}/correctness"
        WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}")
    set_tests_properties(dawn_correctness_generated PROPERTIES LABELS "correctness;regression")

    add_test(NAME dawn_correctness_generated_strict
        COMMAND "${Python3_EXECUTABLE}"
            "${PROJECT_SOURCE_DIR}/validation/correctness/generate_and_check.py"
            --dawn $<TARGET_FILE:dawn_cli>
            --workdir "${CMAKE_CURRENT_BINARY_DIR}/correctness-strict"
            --strict
        WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}")
    set_tests_properties(dawn_correctness_generated_strict PROPERTIES
        LABELS "correctness;regression;strict")
  endif()
endif()
