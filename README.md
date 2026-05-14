# GAL-DAWN

GAL-DAWN is a DAWN-based graph-computation library with a single user-facing
executable, `dawn`, and a small public C++ API for graph loading, runtime
management, and algorithm dispatch.

## Build

CPU-only builds do not require CUDA or OpenMP:

```bash
cmake -S . -B build -DDAWN_BUILD_TESTS=ON -DDAWN_BUILD_CLI=ON -DDAWN_ENABLE_CUDA=OFF
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

OpenMP is optional. If it is not found, DAWN builds a single-thread CPU
fallback and reports that at configure time.

CUDA is opt-in:

```bash
cmake -S . -B build-cuda -DDAWN_ENABLE_CUDA=ON -DDAWN_BUILD_TESTS=ON
cmake --build build-cuda --parallel
```

## CLI

`dawn` is the primary entry point.

```bash
./build/dawn inspect --input validation/fixtures/tiny_unweighted.mtx
./build/dawn kernels --algorithm bfs
./build/dawn run --algorithm bfs --input validation/fixtures/tiny_unweighted.mtx --source 0 --backend auto
./build/dawn run --algorithm sssp --input validation/fixtures/tiny_weighted.mtx --source 0 --backend auto
./build/dawn run --algorithm mssp --input validation/fixtures/tiny_unweighted.mtx --source-list validation/fixtures/tiny_sources.txt --backend auto
```

Supported commands:

- `dawn run`
- `dawn inspect`
- `dawn kernels`
- `dawn version`

Supported algorithms:

- `bfs`
- `sssp`
- `mssp`
- `apsp`
- `cc`
- `bc`

Supported backend policies:

- `auto`
- `best`
- `cpu`
- `cuda`

In CPU-only builds, `auto` and `best` select CPU. Requesting `cuda` returns an
unavailable status.

MSSP and APSP do not write full multi-source or all-pairs output by default.
Use `--output-source N --output result.txt` to save one source row.

## Graph Input

The loader reads MatrixMarket coordinate graphs. It preserves the declared
vertex count, validates index bounds, expands `symmetric` graphs, keeps
`general` graphs directed, detects weighted inputs, and ignores self-loops by
default.

CSR storage enforces:

- `row_offsets.size() == num_vertices + 1`
- `row_offsets[0] == 0`
- `row_offsets[num_vertices] == num_edges`

Graph objects do not own runtime launch details such as source vertices,
threads, streams, block sizes, or device ids.

## Library Layers

The production dependency direction is:

```text
src/dawn.cpp
  -> src/algorithms.cpp
  -> src/runtime.cpp
  -> src/cpu_kernels.cpp and src/cpu_runner.hxx
  -> src/graph.cpp and src/io.cpp
  -> src/status.cpp
  -> include/dawn/*
  -> src/legacy/*
```

Public library APIs return `Status` or `StatusOr<T>`. CLI code owns process
exit codes.

## Quality Gates

Useful validation commands:

```bash
cmake -S . -B build -DDAWN_BUILD_TESTS=ON
cmake --build build --parallel
ctest --test-dir build --output-on-failure
ctest --test-dir build -L unit --output-on-failure
ctest --test-dir build -L smoke --output-on-failure
```

Optional checks:

```bash
cmake -S . -B build-asan -DDAWN_BUILD_TESTS=ON -DDAWN_ENABLE_SANITIZERS=ON
cmake --build build-asan --parallel
ctest --test-dir build-asan --output-on-failure

cmake -S . -B build-format -DDAWN_ENABLE_FORMAT_CHECK=ON
cmake --build build-format --target dawn_format_check
```

Correctness regression:

```bash
ctest --test-dir build -R dawn_correctness_generated --output-on-failure
ctest --test-dir build -R dawn_correctness_generated_strict --output-on-failure
```

The generated correctness harness lives under `validation/correctness/` and
records per-algorithm results independently. The strict target fails on any
correctness mismatch.

## Install And Consume

```bash
cmake -S . -B build-install -DDAWN_ENABLE_INSTALL=ON -DDAWN_BUILD_CLI=ON
cmake --build build-install --parallel
cmake --install build-install --prefix /tmp/dawn-install
```

Downstream CMake projects can use:

```cmake
find_package(dawn CONFIG REQUIRED)
target_link_libraries(my_app PRIVATE dawn::dawn)
```

## Citation

```bibtex
@inproceedings{dawn,
author = {Feng, Yelai and Wang, Huaixi and Zhu, Yining and Liu, Xiandong and Lu, Hongyi and Liu, Qing},
title = {DAWN: Matrix Operation-Optimized Algorithm for Shortest Paths Problem on Unweighted Graphs},
year = {2024},
publisher = {Association for Computing Machinery},
doi = {10.1145/3650200.3656600},
booktitle = {Proceedings of the 38th ACM International Conference on Supercomputing},
series = {ICS '24}
}
```
