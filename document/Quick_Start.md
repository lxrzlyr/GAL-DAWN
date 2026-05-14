# Quick Start

This guide gets a user from a fresh checkout to a working GAL-DAWN command
line run. The production entry point is `dawn`.

## Requirements

- CMake 3.16 or newer.
- A C++14 compiler.
- CUDA is optional and disabled by default.
- OpenMP is optional. If OpenMP is not found, DAWN builds a single-thread CPU
  fallback.

## Build And Test

```bash
cmake -S . -B build -DDAWN_BUILD_TESTS=ON -DDAWN_BUILD_CLI=ON -DDAWN_ENABLE_CUDA=OFF
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

The executable is written to:

```bash
./build/dawn
```

## Inspect A Graph

```bash
./build/dawn inspect --input validation/fixtures/tiny_unweighted.mtx
```

This prints the vertex count, edge count, directed and weighted flags, CSR
format, and degree summary.

## List Kernels

```bash
./build/dawn kernels
./build/dawn kernels --algorithm bfs
```

CPU kernels are available in the default CPU-only build. CUDA kernels require
`DAWN_ENABLE_CUDA=ON`.

## Run Algorithms

Breadth-first search on an unweighted graph:

```bash
./build/dawn run --algorithm bfs --input validation/fixtures/tiny_unweighted.mtx --source 0
```

Single-source shortest path on a weighted graph:

```bash
./build/dawn run --algorithm sssp --input validation/fixtures/tiny_weighted.mtx --source 0
```

Multi-source shortest path:

```bash
./build/dawn run --algorithm mssp --input validation/fixtures/tiny_unweighted.mtx --source-list validation/fixtures/tiny_sources.txt
```

All-pairs shortest path:

```bash
./build/dawn run --algorithm apsp --input validation/fixtures/tiny_unweighted.mtx
```

Closeness centrality:

```bash
./build/dawn run --algorithm cc --input validation/fixtures/tiny_unweighted.mtx --source 0
```

Betweenness centrality:

```bash
./build/dawn run --algorithm bc --input validation/fixtures/tiny_unweighted.mtx
```

## Save Output

For BFS, SSSP, and BC, pass `--output`:

```bash
./build/dawn run --algorithm bfs --input validation/fixtures/tiny_unweighted.mtx --source 0 --output /tmp/bfs.out
```

MSSP and APSP do not save full multi-source or all-pairs output by default. To
save one source row:

```bash
./build/dawn run --algorithm mssp --input validation/fixtures/tiny_unweighted.mtx --source-list validation/fixtures/tiny_sources.txt --output-source 0 --output /tmp/mssp_0.out
./build/dawn run --algorithm apsp --input validation/fixtures/tiny_unweighted.mtx --output-source 0 --output /tmp/apsp_0.out
```

## Input Format

GAL-DAWN reads MatrixMarket coordinate matrices:

```text
%%MatrixMarket matrix coordinate pattern symmetric
4 4 3
1 2
2 3
3 4
```

- `pattern` inputs are unweighted.
- `real` inputs are weighted.
- `symmetric` inputs are expanded to undirected CSR.
- `general` inputs are kept directed.
- Self-loops are ignored by default.

## CUDA Build

```bash
cmake -S . -B build-cuda -DDAWN_ENABLE_CUDA=ON -DDAWN_BUILD_TESTS=ON -DDAWN_CUDA_ARCHITECTURES=86
cmake --build build-cuda --parallel
```

CPU-only builds are sufficient on machines without CUDA.
