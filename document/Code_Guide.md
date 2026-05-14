# Code Guide

This guide explains how to extend GAL-DAWN 2.x. New development should
integrate with the production library layers and the unified `dawn` CLI.

## Repository Layers

The production dependency direction is:

```text
cli/dawn.cpp
  -> include/dawn/algorithms + src/algorithms
  -> include/dawn/runtime + src/runtime
  -> src/adapters and src/dawn_internal adapters
  -> src/algorithm/cpu, src/algorithm/gpu, and src/dawn_internal kernels
  -> include/dawn/graph + src/graph
  -> include/dawn/io + src/io
  -> include/dawn/memory + src/memory
  -> include/dawn/common + src/common
```

Keep new code inside this direction. Lower layers should not depend on higher
layers.

## Development Rules

- Public APIs return `DAWN::Status` or `DAWN::StatusOr<T>`.
- Prefer `DAWN::VertexId`, `DAWN::EdgeId`, and `DAWN::Weight` over raw integer
  types in new public code.
- Use `DAWN::HostCsrGraph` or `DAWN::CsrView` for graph input.
- Keep CLI parsing in `cli/dawn.cpp`; do not add new top-level per-algorithm
  user binaries.
- Keep CUDA optional behind `DAWN_ENABLE_CUDA` and `DAWN_HAS_CUDA`.
- Add tests under `validation/`.
- Keep long-running performance experiments outside the production tree unless
  they have an explicit build and validation path.

## Adding A New Algorithm Operator

The recommended path is:

1. Add or modernize the kernel implementation.
2. Add an operator-level options struct and `RunXxx` function.
3. Register a kernel descriptor.
4. Add CLI dispatch.
5. Add validation coverage.

The examples below use `pagerank` as a placeholder.

### 1. Add A Public Operator Declaration

Edit `include/dawn/algorithms/operators.hxx`.

```cpp
namespace DAWN {
namespace Algorithms {

struct PagerankOptions {
  int max_iterations = 20;
  double tolerance = 1e-6;
  std::string output_path;
  std::string kernel_name;
};

using PagerankResult = AlgorithmResult;

StatusOr<PagerankResult> RunPagerank(
    const HostCsrGraph& graph,
    const PagerankOptions& options,
    dawnHandle_t handle);

}  // namespace Algorithms
}  // namespace DAWN
```

If the algorithm fits the generic CLI dispatcher, also extend `RunAlgorithm`.

### 2. Implement The Operator

Edit `src/algorithms/operators.cpp`.

An operator should:

- validate input graph properties;
- validate option ranges;
- call `ResolveKernel`;
- create or adapt graph views for the kernel;
- return `StatusOr<AlgorithmResult>`;
- avoid `exit`, process-global state, or direct CLI parsing.

Skeleton:

```cpp
StatusOr<PagerankResult> RunPagerank(
    const HostCsrGraph& graph,
    const PagerankOptions& options,
    dawnHandle_t handle) {
  if (graph.num_vertices() == 0) {
    return Status(StatusCode::kInvalidArgument, "PageRank requires vertices");
  }

  SelectedDevice selected;
  StatusOr<KernelDescriptor> kernel =
      ResolveKernel("pagerank", graph, 1, options.kernel_name, handle,
                    &selected);
  if (!kernel.ok()) {
    return kernel.status();
  }

  RunResult run = RunYourKernel(graph.view(), options);
  if (!run.success) {
    return Status(StatusCode::kInvalidArgument, run.error_msg);
  }
  return FromRunResult(run, kernel.value(), selected.backend);
}
```

### 3. Add Kernel Registration

Edit `src/runtime/kernel_registry.cpp`.

```cpp
registry.push_back(
    {"cpu_pagerank", "pagerank", BackendKind::kCpu, true, true});
```

Set `supports_weighted` and `supports_unweighted` to match the actual kernel.
If CUDA support exists, add CUDA descriptors behind `#if DAWN_HAS_CUDA`.

### 4. Add CLI Support

Edit `cli/dawn.cpp`.

Add the algorithm name to help text and parse any new options. Then extend the
call to `DAWN::Algorithms::RunAlgorithm` or call a dedicated operator function.

CLI behavior should remain consistent:

- invalid input prints a `Status` message and returns failure;
- successful runs print `algorithm`, `backend`, `kernel`, and
  `elapsed_seconds`;
- large outputs should require an explicit `--output` option.

### 5. Add Tests

Use the validation layout:

```text
validation/
  unit/
  fixtures/
  correctness/
  downstream/
```

Add fast C++ unit tests when validating API behavior, parsing, or data
structure invariants.

Add generated correctness checks in
`validation/correctness/generate_and_check.py` when Python can compute a clear
reference result.

The generated correctness suite should:

- use deterministic graphs;
- cover directed/undirected and weighted/unweighted cases when applicable;
- keep runtime short;
- report each algorithm independently.

### 6. Update Documentation

Update:

- `document/Documentation.md` for the new operator reference;
- `document/Quick_Start.md` if users need a new command example;
- `README.md` only if the public surface changes.

Do not add internal planning documents to `document/`. Temporary release notes,
drafts, and private checklists belong under `tmp/`, which is ignored by Git.

## Adding A New Data Type

For public data types:

1. Add declarations under `include/dawn/<layer>/`.
2. Add implementations under `src/<layer>/`.
3. Keep ownership explicit: owning types should be move-only unless copying is
   intentionally supported.
4. Add validation methods for structural invariants.
5. Add unit coverage in `validation/unit/`.
6. Document the type in `document/Documentation.md`.

## Adding A New Input Format

The current production reader is MatrixMarket. To add another format:

1. Add a public header under `include/dawn/io/`.
2. Implement the reader under `src/io/`.
3. Return `StatusOr<HostCsrGraph>`.
4. Preserve graph metadata: vertex count, directed flag, weighted flag, and CSR
   invariants.
5. Add fixtures under `validation/fixtures/`.
6. Add unit tests in `validation/unit/`.

## Adding A CUDA Path

CUDA code must remain optional.

- Guard public CUDA-only storage with `#if DAWN_HAS_CUDA`.
- Add CUDA sources only under the `DAWN_ENABLE_CUDA` branch in `CMakeLists.txt`.
- Link CUDA targets to `CUDA::cudart`.
- Keep CPU-only builds working on machines without CUDA.

## Validation Commands

Run these before proposing a change:

```bash
cmake -S . -B build -DDAWN_BUILD_TESTS=ON -DDAWN_BUILD_CLI=ON -DDAWN_ENABLE_CUDA=OFF
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

Optional checks:

```bash
cmake -S . -B build-format -DDAWN_ENABLE_FORMAT_CHECK=ON
cmake --build build-format --target dawn_format_check
```

Install/export smoke:

```bash
cmake -S . -B build-install -DDAWN_ENABLE_INSTALL=ON -DDAWN_BUILD_CLI=ON
cmake --build build-install --parallel
cmake --install build-install --prefix /tmp/dawn-install
cmake -S validation/downstream -B /tmp/dawn-downstream-build -DCMAKE_PREFIX_PATH=/tmp/dawn-install
cmake --build /tmp/dawn-downstream-build --parallel
```
