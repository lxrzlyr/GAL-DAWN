# GAL-DAWN Library Reference

This document describes the public GAL-DAWN 2.x library surface. The structure
is modeled after operator-oriented numerical library references: data types
first, then algorithm operators grouped by computational complexity.

## Header Overview

Core public headers:

```cpp
#include <dawn/common/status.hxx>
#include <dawn/common/types.hxx>
#include <dawn/graph/csr_graph.hxx>
#include <dawn/io/matrix_market.hxx>
#include <dawn/runtime/handle.hxx>
#include <dawn/runtime/device_selector.hxx>
#include <dawn/runtime/kernel_registry.hxx>
#include <dawn/algorithms/operators.hxx>
```

Raw-pointer kernel headers live under `src/dawn_internal/` and are not installed
as public API. Library users should prefer
`include/dawn/algorithms/operators.hxx`.

## Status Types

### `DAWN::StatusCode`

Error category enum:

- `kOk`
- `kInvalidArgument`
- `kOutOfRange`
- `kNotFound`
- `kUnavailable`
- `kUnsupported`
- `kInternal`

### `DAWN::Status`

Represents success or failure.

Important members:

```cpp
bool ok() const;
StatusCode code() const;
const std::string& message() const;
std::string ToString() const;
```

### `DAWN::StatusOr<T>`

Return wrapper for functions that either produce a value or fail.

Important members:

```cpp
bool ok() const;
const Status& status() const;
T& value();
const T& value() const;
```

## Scalar Types

Defined in `dawn/common/types.hxx`.

| Type | Definition | Meaning |
|---|---:|---|
| `DAWN::VertexId` | `int32_t` | Vertex identifier. |
| `DAWN::EdgeId` | `int64_t` | Edge count or CSR offset. |
| `DAWN::Weight` | `float` | Edge weight and weighted distance value. |

The library also provides checked downcast helpers for kernels that use `int`
internally:

```cpp
StatusOr<int> checked_int_downcast(EdgeId value, const char* field_name);
StatusOr<int> checked_int_downcast(VertexId value, const char* field_name);
```

## Memory Types

Defined in `dawn/memory/buffer.hxx` and `dawn/memory/span.hxx`.

### `DAWN::MemorySpace`

```cpp
enum class MemorySpace { kHost, kDevice, kPinned };
```

`kDevice` and `kPinned` storage are available only when DAWN is built with CUDA.

### `DAWN::Buffer<T, MemorySpace::kHost>`

Move-only owning host buffer.

Common operations:

```cpp
T* data();
const T* data() const;
size_t size() const;
bool empty() const;
void fill(const T& value);
void reset();
T* release();
```

Alias:

```cpp
template <typename T>
using HostBuffer = Buffer<T, MemorySpace::kHost>;
```

### `DAWN::Span<T>` and `DAWN::ConstSpan<T>`

Non-owning contiguous view:

```cpp
Span(T* data, size_t size);
T* data() const;
size_t size() const;
bool empty() const;
```

## Graph Types

Defined in `dawn/graph/csr_graph.hxx`.

### `DAWN::GraphStorageFormat`

Currently supported:

```cpp
enum class GraphStorageFormat { kCsr };
```

### `DAWN::GraphMetadata`

```cpp
struct GraphMetadata {
  VertexId num_vertices;
  EdgeId num_edges;
  bool directed;
  bool weighted;
  GraphStorageFormat storage_format;
};
```

### `DAWN::CsrView`

Non-owning CSR graph view:

```cpp
struct CsrView {
  const EdgeId* row_offsets;
  const VertexId* column_indices;
  const Weight* weights;
  VertexId num_vertices;
  EdgeId num_edges;
  bool directed;
  bool weighted;
};
```

For unweighted graphs, `weights` is `nullptr`.

### `DAWN::HostCsrGraph`

Move-only owning host CSR graph.

Important members:

```cpp
static StatusOr<HostCsrGraph> Create(
    GraphMetadata metadata,
    HostBuffer<EdgeId> row_offsets,
    HostBuffer<VertexId> column_indices,
    HostBuffer<Weight> weights);

const GraphMetadata& metadata() const;
VertexId num_vertices() const;
EdgeId num_edges() const;
bool directed() const;
bool weighted() const;
CsrView view() const;
Status Validate() const;
```

CSR invariants:

- `row_offsets.size() == num_vertices + 1`
- `row_offsets[0] == 0`
- `row_offsets[num_vertices] == num_edges`
- `column_indices.size() == num_edges`
- weighted graphs have `weights.size() == num_edges`
- unweighted graphs have an empty weights buffer

## MatrixMarket I/O

Defined in `dawn/io/matrix_market.hxx`.

### `DAWN::IO::MatrixMarketOptions`

```cpp
struct MatrixMarketOptions {
  bool ignore_self_loops = true;
};
```

### `DAWN::IO::ReadMatrixMarket`

```cpp
StatusOr<HostCsrGraph> ReadMatrixMarket(
    const std::string& input_path,
    const MatrixMarketOptions& options = MatrixMarketOptions());
```

Input support:

- MatrixMarket coordinate matrices.
- `pattern` field for unweighted graphs.
- `real`, `integer`, or numeric fields for weighted graphs.
- `symmetric` matrices become undirected graphs.
- `general` matrices remain directed graphs.
- self-loops are ignored by default.

### `DAWN::IO::DegreeSummary`

```cpp
struct DegreeSummary {
  EdgeId min_degree;
  EdgeId max_degree;
  double average_degree;
};
```

```cpp
DegreeSummary ComputeDegreeSummary(CsrView view);
```

## Runtime Types

Defined in `dawn/runtime/handle.hxx`.

### `DAWN::BackendKind`

```cpp
enum class BackendKind { kCpu, kCuda };
```

### `DAWN::BackendPolicy`

```cpp
enum class BackendPolicy { kAuto, kBest, kCpu, kCuda };
```

### `DAWN::RuntimeOptions`

```cpp
struct RuntimeOptions {
  DevicePolicy device_policy;
  CpuThreadingPolicy cpu_threading;
};
```

### Runtime Handle

```cpp
using dawnHandle_t = dawnContext*;

StatusOr<dawnHandle_t> dawnCreate(const RuntimeOptions& options);
Status dawnDestroy(dawnHandle_t handle);
Status dawnSetBackendPolicy(dawnHandle_t handle, BackendPolicy policy);
Status dawnSetDeviceId(dawnHandle_t handle, int device_id);
```

The CLI creates a handle per command. Library users should destroy handles with
`dawnDestroy`.

## Kernel Registry

Defined in `dawn/runtime/kernel_registry.hxx`.

### `DAWN::KernelDescriptor`

```cpp
struct KernelDescriptor {
  std::string name;
  std::string algorithm;
  BackendKind backend;
  bool supports_weighted;
  bool supports_unweighted;
};
```

### Registry Functions

```cpp
GraphProfile ProfileGraph(CsrView view, int source_count);
std::vector<KernelDescriptor> BuildDefaultKernelRegistry();
std::vector<KernelDescriptor> ListKernelsForAlgorithm(
    const std::vector<KernelDescriptor>& registry,
    const std::string& algorithm);
StatusOr<KernelDescriptor> SelectKernel(
    const std::vector<KernelDescriptor>& registry,
    const std::string& algorithm,
    BackendKind backend,
    const GraphProfile& profile,
    const std::string& explicit_kernel = "");
```

Current CPU kernel names:

| Kernel | Algorithm | Weighted | Unweighted |
|---|---|---:|---:|
| `cpu_bfs` | `bfs` | no | yes |
| `cpu_sssp` | `sssp` | yes | no |
| `cpu_mssp` | `mssp` | yes | yes |
| `cpu_apsp` | `apsp` | yes | yes |
| `cpu_cc` | `cc` | yes | yes |
| `cpu_bc` | `bc` | no | yes |

## Algorithm Result Type

Defined in `dawn/algorithms/operators.hxx`.

```cpp
struct AlgorithmResult {
  double elapsed_seconds;
  double value;
  std::string kernel_name;
  BackendKind backend;
};
```

- `elapsed_seconds`: measured runtime from the selected operator.
- `value`: scalar result for operators such as closeness centrality.
- `kernel_name`: selected kernel descriptor name.
- `backend`: selected backend.

## Algorithm Operators By Complexity

Complexities below are expressed with `n = |V|`, `m = |E|`, and `k = number of
sources`. They describe the intended graph algorithm category and adapter
surface, not a guarantee of hardware-specific performance.

### Level 1: Single-Source Traversal And Shortest Path

These operators are below `O(n^2)` on sparse graphs.

#### BFS

Purpose:

Breadth-first search from one source on an unweighted graph.

Function:

```cpp
StatusOr<BfsResult> RunBfs(
    const HostCsrGraph& graph,
    const BfsOptions& options,
    dawnHandle_t handle);
```

Options:

```cpp
struct BfsOptions {
  VertexId source = 0;
  std::string output_path;
  std::string kernel_name;
};
```

Input graph:

- CSR graph.
- Unweighted.
- Directed or undirected.

Output:

- `BfsResult` (`AlgorithmResult`).
- Optional output file lines: `source vertex distance`.
- The source vertex and unreachable vertices are omitted from the output file.

Complexity class:

- `O(n + m)`.

#### SSSP

Purpose:

Single-source shortest path from one source on a weighted graph.

Function:

```cpp
StatusOr<SsspResult> RunSssp(
    const HostCsrGraph& graph,
    const SsspOptions& options,
    dawnHandle_t handle);
```

Options:

```cpp
struct SsspOptions {
  VertexId source = 0;
  std::string output_path;
  std::string kernel_name;
};
```

Input graph:

- CSR graph.
- Weighted.
- Directed or undirected.

Output:

- `SsspResult` (`AlgorithmResult`).
- Optional output file lines: `source vertex distance`.
- Infinite/unreachable distances are omitted.

Complexity class:

- Up to `O(nm)` in the current CPU backend.
- Below `O(n^3)` for sparse graphs where `m = O(n)`.

### Level 2: Multi-Source And Whole-Graph Operators

These operators are at most `O(n^3)` in the current supported surface.

#### MSSP

Purpose:

Shortest paths from a user-provided source list.

Function:

```cpp
StatusOr<MsspResult> RunMssp(
    const HostCsrGraph& graph,
    const MsspOptions& options,
    dawnHandle_t handle);
```

Options:

```cpp
struct MsspOptions {
  std::vector<VertexId> sources;
  std::string output_path;
  std::string kernel_name;
  VertexId output_source = -1;
};
```

Input graph:

- CSR graph.
- Weighted or unweighted.
- Directed or undirected.
- `sources` must be non-empty.

Output:

- `MsspResult` (`AlgorithmResult`).
- No file is written by default, even when multiple sources are evaluated.
- To save one source row, set both `output_path` and `output_source`.
- `output_source` must be present in `sources`.

Complexity class:

- `O(k(n + m))` for unweighted graphs.
- Up to `O(knm)` for weighted graphs in the current CPU backend.

#### APSP

Purpose:

Shortest paths from all non-isolated sources.

Function:

```cpp
StatusOr<ApspResult> RunApsp(
    const HostCsrGraph& graph,
    const ApspOptions& options,
    dawnHandle_t handle);
```

Options:

```cpp
struct ApspOptions {
  std::string output_path;
  std::string kernel_name;
  VertexId output_source = -1;
};
```

Input graph:

- CSR graph.
- Weighted or unweighted.
- Directed or undirected.

Output:

- `ApspResult` (`AlgorithmResult`).
- No all-pairs file is written by default.
- To save one source row, set both `output_path` and `output_source`.

Complexity class:

- `O(n(n + m))` for unweighted graphs.
- Up to `O(n^2m)` for weighted graphs in the current CPU backend.
- At most `O(n^3)` on sparse graphs.

#### CC

Purpose:

Closeness centrality for a single source.

Function:

```cpp
StatusOr<CcResult> RunCc(
    const HostCsrGraph& graph,
    const CcOptions& options,
    dawnHandle_t handle);
```

Options:

```cpp
struct CcOptions {
  VertexId source = 0;
  std::string kernel_name;
};
```

Input graph:

- CSR graph.
- Weighted or unweighted.
- Directed or undirected.

Output:

- `CcResult` (`AlgorithmResult`).
- `result.value` contains the closeness centrality value.
- The CLI prints the value as `value: ...`.

Complexity class:

- Same traversal class as BFS or SSSP from one source.

#### BC

Purpose:

Betweenness centrality over an unweighted graph.

Function:

```cpp
StatusOr<BcResult> RunBc(
    const HostCsrGraph& graph,
    const BcOptions& options,
    dawnHandle_t handle);
```

Options:

```cpp
struct BcOptions {
  std::string output_path;
  std::string kernel_name;
};
```

Input graph:

- CSR graph.
- Unweighted.
- Directed or undirected.

Output:

- `BcResult` (`AlgorithmResult`).
- Optional output file lines: `vertex centrality`.
- Zero centrality values are omitted.

Complexity class:

- `O(nm)` for unweighted Brandes-style centrality.
- At most `O(n^3)` on dense graphs.

## Generic Dispatcher

For CLI-style dispatch, use:

```cpp
StatusOr<AlgorithmResult> RunAlgorithm(
    const std::string& algorithm,
    const HostCsrGraph& graph,
    VertexId source,
    const std::vector<VertexId>& sources,
    const std::string& output_path,
    const std::string& kernel_name,
    VertexId output_source,
    dawnHandle_t handle);
```

Valid `algorithm` values:

- `bfs`
- `sssp`
- `mssp`
- `apsp`
- `cc`
- `bc`

## Minimal Library Example

```cpp
#include <dawn/algorithms/operators.hxx>
#include <dawn/io/matrix_market.hxx>
#include <dawn/runtime/handle.hxx>

#include <iostream>

int main() {
  DAWN::StatusOr<DAWN::HostCsrGraph> graph =
      DAWN::IO::ReadMatrixMarket("graph.mtx");
  if (!graph.ok()) {
    std::cerr << graph.status().ToString() << "\n";
    return 1;
  }

  DAWN::RuntimeOptions options;
  DAWN::StatusOr<DAWN::dawnHandle_t> handle = DAWN::dawnCreate(options);
  if (!handle.ok()) {
    std::cerr << handle.status().ToString() << "\n";
    return 1;
  }

  DAWN::Algorithms::BfsOptions bfs;
  bfs.source = 0;
  DAWN::StatusOr<DAWN::Algorithms::BfsResult> result =
      DAWN::Algorithms::RunBfs(graph.value(), bfs, handle.value());

  DAWN::dawnDestroy(handle.value());

  if (!result.ok()) {
    std::cerr << result.status().ToString() << "\n";
    return 1;
  }

  std::cout << result->elapsed_seconds << "\n";
  return 0;
}
```
