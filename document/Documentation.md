# GAL-DAWN Library Reference

This document describes the public GAL-DAWN library surface.

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

Implementation headers live under `src/` and are not part of the installed
public API. Library users should prefer `include/dawn/algorithms/operators.hxx`.

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

### `DAWN::CpuThreadingPolicy`

```cpp
struct CpuThreadingPolicy {
  int thread_count = 1;
};
```

### `DAWN::DevicePolicy`

```cpp
struct DevicePolicy {
  BackendPolicy backend_policy = BackendPolicy::kAuto;
  int device_id = 0;
  bool explicit_device_id = false;
};
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
BackendPolicy dawnGetBackendPolicy(dawnHandle_t handle);
int dawnGetDeviceId(dawnHandle_t handle);
int dawnGetCpuThreadCount(dawnHandle_t handle);
```

## Kernel Registry

Defined in `dawn/runtime/kernel_registry.hxx`.

### `DAWN::GraphProfile`

```cpp
struct GraphProfile {
  VertexId vertex_count = 0;
  EdgeId edge_count = 0;
  IO::DegreeSummary degree_summary;
  bool weighted = false;
  int source_count = 1;
};
```

### `DAWN::KernelDescriptor`

```cpp
struct KernelDescriptor {
  std::string name;
  std::string algorithm;
  BackendKind backend = BackendKind::kCpu;
  bool supports_weighted = false;
  bool supports_unweighted = true;
};
```

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

## Algorithm Operators

Defined in `dawn/algorithms/operators.hxx`.

### `DAWN::Algorithms::AlgorithmResult`

```cpp
struct AlgorithmResult {
  double elapsed_seconds = 0.0;
  double value = 0.0;
  std::string kernel_name;
  BackendKind backend = BackendKind::kCpu;
};
```

### Options

```cpp
struct BfsOptions {
  VertexId source = 0;
  std::string output_path;
  std::string kernel_name;
};

struct SsspOptions {
  VertexId source = 0;
  std::string output_path;
  std::string kernel_name;
};

struct MsspOptions {
  std::vector<VertexId> sources;
  std::string output_path;
  std::string kernel_name;
  VertexId output_source = -1;
};

struct ApspOptions {
  std::string output_path;
  std::string kernel_name;
  VertexId output_source = -1;
};

struct CcOptions {
  VertexId source = 0;
  std::string kernel_name;
};

struct BcOptions {
  std::string output_path;
  std::string kernel_name;
};
```

### Operators

```cpp
StatusOr<BfsResult> RunBfs(const HostCsrGraph& graph,
                           const BfsOptions& options,
                           dawnHandle_t handle);
StatusOr<SsspResult> RunSssp(const HostCsrGraph& graph,
                             const SsspOptions& options,
                             dawnHandle_t handle);
StatusOr<MsspResult> RunMssp(const HostCsrGraph& graph,
                             const MsspOptions& options,
                             dawnHandle_t handle);
StatusOr<ApspResult> RunApsp(const HostCsrGraph& graph,
                             const ApspOptions& options,
                             dawnHandle_t handle);
StatusOr<CcResult> RunCc(const HostCsrGraph& graph,
                         const CcOptions& options,
                         dawnHandle_t handle);
StatusOr<BcResult> RunBc(const HostCsrGraph& graph,
                         const BcOptions& options,
                         dawnHandle_t handle);
StatusOr<AlgorithmResult> RunAlgorithm(const std::string& algorithm,
                                       const HostCsrGraph& graph,
                                       VertexId source,
                                       const std::vector<VertexId>& sources,
                                       const std::string& output_path,
                                       const std::string& kernel_name,
                                       VertexId output_source,
                                       dawnHandle_t handle);
```
