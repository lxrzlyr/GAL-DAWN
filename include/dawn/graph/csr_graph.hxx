#pragma once

#include <type_traits>

#include <dawn/common/status.hxx>
#include <dawn/common/types.hxx>
#include <dawn/memory/buffer.hxx>

namespace DAWN {

enum class GraphStorageFormat { kCsr };

struct GraphMetadata {
  VertexId num_vertices = 0;
  EdgeId num_edges = 0;
  bool directed = true;
  bool weighted = false;
  GraphStorageFormat storage_format = GraphStorageFormat::kCsr;
};

struct CsrView {
  const EdgeId* row_offsets = nullptr;
  const VertexId* column_indices = nullptr;
  const Weight* weights = nullptr;
  VertexId num_vertices = 0;
  EdgeId num_edges = 0;
  bool directed = true;
  bool weighted = false;
};

static_assert(std::is_trivially_copyable<CsrView>::value,
              "CsrView must be cheap to pass by value");

class HostCsrGraph {
 public:
  HostCsrGraph() = default;

  HostCsrGraph(GraphMetadata metadata,
               HostBuffer<EdgeId> row_offsets,
               HostBuffer<VertexId> column_indices,
               HostBuffer<Weight> weights);

  HostCsrGraph(HostCsrGraph&&) noexcept = default;
  HostCsrGraph& operator=(HostCsrGraph&&) noexcept = default;

  HostCsrGraph(const HostCsrGraph&) = delete;
  HostCsrGraph& operator=(const HostCsrGraph&) = delete;

  static StatusOr<HostCsrGraph> Create(GraphMetadata metadata,
                                       HostBuffer<EdgeId> row_offsets,
                                       HostBuffer<VertexId> column_indices,
                                       HostBuffer<Weight> weights);

  const GraphMetadata& metadata() const { return metadata_; }
  VertexId num_vertices() const { return metadata_.num_vertices; }
  EdgeId num_edges() const { return metadata_.num_edges; }
  bool directed() const { return metadata_.directed; }
  bool weighted() const { return metadata_.weighted; }

  const HostBuffer<EdgeId>& row_offsets() const { return row_offsets_; }
  const HostBuffer<VertexId>& column_indices() const { return column_indices_; }
  const HostBuffer<Weight>& weights() const { return weights_; }

  CsrView view() const {
    return {row_offsets_.data(),
            column_indices_.data(),
            weights_.empty() ? nullptr : weights_.data(),
            metadata_.num_vertices,
            metadata_.num_edges,
            metadata_.directed,
            metadata_.weighted};
  }

  Status Validate() const;

 private:
  GraphMetadata metadata_;
  HostBuffer<EdgeId> row_offsets_;
  HostBuffer<VertexId> column_indices_;
  HostBuffer<Weight> weights_;
};

struct DeviceCsrView {
  const EdgeId* row_offsets = nullptr;
  const VertexId* column_indices = nullptr;
  const Weight* weights = nullptr;
  VertexId num_vertices = 0;
  EdgeId num_edges = 0;
  bool directed = true;
  bool weighted = false;
};

class DeviceCsrGraph {
 public:
  DeviceCsrGraph() = default;

 private:
  GraphMetadata metadata_;
#if DAWN_HAS_CUDA
  Buffer<EdgeId, MemorySpace::kDevice> row_offsets_;
  Buffer<VertexId, MemorySpace::kDevice> column_indices_;
  Buffer<Weight, MemorySpace::kDevice> weights_;
#endif
};

}  // namespace DAWN
