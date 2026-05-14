#include <dawn/graph/csr_graph.hxx>

namespace DAWN {

HostCsrGraph::HostCsrGraph(GraphMetadata metadata,
                           HostBuffer<EdgeId> row_offsets,
                           HostBuffer<VertexId> column_indices,
                           HostBuffer<Weight> weights)
    : metadata_(metadata),
      row_offsets_(std::move(row_offsets)),
      column_indices_(std::move(column_indices)),
      weights_(std::move(weights)) {}

StatusOr<HostCsrGraph> HostCsrGraph::Create(GraphMetadata metadata,
                                            HostBuffer<EdgeId> row_offsets,
                                            HostBuffer<VertexId> column_indices,
                                            HostBuffer<Weight> weights) {
  HostCsrGraph graph(metadata, std::move(row_offsets),
                     std::move(column_indices), std::move(weights));
  Status status = graph.Validate();
  if (!status.ok()) {
    return status;
  }
  return graph;
}

Status HostCsrGraph::Validate() const {
  if (metadata_.num_vertices < 0) {
    return Status(StatusCode::kInvalidArgument,
                  "vertex count must be non-negative");
  }
  if (metadata_.num_edges < 0) {
    return Status(StatusCode::kInvalidArgument,
                  "edge count must be non-negative");
  }

  const size_t expected_rows = static_cast<size_t>(metadata_.num_vertices) + 1;
  if (row_offsets_.size() != expected_rows) {
    return Status(StatusCode::kInvalidArgument,
                  "CSR row_offsets size must equal num_vertices + 1");
  }
  if (row_offsets_.empty() || row_offsets_[0] != 0) {
    return Status(StatusCode::kInvalidArgument,
                  "CSR row_offsets must start at zero");
  }
  if (row_offsets_[metadata_.num_vertices] != metadata_.num_edges) {
    return Status(StatusCode::kInvalidArgument,
                  "CSR terminal row offset must equal num_edges");
  }
  if (column_indices_.size() != static_cast<size_t>(metadata_.num_edges)) {
    return Status(StatusCode::kInvalidArgument,
                  "CSR column_indices size must equal num_edges");
  }
  if (metadata_.weighted &&
      weights_.size() != static_cast<size_t>(metadata_.num_edges)) {
    return Status(StatusCode::kInvalidArgument,
                  "weighted CSR weights size must equal num_edges");
  }
  if (!metadata_.weighted && !weights_.empty()) {
    return Status(StatusCode::kInvalidArgument,
                  "unweighted CSR must not carry weights");
  }

  EdgeId previous = 0;
  for (VertexId row = 0; row <= metadata_.num_vertices; ++row) {
    const EdgeId current = row_offsets_[row];
    if (current < previous || current < 0 || current > metadata_.num_edges) {
      return Status(StatusCode::kInvalidArgument,
                    "CSR row_offsets must be monotonic and in range");
    }
    previous = current;
  }
  for (EdgeId edge = 0; edge < metadata_.num_edges; ++edge) {
    const VertexId col = column_indices_[edge];
    if (col < 0 || col >= metadata_.num_vertices) {
      return Status(StatusCode::kOutOfRange,
                    "CSR column index is outside vertex range");
    }
  }
  return Status::OK();
}

}  // namespace DAWN
