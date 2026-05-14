#include <dawn_internal/adapters/kernel_csr.hxx>

#include <limits>

namespace DAWN {
namespace Adapters {

StatusOr<KernelCsrGraph> MakeKernelCsrGraph(const CsrView& view) {
  StatusOr<int> rows = checked_int_downcast(view.num_vertices, "num_vertices");
  if (!rows.ok()) {
    return rows.status();
  }
  StatusOr<int> nnz = checked_int_downcast(view.num_edges, "num_edges");
  if (!nnz.ok()) {
    return nnz.status();
  }
  for (VertexId row = 0; row <= view.num_vertices; ++row) {
    StatusOr<int> offset =
        checked_int_downcast(view.row_offsets[row], "row_offsets");
    if (!offset.ok()) {
      return offset.status();
    }
  }
  for (EdgeId edge = 0; edge < view.num_edges; ++edge) {
    StatusOr<int> col =
        checked_int_downcast(view.column_indices[edge], "column_indices");
    if (!col.ok()) {
      return col.status();
    }
  }

  KernelCsrGraph kernel_graph;
  kernel_graph.rows = rows.value();
  kernel_graph.nnz = nnz.value();
  kernel_graph.row_ptr =
      HostBuffer<int>(static_cast<size_t>(kernel_graph.rows) + 1);
  kernel_graph.col = HostBuffer<int>(static_cast<size_t>(kernel_graph.nnz));
  if (view.weighted) {
    kernel_graph.val = HostBuffer<float>(static_cast<size_t>(kernel_graph.nnz));
  }

  for (int row = 0; row <= kernel_graph.rows; ++row) {
    kernel_graph.row_ptr[row] = static_cast<int>(view.row_offsets[row]);
  }
  for (int edge = 0; edge < kernel_graph.nnz; ++edge) {
    kernel_graph.col[edge] = static_cast<int>(view.column_indices[edge]);
    if (view.weighted) {
      kernel_graph.val[edge] = view.weights[edge];
    }
  }

  return kernel_graph;
}

}  // namespace Adapters
}  // namespace DAWN
