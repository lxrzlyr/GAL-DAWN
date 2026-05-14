#include <dawn/algorithms/operators.hxx>
#include <dawn/common/version.hxx>
#include <dawn/runtime/handle.hxx>

#include <cstdlib>

#include <iostream>
#include <utility>

int main() {
  std::cout << DAWN::VersionString() << "\n";

  DAWN::GraphMetadata metadata;
  metadata.num_vertices = 3;
  metadata.num_edges = 2;
  metadata.directed = true;
  metadata.weighted = false;

  DAWN::HostBuffer<DAWN::EdgeId> row_offsets(4, 0);
  row_offsets[0] = 0;
  row_offsets[1] = 1;
  row_offsets[2] = 2;
  row_offsets[3] = 2;
  DAWN::HostBuffer<DAWN::VertexId> column_indices(2, 0);
  column_indices[0] = 1;
  column_indices[1] = 2;

  DAWN::StatusOr<DAWN::HostCsrGraph> graph = DAWN::HostCsrGraph::Create(
      metadata,
      std::move(row_offsets),
      std::move(column_indices),
      DAWN::HostBuffer<DAWN::Weight>());
  if (!graph.ok()) {
    std::cerr << graph.status().ToString() << "\n";
    return EXIT_FAILURE;
  }

  DAWN::RuntimeOptions runtime_options;
  DAWN::StatusOr<DAWN::dawnHandle_t> handle =
      DAWN::dawnCreate(runtime_options);
  if (!handle.ok()) {
    std::cerr << handle.status().ToString() << "\n";
    return EXIT_FAILURE;
  }

  DAWN::Algorithms::BfsOptions options;
  options.source = 0;
  DAWN::StatusOr<DAWN::Algorithms::BfsResult> bfs =
      DAWN::Algorithms::RunBfs(graph.value(), options, handle.value());
  DAWN::dawnDestroy(handle.value());
  if (!bfs.ok()) {
    std::cerr << bfs.status().ToString() << "\n";
    return EXIT_FAILURE;
  }
  return bfs->kernel_name == "cpu_bfs" ? EXIT_SUCCESS : EXIT_FAILURE;
}
