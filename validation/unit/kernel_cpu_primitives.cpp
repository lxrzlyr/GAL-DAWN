#include <dawn/graph/csr_graph.hxx>
#include <dawn/memory/buffer.hxx>
#include "cpu_kernels.hxx"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

int Fail(const std::string& message) {
  std::cerr << message << std::endl;
  return EXIT_FAILURE;
}

DAWN::HostBuffer<DAWN::EdgeId> EdgeBuffer(
    const std::vector<DAWN::EdgeId>& values) {
  DAWN::HostBuffer<DAWN::EdgeId> out(values.size(), 0);
  for (size_t i = 0; i < values.size(); ++i) {
    out[i] = values[i];
  }
  return out;
}

DAWN::HostBuffer<DAWN::VertexId> VertexBuffer(
    const std::vector<DAWN::VertexId>& values) {
  DAWN::HostBuffer<DAWN::VertexId> out(values.size(), 0);
  for (size_t i = 0; i < values.size(); ++i) {
    out[i] = values[i];
  }
  return out;
}

DAWN::HostBuffer<DAWN::Weight> WeightBuffer(
    const std::vector<DAWN::Weight>& values) {
  DAWN::HostBuffer<DAWN::Weight> out(values.size(), 0.0f);
  for (size_t i = 0; i < values.size(); ++i) {
    out[i] = values[i];
  }
  return out;
}

DAWN::StatusOr<DAWN::HostCsrGraph> MakeGraph(
    DAWN::VertexId vertices,
    bool directed,
    bool weighted,
    const std::vector<DAWN::EdgeId>& rows,
    const std::vector<DAWN::VertexId>& cols,
    const std::vector<DAWN::Weight>& weights = std::vector<DAWN::Weight>()) {
  DAWN::GraphMetadata metadata;
  metadata.num_vertices = vertices;
  metadata.num_edges = static_cast<DAWN::EdgeId>(cols.size());
  metadata.directed = directed;
  metadata.weighted = weighted;
  return DAWN::HostCsrGraph::Create(metadata, EdgeBuffer(rows),
                                    VertexBuffer(cols), WeightBuffer(weights));
}

bool Near(float a, float b, float tolerance = 1e-5f) {
  return std::fabs(a - b) <= tolerance;
}

int CheckBfs() {
  DAWN::StatusOr<DAWN::HostCsrGraph> graph = MakeGraph(
      5,
      true,
      false,
      std::vector<DAWN::EdgeId>{0, 2, 3, 4, 4, 4},
      std::vector<DAWN::VertexId>{1, 2, 3, 3});
  if (!graph.ok()) {
    return Fail("BFS graph creation failed");
  }

  DAWN::Kernel::CPU::BfsOptions options;
  options.source = 0;
  DAWN::StatusOr<DAWN::Kernel::CPU::BfsResult> bfs =
      DAWN::Kernel::CPU::RunBfs(graph->view(), options);
  if (!bfs.ok()) {
    return Fail("RunBfs failed: " + bfs.status().ToString());
  }
  const std::vector<int> expected{0, 1, 1, 2, 0};
  if (bfs->distances != expected) {
    return Fail("RunBfs distances mismatch");
  }
  return EXIT_SUCCESS;
}

int CheckSssp() {
  DAWN::StatusOr<DAWN::HostCsrGraph> graph = MakeGraph(
      4,
      true,
      true,
      std::vector<DAWN::EdgeId>{0, 2, 3, 4, 4},
      std::vector<DAWN::VertexId>{1, 2, 3, 3},
      std::vector<DAWN::Weight>{2.0f, 5.0f, 1.0f, 1.0f});
  if (!graph.ok()) {
    return Fail("SSSP graph creation failed");
  }

  DAWN::Kernel::CPU::SsspOptions options;
  options.source = 0;
  DAWN::StatusOr<DAWN::Kernel::CPU::SsspResult> sssp =
      DAWN::Kernel::CPU::RunSssp(graph->view(), options);
  if (!sssp.ok()) {
    return Fail("RunSssp failed: " + sssp.status().ToString());
  }
  const std::vector<float> expected{0.0f, 2.0f, 5.0f, 3.0f};
  for (size_t i = 0; i < expected.size(); ++i) {
    if (!Near(sssp->distances[i], expected[i])) {
      return Fail("RunSssp distances mismatch");
    }
  }
  return EXIT_SUCCESS;
}

int CheckSsspRejectsNegativeWeights() {
  DAWN::StatusOr<DAWN::HostCsrGraph> graph = MakeGraph(
      2,
      true,
      true,
      std::vector<DAWN::EdgeId>{0, 1, 1},
      std::vector<DAWN::VertexId>{1},
      std::vector<DAWN::Weight>{-1.0f});
  if (!graph.ok()) {
    return Fail("negative-weight graph creation failed");
  }

  DAWN::Kernel::CPU::SsspOptions options;
  options.source = 0;
  DAWN::StatusOr<DAWN::Kernel::CPU::SsspResult> sssp =
      DAWN::Kernel::CPU::RunSssp(graph->view(), options);
  if (sssp.ok()) {
    return Fail("RunSssp accepted negative edge weight");
  }
  return EXIT_SUCCESS;
}

int CheckBcPathAndSmallGraph() {
  DAWN::StatusOr<DAWN::HostCsrGraph> path = MakeGraph(
      3,
      false,
      false,
      std::vector<DAWN::EdgeId>{0, 1, 3, 4},
      std::vector<DAWN::VertexId>{1, 0, 2, 1});
  if (!path.ok()) {
    return Fail("BC path graph creation failed");
  }

  DAWN::Kernel::CPU::BcOptions options;
  DAWN::StatusOr<DAWN::Kernel::CPU::BcResult> bc =
      DAWN::Kernel::CPU::RunBc(path->view(), options);
  if (!bc.ok()) {
    return Fail("RunBc failed: " + bc.status().ToString());
  }
  const std::vector<float> expected{0.0f, 0.5f, 0.0f};
  for (size_t i = 0; i < expected.size(); ++i) {
    if (!Near(bc->centrality[i], expected[i])) {
      return Fail("RunBc path centrality mismatch");
    }
  }

  DAWN::StatusOr<DAWN::HostCsrGraph> two = MakeGraph(
      2,
      false,
      false,
      std::vector<DAWN::EdgeId>{0, 1, 2},
      std::vector<DAWN::VertexId>{1, 0});
  if (!two.ok()) {
    return Fail("BC small graph creation failed");
  }
  DAWN::StatusOr<DAWN::Kernel::CPU::BcResult> small =
      DAWN::Kernel::CPU::RunBc(two->view(), options);
  if (!small.ok()) {
    return Fail("RunBc small graph failed");
  }
  if (!Near(small->centrality[0], 0.0f) ||
      !Near(small->centrality[1], 0.0f)) {
    return Fail("RunBc n<=2 normalization mismatch");
  }
  return EXIT_SUCCESS;
}

}  // namespace

int main() {
  if (CheckBfs() != EXIT_SUCCESS) {
    return EXIT_FAILURE;
  }
  if (CheckSssp() != EXIT_SUCCESS) {
    return EXIT_FAILURE;
  }
  if (CheckSsspRejectsNegativeWeights() != EXIT_SUCCESS) {
    return EXIT_FAILURE;
  }
  if (CheckBcPathAndSmallGraph() != EXIT_SUCCESS) {
    return EXIT_FAILURE;
  }
  std::cout << "PASS: DAWN CPU primitive kernel tests" << std::endl;
  return EXIT_SUCCESS;
}
