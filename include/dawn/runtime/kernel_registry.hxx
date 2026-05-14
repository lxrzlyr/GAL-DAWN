#pragma once

#include <string>
#include <vector>

#include <dawn/common/status.hxx>
#include <dawn/graph/csr_graph.hxx>
#include <dawn/io/matrix_market.hxx>
#include <dawn/runtime/handle.hxx>

namespace DAWN {

struct GraphProfile {
  VertexId vertex_count = 0;
  EdgeId edge_count = 0;
  IO::DegreeSummary degree_summary;
  bool weighted = false;
  int source_count = 1;
};

struct KernelDescriptor {
  std::string name;
  std::string algorithm;
  BackendKind backend = BackendKind::kCpu;
  bool supports_weighted = false;
  bool supports_unweighted = true;
};

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

}  // namespace DAWN
