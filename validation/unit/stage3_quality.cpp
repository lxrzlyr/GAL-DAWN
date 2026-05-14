#include <dawn/io/matrix_market.hxx>
#include <dawn/runtime/device_selector.hxx>
#include <dawn/runtime/kernel_registry.hxx>

#include <cstdlib>
#include <iostream>
#include <string>

namespace {

int Fail(const std::string& message) {
  std::cerr << message << std::endl;
  return EXIT_FAILURE;
}

}  // namespace

int main() {
  DAWN::StatusOr<DAWN::HostCsrGraph> graph =
      DAWN::IO::ReadMatrixMarket("validation/fixtures/tiny_tail_isolated.mtx");
  if (!graph.ok()) {
    return Fail(graph.status().ToString());
  }
  if (graph->num_vertices() != 5 || graph->num_edges() != 4) {
    return Fail("graph metadata mismatch");
  }

  DAWN::GraphProfile profile = DAWN::ProfileGraph(graph->view(), 1);
  if (profile.degree_summary.max_degree != 2) {
    return Fail("graph profile mismatch");
  }

  std::vector<DAWN::KernelDescriptor> registry =
      DAWN::BuildDefaultKernelRegistry();
  DAWN::StatusOr<DAWN::KernelDescriptor> kernel =
      DAWN::SelectKernel(registry, "bfs", DAWN::BackendKind::kCpu, profile);
  if (!kernel.ok()) {
    return Fail(kernel.status().ToString());
  }

  DAWN::RuntimeOptions options;
  DAWN::StatusOr<DAWN::dawnHandle_t> handle = DAWN::dawnCreate(options);
  if (!handle.ok()) {
    return Fail(handle.status().ToString());
  }
  DAWN::StatusOr<DAWN::SelectedDevice> selected =
      DAWN::SelectDevice(handle.value(), graph.value());
  DAWN::dawnDestroy(handle.value());
  if (!selected.ok() || selected->backend != DAWN::BackendKind::kCpu) {
    return Fail("device selection failed");
  }

  std::cout << "PASS: DAWN stage3 quality test" << std::endl;
  return EXIT_SUCCESS;
}
