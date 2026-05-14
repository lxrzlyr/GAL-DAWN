#include <dawn/algorithms/operators.hxx>
#include <dawn/io/matrix_market.hxx>
#include <dawn/runtime/device_selector.hxx>
#include <dawn/runtime/kernel_registry.hxx>

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

int Fail(const std::string& message) {
  std::cerr << message << std::endl;
  return EXIT_FAILURE;
}

int CheckRuntimeAndDeviceSelector() {
  DAWN::RuntimeOptions options;
  options.device_policy.backend_policy = DAWN::BackendPolicy::kAuto;
  DAWN::StatusOr<DAWN::dawnHandle_t> handle = DAWN::dawnCreate(options);
  if (!handle.ok()) {
    return Fail("dawnCreate failed");
  }

  DAWN::StatusOr<DAWN::HostCsrGraph> graph =
      DAWN::IO::ReadMatrixMarket("validation/fixtures/tiny_unweighted.mtx");
  if (!graph.ok()) {
    DAWN::dawnDestroy(handle.value());
    return Fail("graph load failed");
  }

  DAWN::StatusOr<DAWN::SelectedDevice> selected =
      DAWN::SelectDevice(handle.value(), graph.value());
  if (!selected.ok() || selected->backend != DAWN::BackendKind::kCpu) {
    DAWN::dawnDestroy(handle.value());
    return Fail("auto backend did not select CPU in CPU-only test");
  }

  DAWN::dawnSetBackendPolicy(handle.value(), DAWN::BackendPolicy::kCuda);
  DAWN::StatusOr<DAWN::SelectedDevice> cuda =
      DAWN::SelectDevice(handle.value(), graph.value());
  if (cuda.ok()) {
    DAWN::dawnDestroy(handle.value());
    return Fail("CUDA backend unexpectedly selected in CPU-only build");
  }

  DAWN::dawnDestroy(handle.value());
  return EXIT_SUCCESS;
}

int CheckKernelRegistry() {
  std::vector<DAWN::KernelDescriptor> registry =
      DAWN::BuildDefaultKernelRegistry();
  std::vector<DAWN::KernelDescriptor> bfs =
      DAWN::ListKernelsForAlgorithm(registry, "bfs");
  if (bfs.empty()) {
    return Fail("BFS kernel was not registered");
  }

  DAWN::StatusOr<DAWN::HostCsrGraph> graph =
      DAWN::IO::ReadMatrixMarket("validation/fixtures/tiny_unweighted.mtx");
  if (!graph.ok()) {
    return Fail("graph load failed");
  }
  DAWN::GraphProfile profile = DAWN::ProfileGraph(graph->view(), 1);
  DAWN::StatusOr<DAWN::KernelDescriptor> kernel =
      DAWN::SelectKernel(registry, "bfs", DAWN::BackendKind::kCpu, profile);
  if (!kernel.ok() || kernel->name != "cpu_bfs") {
    return Fail("BFS kernel selection failed");
  }
  DAWN::StatusOr<DAWN::KernelDescriptor> missing = DAWN::SelectKernel(
      registry, "bfs", DAWN::BackendKind::kCpu, profile, "missing_kernel");
  if (missing.ok()) {
    return Fail("missing kernel override was accepted");
  }
  return EXIT_SUCCESS;
}

int CheckOperators() {
  DAWN::RuntimeOptions options;
  options.device_policy.backend_policy = DAWN::BackendPolicy::kAuto;
  DAWN::StatusOr<DAWN::dawnHandle_t> handle = DAWN::dawnCreate(options);
  if (!handle.ok()) {
    return Fail("dawnCreate failed");
  }

  DAWN::StatusOr<DAWN::HostCsrGraph> unweighted =
      DAWN::IO::ReadMatrixMarket("validation/fixtures/tiny_unweighted.mtx");
  DAWN::StatusOr<DAWN::HostCsrGraph> weighted =
      DAWN::IO::ReadMatrixMarket("validation/fixtures/tiny_weighted.mtx");
  if (!unweighted.ok() || !weighted.ok()) {
    DAWN::dawnDestroy(handle.value());
    return Fail("operator fixtures failed to load");
  }

  DAWN::StatusOr<DAWN::Algorithms::BfsResult> bfs = DAWN::Algorithms::RunBfs(
      unweighted.value(), DAWN::Algorithms::BfsOptions(), handle.value());
  if (!bfs.ok() || bfs->kernel_name != "cpu_bfs") {
    DAWN::dawnDestroy(handle.value());
    return Fail("RunBfs failed");
  }

  DAWN::Algorithms::SsspOptions sssp_options;
  sssp_options.source = 0;
  DAWN::StatusOr<DAWN::Algorithms::SsspResult> sssp =
      DAWN::Algorithms::RunSssp(weighted.value(), sssp_options, handle.value());
  if (!sssp.ok() || sssp->kernel_name != "cpu_sssp") {
    DAWN::dawnDestroy(handle.value());
    return Fail("RunSssp failed");
  }

  DAWN::Algorithms::MsspOptions mssp_options;
  mssp_options.sources.push_back(0);
  mssp_options.sources.push_back(1);
  DAWN::StatusOr<DAWN::Algorithms::MsspResult> mssp = DAWN::Algorithms::RunMssp(
      unweighted.value(), mssp_options, handle.value());
  if (!mssp.ok()) {
    DAWN::dawnDestroy(handle.value());
    return Fail("RunMssp failed: " + mssp.status().ToString());
  }

  DAWN::Algorithms::CcOptions cc_options;
  cc_options.source = 0;
  DAWN::StatusOr<DAWN::Algorithms::CcResult> cc =
      DAWN::Algorithms::RunCc(unweighted.value(), cc_options, handle.value());
  if (!cc.ok()) {
    DAWN::dawnDestroy(handle.value());
    return Fail("RunCc failed");
  }

  DAWN::StatusOr<DAWN::Algorithms::BfsResult> bad_kernel =
      DAWN::Algorithms::RunBfs(
          unweighted.value(),
          DAWN::Algorithms::BfsOptions{0, "", "missing_kernel"},
          handle.value());
  if (bad_kernel.ok()) {
    DAWN::dawnDestroy(handle.value());
    return Fail("RunBfs accepted missing kernel override");
  }

  DAWN::dawnDestroy(handle.value());
  return EXIT_SUCCESS;
}

}  // namespace

int main() {
  if (CheckRuntimeAndDeviceSelector() != EXIT_SUCCESS) {
    return EXIT_FAILURE;
  }
  if (CheckKernelRegistry() != EXIT_SUCCESS) {
    return EXIT_FAILURE;
  }
  if (CheckOperators() != EXIT_SUCCESS) {
    return EXIT_FAILURE;
  }
  std::cout << "PASS: DAWN stage2 unit tests" << std::endl;
  return EXIT_SUCCESS;
}
