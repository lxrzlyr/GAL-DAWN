#include <dawn/runtime/kernel_registry.hxx>

namespace DAWN {

GraphProfile ProfileGraph(CsrView view, int source_count) {
  GraphProfile profile;
  profile.vertex_count = view.num_vertices;
  profile.edge_count = view.num_edges;
  profile.degree_summary = IO::ComputeDegreeSummary(view);
  profile.weighted = view.weighted;
  profile.source_count = source_count;
  return profile;
}

std::vector<KernelDescriptor> BuildDefaultKernelRegistry() {
  std::vector<KernelDescriptor> registry;
  registry.push_back({"cpu_bfs", "bfs", BackendKind::kCpu, false, true});
  registry.push_back({"cpu_sssp", "sssp", BackendKind::kCpu, true, false});
  registry.push_back({"cpu_mssp", "mssp", BackendKind::kCpu, true, true});
  registry.push_back({"cpu_apsp", "apsp", BackendKind::kCpu, true, true});
  registry.push_back({"cpu_cc", "cc", BackendKind::kCpu, true, true});
  registry.push_back({"cpu_bc", "bc", BackendKind::kCpu, false, true});
#if DAWN_HAS_CUDA
  registry.push_back({"cuda_bfs", "bfs", BackendKind::kCuda, false, true});
  registry.push_back({"cuda_sssp", "sssp", BackendKind::kCuda, true, false});
#endif
  return registry;
}

std::vector<KernelDescriptor> ListKernelsForAlgorithm(
    const std::vector<KernelDescriptor>& registry,
    const std::string& algorithm) {
  std::vector<KernelDescriptor> out;
  for (const KernelDescriptor& descriptor : registry) {
    if (algorithm.empty() || descriptor.algorithm == algorithm) {
      out.push_back(descriptor);
    }
  }
  return out;
}

StatusOr<KernelDescriptor> SelectKernel(
    const std::vector<KernelDescriptor>& registry,
    const std::string& algorithm,
    BackendKind backend,
    const GraphProfile& profile,
    const std::string& explicit_kernel) {
  for (const KernelDescriptor& descriptor : registry) {
    if (descriptor.algorithm != algorithm || descriptor.backend != backend) {
      continue;
    }
    if (!explicit_kernel.empty() && descriptor.name != explicit_kernel) {
      continue;
    }
    if (profile.weighted && !descriptor.supports_weighted) {
      continue;
    }
    if (!profile.weighted && !descriptor.supports_unweighted) {
      continue;
    }
    return descriptor;
  }

  if (!explicit_kernel.empty()) {
    return Status(
        StatusCode::kNotFound,
        "kernel override not found or incompatible: " + explicit_kernel);
  }
  return Status(StatusCode::kNotFound,
                "no compatible kernel registered for algorithm " + algorithm);
}

}  // namespace DAWN
