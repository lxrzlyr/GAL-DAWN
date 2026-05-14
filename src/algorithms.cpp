#include <dawn/algorithms/operators.hxx>
#include <dawn/result.hxx>
#include <dawn/runtime/device_selector.hxx>

#include "cpu_kernels.hxx"
#include "cpu_runner.hxx"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <limits>

namespace DAWN {
namespace Algorithms {
namespace Internal {

namespace {

Status WriteBfsOutput(const std::string& output_path,
                      VertexId source,
                      const std::vector<int>& distances) {
  std::ofstream file(output_path.c_str(), std::ios::app);
  if (!file.is_open()) {
    return Status(StatusCode::kUnavailable,
                  "failed to open output file: " + output_path);
  }
  for (size_t vertex = 0; vertex < distances.size(); ++vertex) {
    if (static_cast<VertexId>(vertex) != source && distances[vertex] > 0) {
      file << source << " " << vertex << " " << distances[vertex] << "\n";
    }
  }
  return Status::OK();
}

Status WriteSsspOutput(const std::string& output_path,
                       VertexId source,
                       const std::vector<Weight>& distances) {
  std::ofstream file(output_path.c_str(), std::ios::app);
  if (!file.is_open()) {
    return Status(StatusCode::kUnavailable,
                  "failed to open output file: " + output_path);
  }
  const Weight inf = std::numeric_limits<Weight>::max();
  for (size_t vertex = 0; vertex < distances.size(); ++vertex) {
    const Weight distance = distances[vertex];
    if (static_cast<VertexId>(vertex) != source && std::isfinite(distance) &&
        distance != 0.0f && distance < inf) {
      file << source << " " << vertex << " " << distance << "\n";
    }
  }
  return Status::OK();
}

Status WriteVectorOutput(const std::string& output_path,
                         const std::vector<float>& values) {
  std::ofstream file(output_path.c_str(), std::ios::app);
  if (!file.is_open()) {
    return Status(StatusCode::kUnavailable,
                  "failed to open output file: " + output_path);
  }
  for (size_t vertex = 0; vertex < values.size(); ++vertex) {
    const float value = values[vertex];
    if (std::isfinite(value) && value != 0.0f &&
        value < std::numeric_limits<float>::max()) {
      file << vertex << " " << value << "\n";
    }
  }
  return Status::OK();
}

float ClosenessFromBfsDistances(const std::vector<int>& distances) {
  float total_distance = 0.0f;
  for (int distance : distances) {
    if (distance > 0) {
      total_distance += static_cast<float>(distance);
    }
  }
  if (total_distance == 0.0f) {
    return 0.0f;
  }
  return (static_cast<float>(distances.size()) - 1.0f) / total_distance;
}

float ClosenessFromSsspDistances(const std::vector<Weight>& distances) {
  float total_distance = 0.0f;
  const Weight inf = std::numeric_limits<Weight>::max();
  for (Weight distance : distances) {
    if (std::isfinite(distance) && distance > 0.0f && distance < inf) {
      total_distance += distance;
    }
  }
  if (total_distance == 0.0f) {
    return 0.0f;
  }
  return (static_cast<float>(distances.size()) - 1.0f) / total_distance;
}

}  // namespace

StatusOr<RunResult> RunCpuBfs(const HostCsrGraph& graph,
                              VertexId source,
                              const std::string& output_path,
                              int thread_count) {
  Kernel::CPU::BfsOptions kernel_options;
  kernel_options.source = source;
  kernel_options.thread_count = thread_count;
  const auto start = std::chrono::high_resolution_clock::now();
  StatusOr<Kernel::CPU::BfsResult> bfs =
      Kernel::CPU::RunBfs(graph.view(), kernel_options);
  const auto end = std::chrono::high_resolution_clock::now();
  if (!bfs.ok()) {
    return bfs.status();
  }
  if (!output_path.empty()) {
    Status output_status = WriteBfsOutput(output_path, source, bfs->distances);
    if (!output_status.ok()) {
      return output_status;
    }
  }
  std::chrono::duration<double> elapsed = end - start;
  return RunResult::ok(static_cast<float>(elapsed.count()));
}

StatusOr<RunResult> RunCpuSssp(const HostCsrGraph& graph,
                               VertexId source,
                               const std::string& output_path,
                               int thread_count) {
  Kernel::CPU::SsspOptions kernel_options;
  kernel_options.source = source;
  kernel_options.thread_count = thread_count;
  const auto start = std::chrono::high_resolution_clock::now();
  StatusOr<Kernel::CPU::SsspResult> sssp =
      Kernel::CPU::RunSssp(graph.view(), kernel_options);
  const auto end = std::chrono::high_resolution_clock::now();
  if (!sssp.ok()) {
    return sssp.status();
  }
  if (!output_path.empty()) {
    Status output_status =
        WriteSsspOutput(output_path, source, sssp->distances);
    if (!output_status.ok()) {
      return output_status;
    }
  }
  std::chrono::duration<double> elapsed = end - start;
  return RunResult::ok(static_cast<float>(elapsed.count()));
}

StatusOr<RunResult> RunCpuShortestPaths(const HostCsrGraph& graph,
                                        VertexId source,
                                        const std::string& output_path,
                                        int thread_count) {
  if (graph.weighted()) {
    return RunCpuSssp(graph, source, output_path, thread_count);
  }
  return RunCpuBfs(graph, source, output_path, thread_count);
}

StatusOr<RunResult> RunCpuMssp(const HostCsrGraph& graph,
                               const std::vector<VertexId>& sources,
                               const std::string& output_path,
                               int thread_count) {
  const auto start = std::chrono::high_resolution_clock::now();
  for (VertexId source : sources) {
    StatusOr<RunResult> run =
        RunCpuShortestPaths(graph, source, output_path, thread_count);
    if (!run.ok()) {
      return run.status();
    }
  }
  const auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  return RunResult::ok(static_cast<float>(elapsed.count()));
}

StatusOr<RunResult> RunCpuApsp(const HostCsrGraph& graph,
                               const std::string& output_path,
                               int thread_count) {
  const auto start = std::chrono::high_resolution_clock::now();
  for (VertexId source = 0; source < graph.num_vertices(); ++source) {
    StatusOr<RunResult> run =
        RunCpuShortestPaths(graph, source, output_path, thread_count);
    if (!run.ok()) {
      return run.status();
    }
  }
  const auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  return RunResult::ok(static_cast<float>(elapsed.count()));
}

StatusOr<RunResult> RunCpuCc(const HostCsrGraph& graph,
                             VertexId source,
                             int thread_count) {
  const auto start = std::chrono::high_resolution_clock::now();
  float value = 0.0f;
  if (graph.weighted()) {
    Kernel::CPU::SsspOptions options;
    options.source = source;
    options.thread_count = thread_count;
    StatusOr<Kernel::CPU::SsspResult> sssp =
        Kernel::CPU::RunSssp(graph.view(), options);
    if (!sssp.ok()) {
      return sssp.status();
    }
    value = ClosenessFromSsspDistances(sssp->distances);
  } else {
    Kernel::CPU::BfsOptions options;
    options.source = source;
    options.thread_count = thread_count;
    StatusOr<Kernel::CPU::BfsResult> bfs =
        Kernel::CPU::RunBfs(graph.view(), options);
    if (!bfs.ok()) {
      return bfs.status();
    }
    value = ClosenessFromBfsDistances(bfs->distances);
  }
  const auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  return RunResult::ok(static_cast<float>(elapsed.count()), value);
}

StatusOr<RunResult> RunCpuBc(const HostCsrGraph& graph,
                             const std::string& output_path,
                             int thread_count) {
  Kernel::CPU::BcOptions kernel_options;
  kernel_options.thread_count = thread_count;
  const auto start = std::chrono::high_resolution_clock::now();
  StatusOr<Kernel::CPU::BcResult> bc =
      Kernel::CPU::RunBc(graph.view(), kernel_options);
  const auto end = std::chrono::high_resolution_clock::now();
  if (!bc.ok()) {
    return bc.status();
  }
  if (!output_path.empty()) {
    Status output_status = WriteVectorOutput(output_path, bc->centrality);
    if (!output_status.ok()) {
      return output_status;
    }
  }
  std::chrono::duration<double> elapsed = end - start;
  return RunResult::ok(static_cast<float>(elapsed.count()));
}

}  // namespace Internal
}  // namespace Algorithms
}  // namespace DAWN

namespace DAWN {
namespace Algorithms {

namespace {

Status CheckSource(const HostCsrGraph& graph, VertexId source) {
  if (source < 0 || source >= graph.num_vertices()) {
    return Status(StatusCode::kOutOfRange, "source is outside graph");
  }
  return Status::OK();
}

std::vector<VertexId> SingleOutputSource(VertexId output_source) {
  if (output_source < 0) {
    return std::vector<VertexId>();
  }
  return std::vector<VertexId>(1, output_source);
}

StatusOr<KernelDescriptor> ResolveKernel(const std::string& algorithm,
                                         const HostCsrGraph& graph,
                                         int source_count,
                                         const std::string& kernel_name,
                                         dawnHandle_t handle,
                                         SelectedDevice* selected) {
  StatusOr<SelectedDevice> selected_or = SelectDevice(handle, graph);
  if (!selected_or.ok()) {
    return selected_or.status();
  }
  *selected = selected_or.value();
  if (selected->backend == BackendKind::kCuda) {
    return Status(StatusCode::kUnsupported,
                  "CUDA algorithm dispatch is not wired in Stage 2 CPU build");
  }

  std::vector<KernelDescriptor> registry = BuildDefaultKernelRegistry();
  GraphProfile profile = ProfileGraph(graph.view(), source_count);
  return SelectKernel(registry, algorithm, selected->backend, profile,
                      kernel_name);
}

AlgorithmResult FromRunResult(const RunResult& run,
                              const KernelDescriptor& kernel,
                              BackendKind backend) {
  AlgorithmResult result;
  result.elapsed_seconds = run.elapsed_time;
  result.value = run.value;
  result.kernel_name = kernel.name;
  result.backend = backend;
  return result;
}

}  // namespace

StatusOr<BfsResult> RunBfs(const HostCsrGraph& graph,
                           const BfsOptions& options,
                           dawnHandle_t handle) {
  Status source_status = CheckSource(graph, options.source);
  if (!source_status.ok()) {
    return source_status;
  }
  SelectedDevice selected;
  StatusOr<KernelDescriptor> kernel =
      ResolveKernel("bfs", graph, 1, options.kernel_name, handle, &selected);
  if (!kernel.ok()) {
    return kernel.status();
  }

  StatusOr<RunResult> run = Internal::RunCpuBfs(
      graph, options.source, options.output_path, dawnGetCpuThreadCount(handle));
  if (!run.ok()) {
    return run.status();
  }
  return FromRunResult(run.value(), kernel.value(), selected.backend);
}

StatusOr<SsspResult> RunSssp(const HostCsrGraph& graph,
                             const SsspOptions& options,
                             dawnHandle_t handle) {
  if (!graph.weighted()) {
    return Status(StatusCode::kInvalidArgument,
                  "SSSP requires a weighted graph");
  }
  Status source_status = CheckSource(graph, options.source);
  if (!source_status.ok()) {
    return source_status;
  }
  SelectedDevice selected;
  StatusOr<KernelDescriptor> kernel =
      ResolveKernel("sssp", graph, 1, options.kernel_name, handle, &selected);
  if (!kernel.ok()) {
    return kernel.status();
  }

  StatusOr<RunResult> run =
      Internal::RunCpuSssp(graph, options.source, options.output_path,
                           dawnGetCpuThreadCount(handle));
  if (!run.ok()) {
    return run.status();
  }
  return FromRunResult(run.value(), kernel.value(), selected.backend);
}

StatusOr<MsspResult> RunMssp(const HostCsrGraph& graph,
                             const MsspOptions& options,
                             dawnHandle_t handle) {
  if (options.sources.empty()) {
    return Status(StatusCode::kInvalidArgument, "MSSP requires sources");
  }
  for (VertexId source : options.sources) {
    Status source_status = CheckSource(graph, source);
    if (!source_status.ok()) {
      return source_status;
    }
  }
  if (options.output_source >= graph.num_vertices()) {
    return Status(StatusCode::kOutOfRange,
                  "MSSP output source is outside graph");
  }
  if (options.output_source >= 0) {
    Status source_status = CheckSource(graph, options.output_source);
    if (!source_status.ok()) {
      return source_status;
    }
    if (std::find(options.sources.begin(), options.sources.end(),
                  options.output_source) == options.sources.end()) {
      return Status(StatusCode::kInvalidArgument,
                    "MSSP output source must be present in source list");
    }
  }
  SelectedDevice selected;
  StatusOr<KernelDescriptor> kernel =
      ResolveKernel("mssp", graph, static_cast<int>(options.sources.size()),
                    options.kernel_name, handle, &selected);
  if (!kernel.ok()) {
    return kernel.status();
  }

  const bool save_output =
      !options.output_path.empty() && options.output_source >= 0;
  std::vector<VertexId> execution_sources =
      save_output ? SingleOutputSource(options.output_source) : options.sources;
  const std::string output = save_output ? options.output_path : "";
  StatusOr<RunResult> run = Internal::RunCpuMssp(
      graph, execution_sources, output, dawnGetCpuThreadCount(handle));
  if (!run.ok()) {
    return run.status();
  }
  return FromRunResult(run.value(), kernel.value(), selected.backend);
}

StatusOr<ApspResult> RunApsp(const HostCsrGraph& graph,
                             const ApspOptions& options,
                             dawnHandle_t handle) {
  if (options.output_source >= graph.num_vertices()) {
    return Status(StatusCode::kOutOfRange,
                  "APSP output source is outside graph");
  }
  SelectedDevice selected;
  StatusOr<KernelDescriptor> kernel =
      ResolveKernel("apsp", graph, graph.num_vertices(), options.kernel_name,
                    handle, &selected);
  if (!kernel.ok()) {
    return kernel.status();
  }

  StatusOr<RunResult> run = Status(StatusCode::kInternal, "APSP was not run");
  if (!options.output_path.empty() && options.output_source >= 0) {
    run = Internal::RunCpuShortestPaths(graph, options.output_source,
                                        options.output_path,
                                        dawnGetCpuThreadCount(handle));
  } else {
    run = Internal::RunCpuApsp(graph, "", dawnGetCpuThreadCount(handle));
  }
  if (!run.ok()) {
    return run.status();
  }
  return FromRunResult(run.value(), kernel.value(), selected.backend);
}

StatusOr<CcResult> RunCc(const HostCsrGraph& graph,
                         const CcOptions& options,
                         dawnHandle_t handle) {
  Status source_status = CheckSource(graph, options.source);
  if (!source_status.ok()) {
    return source_status;
  }
  SelectedDevice selected;
  StatusOr<KernelDescriptor> kernel =
      ResolveKernel("cc", graph, 1, options.kernel_name, handle, &selected);
  if (!kernel.ok()) {
    return kernel.status();
  }

  StatusOr<RunResult> run =
      Internal::RunCpuCc(graph, options.source, dawnGetCpuThreadCount(handle));
  if (!run.ok()) {
    return run.status();
  }
  return FromRunResult(run.value(), kernel.value(), selected.backend);
}

StatusOr<BcResult> RunBc(const HostCsrGraph& graph,
                         const BcOptions& options,
                         dawnHandle_t handle) {
  if (graph.weighted()) {
    return Status(StatusCode::kInvalidArgument,
                  "BC CPU kernel requires an unweighted graph");
  }
  SelectedDevice selected;
  StatusOr<KernelDescriptor> kernel =
      ResolveKernel("bc", graph, graph.num_vertices(), options.kernel_name,
                    handle, &selected);
  if (!kernel.ok()) {
    return kernel.status();
  }

  StatusOr<RunResult> run =
      Internal::RunCpuBc(graph, options.output_path,
                         dawnGetCpuThreadCount(handle));
  if (!run.ok()) {
    return run.status();
  }
  return FromRunResult(run.value(), kernel.value(), selected.backend);
}

StatusOr<AlgorithmResult> RunAlgorithm(const std::string& algorithm,
                                       const HostCsrGraph& graph,
                                       VertexId source,
                                       const std::vector<VertexId>& sources,
                                       const std::string& output_path,
                                       const std::string& kernel_name,
                                       VertexId output_source,
                                       dawnHandle_t handle) {
  if (algorithm == "bfs") {
    return RunBfs(graph, BfsOptions{source, output_path, kernel_name}, handle);
  }
  if (algorithm == "sssp") {
    return RunSssp(graph, SsspOptions{source, output_path, kernel_name},
                   handle);
  }
  if (algorithm == "mssp") {
    MsspOptions options;
    options.sources = sources;
    options.output_path = output_path;
    options.kernel_name = kernel_name;
    options.output_source = output_source;
    return RunMssp(graph, options, handle);
  }
  if (algorithm == "apsp") {
    ApspOptions options;
    options.output_path = output_path;
    options.kernel_name = kernel_name;
    options.output_source = output_source;
    return RunApsp(graph, options, handle);
  }
  if (algorithm == "cc") {
    return RunCc(graph, CcOptions{source, kernel_name}, handle);
  }
  if (algorithm == "bc") {
    return RunBc(graph, BcOptions{output_path, kernel_name}, handle);
  }
  return Status(StatusCode::kInvalidArgument,
                "unsupported algorithm: " + algorithm);
}

}  // namespace Algorithms
}  // namespace DAWN
