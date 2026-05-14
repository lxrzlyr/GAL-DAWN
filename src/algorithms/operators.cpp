#include <dawn/algorithms/operators.hxx>

#include <dawn_internal/adapters/kernel_csr.hxx>
#include <dawn_internal/algorithm/cpu/apsp.hxx>
#include <dawn_internal/algorithm/cpu/bc.hxx>
#include <dawn_internal/algorithm/cpu/bfs.hxx>
#include <dawn_internal/algorithm/cpu/cc.hxx>
#include <dawn_internal/algorithm/cpu/mssp.hxx>
#include <dawn_internal/algorithm/cpu/sssp.hxx>
#include <dawn/runtime/device_selector.hxx>

#include <algorithm>

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

StatusOr<Adapters::KernelCsrGraph> MakeKernelCsr(const HostCsrGraph& graph) {
  return Adapters::MakeKernelCsrGraph(graph.view());
}

Graph::Graph_t MakeKernelGraph(const HostCsrGraph& graph,
                               Adapters::KernelCsrGraph& kernel_graph,
                               bool print,
                               VertexId source,
                               const std::vector<VertexId>& sources,
                               int thread_count) {
  Graph::Graph_t out;
  out.rows = kernel_graph.rows;
  out.cols = kernel_graph.rows;
  out.nnz = kernel_graph.nnz;
  out.csr.row_ptr = kernel_graph.row_ptr.data();
  out.csr.col = kernel_graph.col.data();
  out.csr.val = kernel_graph.val.empty() ? nullptr : kernel_graph.val.data();
  out.coo.row = nullptr;
  out.coo.col = nullptr;
  out.coo.val = nullptr;
  out.thread = thread_count > 0 ? thread_count : 1;
  out.interval = 100;
  out.stream = 0;
  out.block_size = 0;
  out.source = source;
  out.print = print;
  out.weighted = graph.weighted();
  out.directed = graph.directed();
  out.msource.assign(sources.begin(), sources.end());
  return out;
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

  StatusOr<Adapters::KernelCsrGraph> kernel_graph = MakeKernelCsr(graph);
  if (!kernel_graph.ok()) {
    return kernel_graph.status();
  }
  Graph::Graph_t execution_graph = MakeKernelGraph(
      graph, kernel_graph.value(), !options.output_path.empty(), options.source,
      std::vector<VertexId>(), dawnGetCpuThreadCount(handle));
  std::string output = options.output_path;
  RunResult run = BFS_CPU::run(execution_graph, output);
  if (!run.success) {
    return Status(StatusCode::kInvalidArgument, run.error_msg);
  }
  return FromRunResult(run, kernel.value(), selected.backend);
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

  StatusOr<Adapters::KernelCsrGraph> kernel_graph = MakeKernelCsr(graph);
  if (!kernel_graph.ok()) {
    return kernel_graph.status();
  }
  Graph::Graph_t execution_graph = MakeKernelGraph(
      graph, kernel_graph.value(), !options.output_path.empty(), options.source,
      std::vector<VertexId>(), dawnGetCpuThreadCount(handle));
  std::string output = options.output_path;
  RunResult run = SSSP_CPU::run(execution_graph, output);
  if (!run.success) {
    return Status(StatusCode::kInvalidArgument, run.error_msg);
  }
  return FromRunResult(run, kernel.value(), selected.backend);
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

  StatusOr<Adapters::KernelCsrGraph> kernel_graph = MakeKernelCsr(graph);
  if (!kernel_graph.ok()) {
    return kernel_graph.status();
  }
  const bool save_output =
      !options.output_path.empty() && options.output_source >= 0;
  std::vector<VertexId> execution_sources =
      save_output ? SingleOutputSource(options.output_source) : options.sources;
  Graph::Graph_t execution_graph = MakeKernelGraph(
      graph, kernel_graph.value(), save_output, execution_sources.front(),
      execution_sources, dawnGetCpuThreadCount(handle));
  std::string output = options.output_path;
  RunResult run = MSSP_CPU::run(execution_graph, output);
  if (!run.success) {
    return Status(StatusCode::kInvalidArgument, run.error_msg);
  }
  return FromRunResult(run, kernel.value(), selected.backend);
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

  StatusOr<Adapters::KernelCsrGraph> kernel_graph = MakeKernelCsr(graph);
  if (!kernel_graph.ok()) {
    return kernel_graph.status();
  }
  std::string output = options.output_path;
  RunResult run = RunResult::ok(0.0f);
  if (!output.empty() && options.output_source >= 0) {
    Graph::Graph_t execution_graph = MakeKernelGraph(
        graph, kernel_graph.value(), true, options.output_source,
        std::vector<VertexId>(1, options.output_source),
        dawnGetCpuThreadCount(handle));
    run = graph.weighted() ? SSSP_CPU::run(execution_graph, output)
                           : BFS_CPU::run(execution_graph, output);
  } else {
    Graph::Graph_t execution_graph =
        MakeKernelGraph(graph, kernel_graph.value(), false, 0,
                        std::vector<VertexId>(), dawnGetCpuThreadCount(handle));
    run = APSP_CPU::run(execution_graph, output);
  }
  if (!run.success) {
    return Status(StatusCode::kInvalidArgument, run.error_msg);
  }
  return FromRunResult(run, kernel.value(), selected.backend);
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

  StatusOr<Adapters::KernelCsrGraph> kernel_graph = MakeKernelCsr(graph);
  if (!kernel_graph.ok()) {
    return kernel_graph.status();
  }
  Graph::Graph_t execution_graph =
      MakeKernelGraph(graph, kernel_graph.value(), false, options.source,
                      std::vector<VertexId>(), dawnGetCpuThreadCount(handle));
  RunResult run = graph.weighted()
                      ? CC_CPU::run_Weighted(execution_graph, options.source)
                      : CC_CPU::run(execution_graph, options.source);
  if (!run.success) {
    return Status(StatusCode::kInvalidArgument, run.error_msg);
  }
  return FromRunResult(run, kernel.value(), selected.backend);
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

  StatusOr<Adapters::KernelCsrGraph> kernel_graph = MakeKernelCsr(graph);
  if (!kernel_graph.ok()) {
    return kernel_graph.status();
  }
  Graph::Graph_t execution_graph = MakeKernelGraph(
      graph, kernel_graph.value(), !options.output_path.empty(), 0,
      std::vector<VertexId>(), dawnGetCpuThreadCount(handle));
  std::string output = options.output_path;
  RunResult run = BC_CPU::Betweenness_Centrality(execution_graph, output);
  if (!run.success) {
    return Status(StatusCode::kInvalidArgument, run.error_msg);
  }
  return FromRunResult(run, kernel.value(), selected.backend);
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
