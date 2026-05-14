#pragma once

#include <dawn/common/status.hxx>
#include <dawn/common/types.hxx>
#include <dawn/graph/csr_graph.hxx>
#include <dawn/result.hxx>

#include <string>
#include <vector>

namespace DAWN {
namespace Algorithms {
namespace Internal {

StatusOr<RunResult> RunCpuBfs(const HostCsrGraph& graph,
                              VertexId source,
                              const std::string& output_path,
                              int thread_count);

StatusOr<RunResult> RunCpuSssp(const HostCsrGraph& graph,
                               VertexId source,
                               const std::string& output_path,
                               int thread_count);

StatusOr<RunResult> RunCpuShortestPaths(const HostCsrGraph& graph,
                                        VertexId source,
                                        const std::string& output_path,
                                        int thread_count);

StatusOr<RunResult> RunCpuMssp(const HostCsrGraph& graph,
                               const std::vector<VertexId>& sources,
                               const std::string& output_path,
                               int thread_count);

StatusOr<RunResult> RunCpuApsp(const HostCsrGraph& graph,
                               const std::string& output_path,
                               int thread_count);

StatusOr<RunResult> RunCpuCc(const HostCsrGraph& graph,
                             VertexId source,
                             int thread_count);

StatusOr<RunResult> RunCpuBc(const HostCsrGraph& graph,
                             const std::string& output_path,
                             int thread_count);

}  // namespace Internal
}  // namespace Algorithms
}  // namespace DAWN
