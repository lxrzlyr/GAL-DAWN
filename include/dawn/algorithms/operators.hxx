#pragma once

#include <string>
#include <vector>

#include <dawn/common/status.hxx>
#include <dawn/common/types.hxx>
#include <dawn/graph/csr_graph.hxx>
#include <dawn/runtime/handle.hxx>
#include <dawn/runtime/kernel_registry.hxx>

namespace DAWN {
namespace Algorithms {

struct AlgorithmResult {
  double elapsed_seconds = 0.0;
  double value = 0.0;
  std::string kernel_name;
  BackendKind backend = BackendKind::kCpu;
};

struct BfsOptions {
  VertexId source = 0;
  std::string output_path;
  std::string kernel_name;
};

struct SsspOptions {
  VertexId source = 0;
  std::string output_path;
  std::string kernel_name;
};

struct MsspOptions {
  std::vector<VertexId> sources;
  std::string output_path;
  std::string kernel_name;
  VertexId output_source = -1;
};

struct ApspOptions {
  std::string output_path;
  std::string kernel_name;
  VertexId output_source = -1;
};

struct CcOptions {
  VertexId source = 0;
  std::string kernel_name;
};

struct BcOptions {
  std::string output_path;
  std::string kernel_name;
};

using BfsResult = AlgorithmResult;
using SsspResult = AlgorithmResult;
using MsspResult = AlgorithmResult;
using ApspResult = AlgorithmResult;
using CcResult = AlgorithmResult;
using BcResult = AlgorithmResult;

StatusOr<BfsResult> RunBfs(const HostCsrGraph& graph,
                           const BfsOptions& options,
                           dawnHandle_t handle);
StatusOr<SsspResult> RunSssp(const HostCsrGraph& graph,
                             const SsspOptions& options,
                             dawnHandle_t handle);
StatusOr<MsspResult> RunMssp(const HostCsrGraph& graph,
                             const MsspOptions& options,
                             dawnHandle_t handle);
StatusOr<ApspResult> RunApsp(const HostCsrGraph& graph,
                             const ApspOptions& options,
                             dawnHandle_t handle);
StatusOr<CcResult> RunCc(const HostCsrGraph& graph,
                         const CcOptions& options,
                         dawnHandle_t handle);
StatusOr<BcResult> RunBc(const HostCsrGraph& graph,
                         const BcOptions& options,
                         dawnHandle_t handle);

StatusOr<AlgorithmResult> RunAlgorithm(const std::string& algorithm,
                                       const HostCsrGraph& graph,
                                       VertexId source,
                                       const std::vector<VertexId>& sources,
                                       const std::string& output_path,
                                       const std::string& kernel_name,
                                       VertexId output_source,
                                       dawnHandle_t handle);

}  // namespace Algorithms
}  // namespace DAWN
